"""Tokenizer and model loading for training and evaluation.

Unifies four scenarios behind one entry point:
    single-GPU / DDP   ×   full FT / LoRA
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.utils import is_main_process

# PEFT and the build_lora_config helper are intentionally lazy-imported below
# (inside the LoRA branches of load_model / load_for_eval). PEFT version
# mismatches with the installed transformers/torch should never block full-FT
# runs that don't use LoRA at all.

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- path
def _resolve_local_path(path: str) -> str:
    """Tolerate case differences between configured path and on-disk dir name.

    Linux is case-sensitive, so a config that says ``qwen2.5-7b`` won't open
    an on-disk ``Qwen2.5-7B`` directory. HF/cluster naming conventions vary
    (Mistral-7B-v0.1 vs mistral-7b-v0.1, DeepSeek-LLM-7B-Base vs deepseek-
    llm-7b-base, Qwen2.5-7B vs qwen2.5-7b, …) and the user's setup_env.sh
    and YAML defaults can drift. If the configured path doesn't exist, look
    for a single case-insensitive sibling in the parent directory and use
    that with a warning. This makes both naming conventions work without
    having to chase the exact case manually.
    """
    if os.path.exists(path):
        return path
    parent = os.path.dirname(path) or "."
    if not os.path.isdir(parent):
        return path  # parent missing too — let HF surface the real error
    name_lower = os.path.basename(path).lower()
    matches: List[str] = [
        entry for entry in os.listdir(parent)
        if entry.lower() == name_lower
    ]
    if len(matches) == 1:
        resolved = os.path.join(parent, matches[0])
        logger.warning(
            "Path %r not found; using case-variant %r instead.", path, resolved,
        )
        return resolved
    return path


# --------------------------------------------------------------------- tokenizer
def load_tokenizer(model_path: str):
    model_path = _resolve_local_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --------------------------------------------------------------------- model
def load_model(
    model_path: str,
    *,
    training_mode: str = "full",
    lora_cfg: Optional[Dict[str, Any]] = None,
    use_ddp: bool = False,
    local_rank: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    attn_implementation: Optional[str] = None,
):
    """Load a causal-LM model for training.

    Single-GPU path uses ``device_map='auto'``; DDP path puts the model on a
    single ``cuda:local_rank`` and wraps it with ``DistributedDataParallel``.

    ``attn_implementation`` is passed through to ``from_pretrained``. Accepted
    values include ``"sdpa"`` (PyTorch ≥ 2.0 efficient default),
    ``"flash_attention_2"`` (needs ``flash-attn`` installed; ~30% speedup),
    or ``"eager"``. ``None`` lets transformers pick.
    """
    model_path = _resolve_local_path(model_path)
    use_ddp = use_ddp and dist.is_initialized()

    kwargs: Dict[str, Any] = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not use_ddp:
        kwargs["device_map"] = "auto"
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ValueError, ImportError) as e:
        if attn_implementation == "flash_attention_2":
            logger.warning(
                "flash_attention_2 unavailable (%s); falling back to sdpa.", e,
            )
            kwargs["attn_implementation"] = "sdpa"
            base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            raise

    if gradient_checkpointing:
        # use_reentrant=False is required for the modern checkpointing path
        # to interact correctly with PEFT (LoRA) and with DDP. The legacy
        # reentrant=True path silently drops gradients for inputs that don't
        # require_grad and breaks under PEFT's parameter-injection scheme.
        # Older transformers may not accept the kwarg — fall back gracefully.
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            base.gradient_checkpointing_enable()
        # enable_input_require_grads is REQUIRED for LoRA (most base params
        # frozen → embedding output needs an explicit require_grad hook so
        # gradient_checkpointing has something to backprop through), but
        # for full-FT it adds an extra hook that DDP may not track,
        # showing up as "first SFT backward never completes". Skip it
        # outside LoRA. The pre-existing comment said "Required for LoRA"
        # — we now honour that literally.
        if training_mode == "lora":
            base.enable_input_require_grads()
        # gradient_checkpointing is fundamentally incompatible with KV cache:
        # the recomputation pass would replay forward without the cached
        # keys/values. transformers auto-overrides to False on every forward
        # and logs a warning each time — pin it once here so the warning
        # stops and the state stays consistent (some DDP gradient-hook bugs
        # have been linked to transformers flipping use_cache mid-forward).
        if hasattr(base, "config") and hasattr(base.config, "use_cache"):
            base.config.use_cache = False

    if training_mode == "lora":
        from peft import get_peft_model  # lazy: only imported for LoRA
        from .lora import build_lora_config
        config = build_lora_config(lora_cfg)
        model = get_peft_model(base, config)
        if is_main_process():
            model.print_trainable_parameters()
    elif training_mode == "full":
        model = base
        for param in model.parameters():
            param.requires_grad = True
        if is_main_process():
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "Full FT | total=%.1fM | trainable=%.1fM (%.1f%%)",
                n_total / 1e6, n_train / 1e6, 100.0 * n_train / max(1, n_total),
            )
    else:
        raise ValueError(
            f"Unknown training_mode={training_mode!r}; expected 'full' or 'lora'",
        )

    if use_ddp:
        model = model.to(f"cuda:{local_rank}")
        # DDP wrap parameters. Three things go wrong here without care:
        # 1. find_unused_parameters=False is "cheaper" but if a single
        #    parameter genuinely doesn't get a gradient on a given step
        #    (gradient_checkpointing rematerialisation, a conditional
        #    layer, a frozen embedding) DDP hangs forever waiting for it.
        #    We've seen "SFT step entry step=0 visible, step backward
        #    done never appears" with both nccl and gloo — symptoms
        #    consistent with this exact deadlock. Default to True for
        #    safety; opt back to False with TADS_DDP_FIND_UNUSED=0 once
        #    the run is verified stable.
        # 2. static_graph=True lets DDP optimise after the first
        #    forward+backward locks the graph in; recommended once safe.
        #    Off by default.
        # 3. broadcast_buffers=True (DDP default) syncs BN/running stats
        #    on every forward. Llama-2 has no BN, so we turn it off to
        #    save the collective. (One less place for the first
        #    post-idle all_reduce to land.)
        _env_fu = os.environ.get("TADS_DDP_FIND_UNUSED")
        if _env_fu is not None:
            find_unused = _env_fu == "1"
        else:
            # Default True — safest for both LoRA and full-FT.
            find_unused = True
        static_graph = os.environ.get("TADS_DDP_STATIC_GRAPH", "0") == "1"
        broadcast_buffers = (
            os.environ.get("TADS_DDP_BROADCAST_BUFFERS", "0") == "1"
        )
        if is_main_process():
            logger.info(
                "DDP wrap | find_unused_parameters=%s | static_graph=%s | "
                "broadcast_buffers=%s",
                find_unused, static_graph, broadcast_buffers,
            )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
            broadcast_buffers=broadcast_buffers,
            static_graph=static_graph,
        )
    return model


# --------------------------------------------------------------------- eval load
def load_for_eval(
    base_model: str,
    ckpt_dir: str,
    *,
    training_mode: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any, torch.device]:
    """Load a checkpoint for evaluation.

    If ``training_mode`` is None, auto-detect:
    LoRA when ``<ckpt_dir>/adapter_config.json`` exists, else full.
    """
    base_model = _resolve_local_path(base_model)
    ckpt_dir = _resolve_local_path(ckpt_dir)
    if training_mode is None:
        training_mode = (
            "lora" if (Path(ckpt_dir) / "adapter_config.json").exists() else "full"
        )

    logger.info(
        "load_for_eval | base=%s | ckpt=%s | mode=%s",
        base_model, ckpt_dir, training_mode,
    )
    tokenizer = load_tokenizer(base_model)

    if training_mode == "lora":
        from peft import PeftModel  # lazy: only imported for LoRA
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, ckpt_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    return model, tokenizer, dev


# --------------------------------------------------------------------- helpers
def get_hidden_size(model) -> int:
    """Return ``config.hidden_size`` regardless of DDP / PEFT wrapping."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model"):
        m = m.base_model
        if hasattr(m, "model"):
            m = m.model
    return int(m.config.hidden_size)
