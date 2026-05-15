"""LoRA configuration helpers.

YAML ``lora:`` section → ``peft.LoraConfig``. Accepts both ``alpha`` and
``lora_alpha`` keys; both ``dropout`` and ``lora_dropout``; ``r`` is required.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)


# Llama/Qwen/Mistral/DeepSeek 7B-family standard names. Phi-3 uses `qkv_proj`
# (fused), Gemma uses different names — DEFAULT works only for the canonical
# Llama-style architecture used in the NAIT paper's main matrix.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_lora_config(
    lora_cfg: Optional[Dict[str, Any]],
    *,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Build a ``LoraConfig`` from a YAML-shaped dict."""
    cfg = dict(lora_cfg or {})
    r = int(cfg.get("r", 16))
    alpha = int(cfg.get("alpha", cfg.get("lora_alpha", 32)))
    dropout = float(cfg.get("dropout", cfg.get("lora_dropout", 0.05)))
    # target_modules drift across architectures (Phi-3 fuses qkv, Gemma uses
    # different names). Surface a warning when we silently fall back to the
    # Llama-style default — that's the right pick for Llama-2 / Qwen2.5 /
    # Mistral / DeepSeek but a foot-gun for anything else.
    if "target_modules" not in cfg:
        target_modules = list(DEFAULT_TARGET_MODULES)
        logger.warning(
            "lora.target_modules unset in config; falling back to Llama-style "
            "default %s. If you're training a non-Llama architecture (Phi-3, "
            "Gemma, Mamba, ...), set lora.target_modules in the YAML.",
            target_modules,
        )
    else:
        target_modules = list(cfg["target_modules"])
    bias = cfg.get("bias", "none")
    if r > 0 and alpha / max(r, 1) > 64:
        # The LoRA scale is alpha / r; ratios above ~8 are unusual and ratios
        # above 64 typically indicate the user mis-typed one of the two.
        logger.warning(
            "lora.alpha/r = %d/%d = %.1f is unusually large; if updates "
            "diverge, lower alpha or raise r.",
            alpha, r, alpha / r,
        )
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
