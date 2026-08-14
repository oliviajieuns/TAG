"""Response-only ("unconditioned") pool loss — one forward pass.

Produces L(y_i) per sample. One .pt file feeds three things at once
(plan §5.1 / §5.3):

  1. IFD baseline (Li et al., NAACL 2024): IFD_i = L(y|x) / L(y) — the
     closest published relative of our Q view; mandatory comparison row.
  2. Base-PPL filter row/arm: rank by L(y) alone — measures (rather than
     assumes away) the "a trivial perplexity filter already solves this
     benchmark" attack for each corruption type.
  3. Superfiltering variant: run this script with the 0.5B model on the 7B
     pool — weak-to-strong IFD row for free.

Two templates (``--template``, default ``direct``):

  * ``direct`` — the response text tokenised ALONE (add_special_tokens=True
    so the model's BOS is present; labels = the response tokens, with only
    the BOS masked to -100). This is the IFD paper's direct-answer score
    s(y): L(y) conditioned on nothing but sequence start. Use this for the
    reported IFD rows.
  * ``alpaca`` — the previous behaviour, kept as an ablation: the response
    under the full SFT template with the instruction BLANKED. The loss is
    then still conditioned on the template scaffold ("### Instruction:" /
    "### Response:" etc.), so it measures template-conditioned fluency,
    NOT the paper's s(y).

Truncation caveat (both templates): when a response overruns
``max_seq_len``, the CONDITIONED pass loses additional response tokens to
the prompt (the instruction eats sequence budget), so conditioned and
unconditioned losses average over different response-token subsets for
truncated samples. The IFD ratio is exact only for samples whose full
response fits in both passes; keep max_seq_len identical across the two
passes to minimise the divergence.

Usage (GPU server):
    python scripts/compute_uncond_loss.py \
        --config configs/experiments/lowq/light_mvf_05b.yaml \
        --out pools/composite20/uncond_loss_05b.pt

The output tensor is index-aligned with ALPACA_DATA_FILES. Pass it to
scripts/score_pool.py --uncond-loss to add the ifd / ppl signal rows.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def build_response_only_dataset(tokenizer, *, cache_dir, max_seq_len, data_files,
                                prompt_style, template="direct"):
    """Tokenise the pool for the unconditioned pass.

    ``template="direct"`` (IFD paper's s(y)): the response text alone,
    encoded with add_special_tokens=True so the model's BOS (when it has
    one) is present. Labels are the response tokens; the "prompt" is just
    the BOS, masked to -100. Truncated to ``max_seq_len`` like the
    conditioned pass.

    ``template="alpaca"`` (ablation): the standard tokenisation path with
    instruction="" (and input=""), so special tokens, EOS handling, and the
    assistant-mask logic stay identical to the conditioned pass — but the
    loss remains conditioned on the template scaffold, measuring
    template-conditioned fluency rather than s(y).
    """
    import json

    from datasets import Dataset

    from tag.data.sft_prompts import tokenize_alpaca

    with open(data_files, encoding="utf-8") as f:
        records = json.load(f)
    blanked = [
        {"instruction": "", "input": "", "output": str(r.get("output", ""))}
        for r in records
    ]
    ds = Dataset.from_list(blanked)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def _tok_direct(example):
        ids = tokenizer(
            example["output"], add_special_tokens=True, return_tensors=None,
        )["input_ids"][:max_seq_len]
        labels = list(ids)
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and labels and labels[0] == bos_id:
            labels[0] = -100  # the whole "prompt" is the BOS token
        pad_len = max_seq_len - len(ids)
        return {
            "input_ids": ids + [pad_id] * pad_len,
            "attention_mask": [1] * len(ids) + [0] * pad_len,
            "labels": labels + [-100] * pad_len,
        }

    def _tok_alpaca(example):
        return tokenize_alpaca(
            example, tokenizer, max_seq_len=max_seq_len, prompt_style=prompt_style,
        )

    if template not in ("direct", "alpaca"):
        raise ValueError(f"unknown template={template!r}; expected direct|alpaca")
    ds = ds.map(_tok_direct if template == "direct" else _tok_alpaca,
                remove_columns=ds.column_names,
                desc=f"Tokenising response-only pool ({template})")
    ds.set_format("torch")
    return ds


def main() -> None:
    # Windows consoles default to legacy codepages (cp949 here) that cannot
    # encode this file's docstring (used as the --help text); force UTF-8
    # where the stream supports it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):  # pragma: no cover — exotic streams
                pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="experiment YAML (model/tokenizer/pool)")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--data-files", default=None,
                    help="override pool json (default: ALPACA_DATA_FILES / config)")
    ap.add_argument("--template", choices=("direct", "alpaca"), default="direct",
                    help="'direct' = response alone (IFD paper's s(y); "
                         "default); 'alpaca' = blanked-instruction SFT "
                         "template (template-conditioned fluency ablation)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Heavy imports deferred so --help works everywhere.
    import os

    import torch

    from tag.core.reliability import compute_pool_loss
    from tag.core.utils import load_config
    from tag.modeling.loader import load_model, load_tokenizer

    cfg = load_config(args.config)
    data_files = args.data_files or os.environ.get("ALPACA_DATA_FILES") or cfg.get("data_files")
    if not data_files:
        sys.exit("No pool: pass --data-files or set ALPACA_DATA_FILES.")

    tokenizer = load_tokenizer(cfg["model_path"])
    model = load_model(
        cfg["model_path"],
        training_mode="full",
        lora_cfg=None,
        use_ddp=False,
        local_rank=0,
        gradient_checkpointing=False,
        attn_implementation=cfg.get("attn_implementation"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_map-dispatched models manage their own placement and raise on
    # .to(); only move models that were loaded onto a single device.
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()

    ds = build_response_only_dataset(
        tokenizer,
        cache_dir=cfg.get("cache_dir"),
        max_seq_len=int(cfg["max_seq_len"]),
        data_files=str(data_files),
        prompt_style=str(cfg.get("prompt_style", "alpaca_default")),
        template=args.template,
    )
    loss = compute_pool_loss(
        model, ds,
        batch_size=int(args.batch_size or cfg.get("episode_batch_size", 1)),
        device=device,
        tag="uncond",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"uncond_loss": loss.cpu(), "n": loss.numel(),
                "data_files": str(data_files), "template": args.template}, out)
    logger.info("Saved unconditioned loss (n=%d, template=%s) to %s",
                loss.numel(), args.template, out)


if __name__ == "__main__":
    main()
