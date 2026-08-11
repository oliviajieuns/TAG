"""Clean-reference ΔL + calibrated sigmoid scale s (plan §2.1, D3 step).

This is the script that PRODUCES the artifact `reliability_ref_file`
expects — without it the calibration chain had no emitter (adversarial
review 2026-08, critical): compute, at the BASE checkpoint,

    ΔL_i = L(y_i | x_i^-) - L(y_i | x_i)

over a CLEAN reference pool (clean Alpaca-GPT4 + its own counterfactual
pool from `make_corrupted_pool.py --emit-counterfactual` on the clean
input), save {"delta": tensor} for `tads.mvf.reliability_ref_file`, and
print the derived scale s = P{pct}(ΔL)/logit(q).

Usage (GPU server, once per backbone):
    python scripts/make_corrupted_pool.py --input alpaca_gpt4.json \
        --out-dir pools/clean_ref --emit-counterfactual --seed 42
    # (no --preset / fraction flags → pool stays clean; only the
    #  counterfactual pairing is emitted)
    python scripts/calibrate_reliability.py \
        --config configs/experiments/lowq/light_tads_mvf_05b.yaml \
        --pool pools/clean_ref/pool.json \
        --counterfactual pools/clean_ref/counterfactual.json \
        --out pools/clean_ref/delta_05b.pt

Then either:
    export TADS_RELIABILITY_REF=pools/clean_ref/delta_05b.pt   # calibrate at load
or pin the printed scale explicitly:
    export TADS_RELIABILITY_SCALE=<printed s>
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="experiment YAML (model/tokenizer)")
    p.add_argument("--pool", required=True, help="CLEAN pool json")
    p.add_argument("--counterfactual", required=True,
                   help="counterfactual json for the CLEAN pool (index-aligned)")
    p.add_argument("--out", required=True, help="output .pt ({'delta': tensor})")
    p.add_argument("--target-pct", type=float, default=0.10)
    p.add_argument("--target-q", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch

    from tads.core.reliability import calibrate_reliability_scale, compute_pool_loss
    from tads.core.utils import load_config
    from tads.data.alpaca import build_alpaca_dataset
    from tads.modeling.loader import load_model, load_tokenizer

    cfg = load_config(args.config)
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
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()

    cache_dir = str(Path(cfg["data_cache"]) / "calibrate_reliability")
    bs = int(args.batch_size or cfg.get("episode_batch_size", 1))
    style = str(cfg.get("prompt_style") or "alpaca_default")

    losses = {}
    for tag, path, sub in (
        ("orig", args.pool, "pool"),
        ("cf", args.counterfactual, "counterfactual"),
    ):
        ds = build_alpaca_dataset(
            tokenizer=tokenizer,
            cache_dir=str(Path(cache_dir) / sub),
            max_seq_len=int(cfg["max_seq_len"]),
            dataset_name=None,
            data_files=str(path),
            prompt_style=style,
        )
        losses[tag] = compute_pool_loss(model, ds, batch_size=bs, device=device, tag=tag)
    if losses["orig"].numel() != losses["cf"].numel():
        sys.exit(
            f"pool/counterfactual size mismatch: {losses['orig'].numel()} vs "
            f"{losses['cf'].numel()} — pools must be index-aligned."
        )

    delta = losses["cf"] - losses["orig"]
    s = calibrate_reliability_scale(
        delta, target_pct=args.target_pct, target_q=args.target_q,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta": delta.cpu(),
            "scale": s,
            "target_pct": args.target_pct,
            "target_q": args.target_q,
            "model_path": str(cfg["model_path"]),
            "pool": str(args.pool),
            "counterfactual": str(args.counterfactual),
        },
        out,
    )
    frac_pos = float((delta > 0).float().mean().item())
    logger.info(
        "Saved clean-reference ΔL (n=%d, %.1f%% positive) to %s | "
        "calibrated s = %.6f  →  export TADS_RELIABILITY_SCALE=%.6f",
        delta.numel(), 100.0 * frac_pos, out, s, s,
    )
    if frac_pos < 0.8:
        logger.warning(
            "Only %.1f%% of the 'clean' reference has ΔL > 0 — the reference "
            "pool looks contaminated or the counterfactuals are not truly "
            "unrelated. Inspect before trusting this calibration.",
            100.0 * frac_pos,
        )


if __name__ == "__main__":
    main()
