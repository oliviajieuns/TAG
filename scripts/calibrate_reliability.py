"""Clean-reference statistic + calibrated sigmoid scale s.

This is the script that PRODUCES the artifact the calibration chain
expects — without it that chain had no emitter (adversarial review
2026-08, critical). Two modes, one per score design:

``--mode mvf`` (plan §2.1) computes, at the BASE checkpoint,

    ΔL_i = L(y_i | x_i^-) - L(y_i | x_i)                       [nats]

and saves ``{"delta": tensor}`` for ``tads.mvf.reliability_ref_file``.

``--mode tag`` (paper Eqs. 3-6) computes the TAG gate's own statistic

    Δ̂_i = min(Δ̄_i, Δ^min_i)                                    [ratio]

— which needs a TOKEN-LEVEL forward, because Δ^min_i is a span
aggregate — and saves ``{"delta_hat": tensor, ...}`` for
``tads.tag.gate_ref_file``.

The two artifacts are NOT interchangeable: one is a difference in nats,
the other a scale-free ratio in (-inf, 1]. Feeding an MVF reference to
the TAG gate would calibrate s against a quantity an order of magnitude
larger and effectively disable the gate, so both loaders hard-error on the
wrong key rather than accept a bare tensor of the wrong kind.

Both modes report the calibrated scale s = P{pct}(stat)/logit(q).

Usage (GPU server, once per backbone):
    python scripts/make_corrupted_pool.py --input alpaca_gpt4.json \
        --out-dir pools/clean_ref --emit-counterfactual --seed 42
    # (no --preset / fraction flags → pool stays clean; only the
    #  counterfactual pairing is emitted)

    # TAG:
    python scripts/calibrate_reliability.py --mode tag \
        --config configs/experiments/lowq/light_tag_05b.yaml \
        --pool pools/clean_ref/pool.json \
        --counterfactual pools/clean_ref/counterfactual.json \
        --out pools/clean_ref/delta_hat_05b.pt
    export TADS_GATE_REF=pools/clean_ref/delta_hat_05b.pt

    # MVF:
    python scripts/calibrate_reliability.py --mode mvf \
        --config configs/experiments/lowq/light_tads_mvf_05b.yaml \
        --pool pools/clean_ref/pool.json \
        --counterfactual pools/clean_ref/counterfactual.json \
        --out pools/clean_ref/delta_05b.pt
    export TADS_RELIABILITY_REF=pools/clean_ref/delta_05b.pt

Or pin the printed scale explicitly (TADS_GATE_SCALE / TADS_RELIABILITY_SCALE).
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
    p.add_argument("--out", required=True,
                   help="output .pt ({'delta': ...} for mvf, "
                        "{'delta_hat': ...} for tag)")
    p.add_argument("--mode", choices=("mvf", "tag"), default=None,
                   help="which statistic to emit. Default: inferred from the "
                        "config (tads.score_mode), falling back to 'mvf'.")
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
    _tads_cfg = cfg.get("tads", {}) or {}
    mode = args.mode or (
        "tag" if str(_tads_cfg.get("score_mode", "")) == "tag" else "mvf"
    )
    logger.info("Calibration mode: %s", mode)
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

    datasets = {}
    for tag, path, sub in (
        ("orig", args.pool, "pool"),
        ("cf", args.counterfactual, "counterfactual"),
    ):
        datasets[tag] = build_alpaca_dataset(
            tokenizer=tokenizer,
            cache_dir=str(Path(cache_dir) / sub),
            max_seq_len=int(cfg["max_seq_len"]),
            dataset_name=None,
            data_files=str(path),
            prompt_style=style,
        )
    if len(datasets["orig"]) != len(datasets["cf"]):
        sys.exit(
            f"pool/counterfactual size mismatch: {len(datasets['orig'])} vs "
            f"{len(datasets['cf'])} — pools must be index-aligned."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if mode == "tag":
        from tads.core import gate as gatelib

        tag_cfg = _tads_cfg.get("tag", {}) or {}
        gcfg = gatelib.GateConfig(
            span_tokens=int(tag_cfg.get("span_tokens", 16)),
            tau=float(tag_cfg.get("tau", 0.5)),
            tau_mode=str(tag_cfg.get("tau_mode", "per_token")),
            min_span_tokens=int(tag_cfg.get("min_span_tokens", 4)),
            tail_mode=str(tag_cfg.get("tail_mode", "min")),
            tail_quantile=float(tag_cfg.get("tail_quantile", 0.0)),
            include_eos=bool(tag_cfg.get("include_eos", False)),
            c_trunc=float(tag_cfg.get("c_trunc", 0.2)),
            eps_den=float(tag_cfg.get("eps_den", 1e-3)),
            min_common_tokens=int(tag_cfg.get("min_common_tokens", 8)),
            undefined_policy=str(tag_cfg.get("undefined_policy", "neutral")),
            undefined_gate_value=float(tag_cfg.get("undefined_gate_value", 0.6)),
            # The scale is what we are about to derive.
            scale=1.0,
        )
        tok, n_tok = {}, {}
        for tag in ("orig", "cf"):
            tok[tag], n_tok[tag] = gatelib.compute_pool_token_losses(
                model, datasets[tag], batch_size=bs, device=device, tag=tag,
                eos_token_id=tokenizer.eos_token_id,
                drop_trailing_eos=not gcfg.include_eos,
            )
        comp = gatelib.gate_components(
            tok["orig"], n_tok["orig"], tok["cf"], n_tok["cf"], cfg=gcfg,
        )
        stat = comp["delta_hat"]
        s = gatelib.calibrate_gate_scale(
            stat, target_pct=args.target_pct, target_q=args.target_q,
        )
        torch.save(
            {
                "delta_hat": stat.cpu(),
                "delta_bar": comp["delta_bar"].cpu(),
                "delta_min": comp["delta_min"].cpu(),
                "scale": s,
                "target_pct": args.target_pct,
                "target_q": args.target_q,
                "gate_config": gcfg.identity(),
                "model_path": str(cfg["model_path"]),
                "pool": str(args.pool),
                "counterfactual": str(args.counterfactual),
            },
            out,
        )
        frac_pos = float((stat > 0).float().mean().item())
        logger.info(
            "Saved clean-reference Δ̂ (n=%d, %.1f%% positive; Δ̄ mean %.4f, "
            "Δ^min mean %.4f) to %s | calibrated s = %.6f  →  "
            "export TADS_GATE_SCALE=%.6f",
            stat.numel(), 100.0 * frac_pos,
            float(comp["delta_bar"].mean().item()),
            float(comp["delta_min"].mean().item()),
            out, s, s,
        )
        logger.info(
            "Gate config used for calibration MUST match the training config "
            "(W=%d, tau=%.3f/%s, min_span=%d, tail=%s, include_eos=%s) — a "
            "different span partition yields a different Δ̂ distribution and "
            "silently mis-scales the gate.",
            gcfg.span_tokens, gcfg.tau, gcfg.tau_mode, gcfg.min_span_tokens,
            gcfg.tail_mode, gcfg.include_eos,
        )
        stat_name = "Δ̂"
    else:
        losses = {
            tag: compute_pool_loss(
                model, datasets[tag], batch_size=bs, device=device, tag=tag,
            )
            for tag in ("orig", "cf")
        }
        delta = losses["cf"] - losses["orig"]
        s = calibrate_reliability_scale(
            delta, target_pct=args.target_pct, target_q=args.target_q,
        )
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
        stat_name = "ΔL"

    if frac_pos < 0.8:
        logger.warning(
            "Only %.1f%% of the 'clean' reference has %s > 0 — the reference "
            "pool looks contaminated or the counterfactuals are not truly "
            "unrelated. Inspect before trusting this calibration.",
            100.0 * frac_pos, stat_name,
        )


if __name__ == "__main__":
    main()
