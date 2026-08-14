"""Clean-reference statistic + calibrated sigmoid scale s.

This is the script that PRODUCES the artifact the calibration chain
expects — without it that chain had no emitter (adversarial review
2026-08, critical). Two modes, one per score design:

``--mode mvf`` (plan §2.1) computes, at the BASE checkpoint,

    ΔL_i = L(y_i | x_i^-) - L(y_i | x_i)                       [nats]

and saves ``{"delta": tensor}`` for ``selection.mvf.reliability_ref_file``.

``--mode tag`` (paper Eqs. 3-6) computes the TAG gate's own statistic

    Δ̂_i = min(Δ̄_i, Δ^min_i)                                    [ratio]

— which needs a TOKEN-LEVEL forward, because Δ^min_i is a span
aggregate — and saves ``{"delta_hat": tensor, ...}`` for
``selection.tag.gate_ref_file``.

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
    export TAG_GATE_REF=pools/clean_ref/delta_hat_05b.pt

    # MVF:
    python scripts/calibrate_reliability.py --mode mvf \
        --config configs/experiments/lowq/light_mvf_05b.yaml \
        --pool pools/clean_ref/pool.json \
        --counterfactual pools/clean_ref/counterfactual.json \
        --out pools/clean_ref/delta_05b.pt
    export TAG_RELIABILITY_REF=pools/clean_ref/delta_05b.pt

Or pin the printed scale explicitly (TAG_GATE_SCALE / TAG_RELIABILITY_SCALE).
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
                        "config (selection.score_mode), falling back to 'mvf'.")
    p.add_argument("--target-pct", type=float, default=0.10)
    p.add_argument("--target-q", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--target-zero-rate", type=float, default=None,
                   help="tag mode: clean-reference zero-weight rate for the Eq. 5' "
                        "null correction. Default: selection.tag.target_zero_rate from "
                        "the config (0.05). Must be < --target-pct.")
    p.add_argument("--no-null-correction", action="store_true",
                   help="tag mode: fit s on the RAW Delta_hat, the literal "
                        "Eq. 5. This is the ablation arm — on the 7B backbone "
                        "it sends ~60%% of clean data to zero weight.")
    p.add_argument("--no-token-losses", action="store_true",
                   help="tag mode: omit the per-token NLLs from the artifact. "
                        "Saves disk, but makes the W/tau sweep cost a full "
                        "re-forward per sweep point.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch

    from tag.core.reliability import calibrate_reliability_scale, compute_pool_loss
    from tag.core.utils import load_config
    from tag.data.alpaca import build_alpaca_dataset
    from tag.modeling.loader import load_model, load_tokenizer

    cfg = load_config(args.config)
    _selection_cfg = cfg.get("selection", {}) or {}
    mode = args.mode or (
        "tag" if str(_selection_cfg.get("score_mode", "")) == "tag" else "mvf"
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
        from tag.core import gate as gatelib

        tag_cfg = _selection_cfg.get("tag", {}) or {}
        target_zero_rate = (
            float(args.target_zero_rate)
            if args.target_zero_rate is not None
            else float(tag_cfg.get("target_zero_rate", 0.05))
        )
        fit_null = not args.no_null_correction
        if fit_null and target_zero_rate >= args.target_pct:
            sys.exit(
                f"--target-zero-rate ({target_zero_rate}) must be strictly below "
                f"--target-pct ({args.target_pct}): the Eq. 5' centring puts "
                f"the target_zero_rate quantile at exactly 0, so the scale "
                f"calibration would derive s from a non-positive quantile."
            )
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
            # Both halves of the calibration are what we are about to derive,
            # so the components pass runs uncorrected and unit-scaled.
            null_correction=False,
            target_zero_rate=target_zero_rate,
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
        # gcfg above ran uncorrected, so comp["delta_hat"] IS the raw
        # min(Delta_bar, Delta_min) the null curve must be fit on.
        raw = comp["delta_hat"]
        fit = gatelib.fit_calibration(
            raw, comp["n_spans"],
            span_tokens=gcfg.span_tokens,
            target_zero_rate=target_zero_rate,
            target_pct=args.target_pct,
            target_q=args.target_q,
            null_correction=fit_null,
        )
        stat, s = fit["delta_hat"], fit["scale"]
        payload = {
                # delta_hat is the RAW statistic: it is what a later refit
                # (a different W, a different target_zero_rate) needs, and the
                # centred version is one lookup away given "null".
                "delta_hat": raw.cpu(),
                "delta_hat_centered": stat.cpu(),
                "delta_bar": comp["delta_bar"].cpu(),
                "delta_min": comp["delta_min"].cpu(),
                "n_spans": comp["n_spans"].cpu(),
                "n_common": comp["n_common"].cpu(),
                "null": fit["null"].to_dict() if fit["null"] is not None else None,
                "scale": s,
                "target_pct": args.target_pct,
                "target_q": args.target_q,
                "target_zero_rate": target_zero_rate,
                "gate_config": gcfg.identity(),
                "model_path": str(cfg["model_path"]),
                "pool": str(args.pool),
                "counterfactual": str(args.counterfactual),
        }
        if not args.no_token_losses:
            # Keeping the per-token NLLs (fp16) makes the span-width sweep
            # FREE: Delta_hat can be re-derived for any W / tau / tail_mode
            # with no forward pass. Two 14-minute pool passes per sweep point
            # is otherwise the difference between choosing W on evidence and
            # guessing it. Costs roughly 2 x n x T_max bytes.
            payload["token_true"] = tok["orig"].to(torch.float16)
            payload["n_true"] = n_tok["orig"]
            payload["token_cf"] = [tok["cf"].to(torch.float16)]
            payload["n_cf"] = [n_tok["cf"]]
        torch.save(payload, out)
        logger.info(
            "Saved clean reference (n=%d) to %s", raw.numel(), out,
        )
        raw_pos = float((raw > 0).float().mean().item())
        logger.info(
            "  RAW  Δ̂ = min(Δ̄, Δ^min):  %.1f%% positive | Δ̄ mean %+.4f | "
            "Δ^min mean %+.4f",
            100.0 * raw_pos,
            float(comp["delta_bar"].mean().item()),
            float(comp["delta_min"].mean().item()),
        )
        if raw_pos < 0.8 and float(comp["delta_bar"].mean().item()) > 0:
            # Diagnose the two failure modes apart. A healthy Δ̄ with a deeply
            # negative Δ^min is the order-statistic drift of Eq. 5, not a
            # dirty reference — the distinction decides whether the fix is
            # the null correction or regenerating the pool.
            logger.info(
                "  ^ Δ̄ is healthy but the RAW tail min is not: that is Eq. 5's "
                "order-statistic drift (min over M spans), not a contaminated "
                "reference. It is what the Eq. 5' null correction removes.",
            )
        if fit["null"] is not None:
            logger.info(
                "  Eq.5' null correction ON (target_zero_rate=%.3f): clean zero-weight "
                "rate %.1f%%, %.1f%% positive",
                target_zero_rate, 100.0 * fit["zero_rate"],
                100.0 * fit["frac_positive"],
            )
            # Length-uniformity is the entire claim of the correction, so it
            # is printed rather than assumed. A bin whose rate drifts far from
            # the target means the curve is under-resolved there.
            logger.info("  per-bin clean zero-weight rate (should all be ~%.1f%%):",
                        100.0 * target_zero_rate)
            for row in fit["report"]:
                logger.info(
                    "    M in [%d, %d] | n=%6d | mu=%+.4f | zero=%.1f%%",
                    int(row["m_lo"]), int(row["m_hi"]), int(row["n"]),
                    row["mu"], 100.0 * row["zero_rate"],
                )
        else:
            logger.warning(
                "  Eq.5' null correction OFF — this is the ablation arm. The "
                "clean zero-weight rate below is what the uncorrected Eq. 5 gives: "
                "%.1f%%.", 100.0 * fit["zero_rate"],
            )
        logger.info("  calibrated s = %.6f  →  export TAG_GATE_SCALE=%.6f", s, s)
        logger.info(
            "Gate config used for calibration MUST match the training config "
            "(W=%d, tau=%.3f/%s, min_span=%d, tail=%s, include_eos=%s, "
            "target_zero_rate=%.3f) — a different span partition yields a different "
            "Δ̂ distribution and silently mis-scales the gate.",
            gcfg.span_tokens, gcfg.tau, gcfg.tau_mode, gcfg.min_span_tokens,
            gcfg.tail_mode, gcfg.include_eos, target_zero_rate,
        )
        # The generic contamination check below judges what Eq. 6 will
        # actually see, which is the CENTRED statistic when the correction is
        # on. With it off, this is the raw fraction and the check fires — as
        # it should, since that arm really does zero most of a clean pool.
        frac_pos = fit["frac_positive"]
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
            "calibrated s = %.6f  →  export TAG_RELIABILITY_SCALE=%.6f",
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
