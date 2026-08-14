#!/usr/bin/env python
"""Choose W (and tau, and the tail statistic) from a saved calibration — no GPU.

    python scripts/sweep_gate_config.py --ref pools/clean_ref/delta_hat_7b.pt

    # once you have picked one, write the new reference without touching a GPU:
    python scripts/sweep_gate_config.py --ref pools/clean_ref/delta_hat_7b.pt \\
        --span-tokens 32 --refit-out pools/clean_ref/delta_hat_7b_W32.pt

The span width is a first-order hyper-parameter, not a detail. Delta^min is a
minimum over M = ceil(n/W) spans, so its null drifts downward as responses get
longer, while per-span noise shrinks like 1/sqrt(W). Picking W by re-running
the calibration costs two full pool forwards per candidate; re-deriving it from
the cached per-token NLLs costs seconds.

The Eq. 5' null correction changes what this table is FOR. It pins the clean
zero-weight rate at ``--target-zero-rate`` in every length bin by construction, so "%pos"
and "zero%" are no longer the discriminating columns — they are the same for
every W that fits at all. What still differs across W is how much correction
was needed and how well-separated the corrected statistic is:

  mu-drift  mu(M_max) - mu(M_min): how far the raw null moves across the length
            range. This is the pathology the correction absorbs. Large is not
            fatal, but a W where it is small has a statistic that behaves
            itself natively, and the correction is then a small adjustment
            rather than the thing holding the gate up.
  sep       median(Delta_hat_c) / IQR(Delta_hat_c) on the CLEAN reference: how
            far typical clean data sits above zero relative to its
            own scatter. Higher means a corrupted sample needs a smaller true
            effect to be pushed across zero, i.e. more detection headroom.
            It is a PROXY — it is computed without any dirty labels. Confirm
            the winner on a labelled pool with scripts/score_pool.py before
            committing to it.
  bias      zero-weight rate in the longest response quintile over the shortest.
            Should be ~1.0 once the correction is on; it is printed as a CHECK
            on the correction, not as a criterion to optimise.
  raw%pos   fraction of the clean reference with UNCORRECTED Delta_hat > 0 —
            what the literal Eq. 5 would have passed. Reported so the ablation
            arm's number is visible next to the corrected one.

With --no-null-correction the table reverts to the uncorrected reading, where
%pos IS the hard ceiling on clean survival at any scale s.

The reference must have been written WITH token losses (the default since
calibrate_reliability.py grew --no-token-losses).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", required=True,
                   help="delta_hat_*.pt from calibrate_reliability.py --mode tag")
    p.add_argument("--span-tokens", default="8,16,32,64,128",
                   help="comma-separated W values to try")
    p.add_argument("--prefix-tokens", type=int, default=None,
                   help="restrict the sequence-level contrast to the first\n"
                        "N response tokens (0/unset = whole sequence)")
    p.add_argument("--tau", default=None,
                   help="comma-separated tau values (default: the ref's own)")
    p.add_argument("--tail-modes", default="min",
                   help="comma-separated: min,quantile")
    p.add_argument("--tail-quantile", type=float, default=0.25)
    p.add_argument("--target-pct", type=float, default=0.10)
    p.add_argument("--target-q", type=float, default=0.8)
    p.add_argument("--target-zero-rate", type=float, default=0.05,
                   help="clean-reference zero-weight rate the Eq. 5' correction pins")
    p.add_argument("--no-null-correction", action="store_true",
                   help="sweep the uncorrected Eq. 5 instead (ablation)")
    p.add_argument("--refit-out", default=None,
                   help="write a calibration artifact for the single swept "
                        "configuration to this path (requires exactly one W, "
                        "one tau and one tail mode). No GPU — the token "
                        "losses in --ref are all it needs.")
    args = p.parse_args()

    import torch
    from tag.core.gate import GateConfig, fit_calibration, gate_components

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    if "token_true" not in ref:
        sys.exit(
            f"{args.ref} has no per-token losses, so nothing can be re-derived "
            f"without a forward pass.\nRegenerate it (the default now keeps "
            f"them):\n  bash scripts/gpu_cloud/bootstrap.sh calibrate7b"
        )
    use_null = not args.no_null_correction
    if use_null and not (args.target_zero_rate < args.target_pct):
        sys.exit(
            f"--target-zero-rate ({args.target_zero_rate}) must be strictly below "
            f"--target-pct ({args.target_pct}); the correction puts the "
            f"target_zero_rate quantile at exactly 0."
        )

    tok_true = ref["token_true"].float()
    n_true = ref["n_true"]
    tok_cf = ref["token_cf"][0].float()
    n_cf = ref["n_cf"][0]
    base = ref.get("gate_config") or {}
    n = tok_true.size(0)
    print(f"reference : {args.ref}")
    print(f"samples   : {n}")
    print(f"as-built  : W={base.get('span_tokens')} tau={base.get('tau')}"
          f"({base.get('tau_mode')}) tail={base.get('tail_mode')}")
    print(f"correction: {'Eq.5-prime null, target_zero_rate=%.3f' % args.target_zero_rate if use_null else 'OFF (literal Eq. 5)'}")
    print()

    resp_len = torch.minimum(n_true.long(), n_cf.long()).float()
    order = torch.argsort(resp_len)
    quint = torch.tensor_split(order, 5)

    Ws = [int(x) for x in args.span_tokens.split(",") if x.strip()]
    taus = ([float(x) for x in args.tau.split(",")] if args.tau
            else [float(base.get("tau", 0.5))])
    tails = [t.strip() for t in args.tail_modes.split(",") if t.strip()]

    if args.refit_out and (len(Ws) != 1 or len(taus) != 1 or len(tails) != 1):
        sys.exit(
            "--refit-out writes ONE artifact, so it needs exactly one --span-"
            f"tokens, one --tau and one --tail-modes (got {len(Ws)}/"
            f"{len(taus)}/{len(tails)}). Sweep first, then refit the winner."
        )

    print(f"{'W':>5} {'tau':>5} {'tail':>9} {'raw%pos':>8} {'mu-drift':>9} "
          f"{'s':>9} {'zero%':>7} {'bias':>6} {'sep':>6}  {'note':<30}")
    print("-" * 108)
    best = None
    winner = None
    # With a single swept point there is nothing to choose between, so
    # --refit-out writes THAT point whatever its note says. That is what makes
    # it usable for the deliberately-bad ablation arm (--no-null-correction),
    # whose whole purpose is to capture a configuration that would never win.
    only = None
    for tail in tails:
        for tau in taus:
            for W in Ws:
                # Uncorrected components first: the null curve is fit on the
                # raw statistic, so this pass must not itself be corrected.
                cfg_raw = GateConfig(
                    span_tokens=W, tau=tau,
                    tau_mode=str(base.get("tau_mode", "per_token")),
                    min_span_tokens=int(base.get("min_span_tokens", 4)),
                    tail_mode=tail, tail_quantile=args.tail_quantile,
                    include_eos=bool(base.get("include_eos", False)),
                    c_trunc=float(base.get("c_trunc", 0.2)),
                    min_common_tokens=int(base.get("min_common_tokens", 8)),
                    prefix_tokens=int(
                        args.prefix_tokens if args.prefix_tokens is not None
                        else base.get("prefix_tokens", 0)
                    ),
                    null_correction=False, scale=1.0,
                )
                comp = gate_components(tok_true, n_true, tok_cf, n_cf, cfg=cfg_raw)
                raw = comp["delta_hat"]
                raw_pos = float((raw > 0).float().mean())
                try:
                    fit = fit_calibration(
                        raw, comp["n_spans"], span_tokens=W,
                        target_zero_rate=args.target_zero_rate,
                        target_pct=args.target_pct, target_q=args.target_q,
                        null_correction=use_null,
                    )
                except ValueError as e:
                    print(f"{W:>5} {tau:>5.2f} {tail:>9} {100*raw_pos:>7.1f}% "
                          f"{'':>9} {'':>9} {'':>7} {'':>6} {'':>6}  {str(e)[:30]:<30}")
                    continue
                dh, s = fit["delta_hat"], fit["scale"]
                null = fit["null"]
                only = (cfg_raw, comp, fit)
                drift = (max(null.mu) - min(null.mu)) if null is not None else 0.0

                note = ""
                s_eff = s
                if s <= 0:
                    note = "P10<=0: s=1, UNCALIBRATED"
                    s_eff = 1.0
                G = torch.clamp(2 * torch.sigmoid(dh / s_eff) - 1, min=0.0)
                zero_frac = float((G == 0).float().mean())
                v_first = float((G[quint[0]] == 0).float().mean())
                v_last = float((G[quint[-1]] == 0).float().mean())
                bias = (v_last / v_first) if v_first > 1e-6 else float("inf")
                q1, q2, q3 = (
                    float(torch.quantile(dh, q)) for q in (0.25, 0.5, 0.75)
                )
                iqr = q3 - q1
                sep = (q2 / iqr) if iqr > 1e-9 else float("inf")

                if not note:
                    # Two-sided: the correction can also OVERSHOOT, zeroing
                    # short responses more than long ones. That is just as
                    # much a length confound, and it reads as "safe" if only
                    # the upper tail is checked.
                    if bias > 3 or bias < 1 / 3:
                        note = f"length-confounded ({100*v_first:.0f}%->{100*v_last:.0f}%)"
                    elif not use_null and raw_pos < 0.9:
                        note = "clean ceiling below 90%"
                    else:
                        note = "ok"
                        # Maximise separation; break ties toward the smaller
                        # correction, since a statistic that needs less
                        # propping up is the more trustworthy one.
                        cand = (-sep, drift)
                        if best is None or cand < best[0]:
                            best = (cand, W, tau, tail)
                            winner = (cfg_raw, comp, fit)
                print(f"{W:>5} {tau:>5.2f} {tail:>9} {100*raw_pos:>7.1f}% "
                      f"{drift:>9.4f} {s:>9.4f} {100*zero_frac:>6.1f}% {bias:>6.2f} "
                      f"{sep:>6.2f}  {note:<30}")

    print()
    single = len(Ws) == 1 and len(taus) == 1 and len(tails) == 1
    if best is None and single and args.refit_out and only is not None:
        # Nothing to choose between; write the one point that was asked for.
        _write_refit(args, ref, only)
        print("NOTE: this configuration did not reach 'ok' — written anyway "
              "because exactly one was requested. Correct for an ablation "
              "arm, wrong for a headline run.")
        return
    if best is None:
        print("No configuration reached 'ok'. The two things to try, in order:")
        print("  1. a larger W — per-span noise falls as 1/sqrt(W), which is")
        print("     what pulls Delta^min back above zero on clean data;")
        print("  2. tail_mode=quantile, which stops one unlucky span from")
        print("     deciding the sample (it is Eq. 5 relaxed, so it is an")
        print("     honest ablation, not a fudge — report it as such).")
        print("If nothing works, the tail test is not viable on this backbone")
        print("and the honest move is Delta_bar alone, reported as such.")
        if args.refit_out:
            sys.exit(1)
        return

    (neg_sep, drift), W, tau, tail = best
    print(f"Best by separation, then by smallest null correction: "
          f"W={W} tau={tau} tail={tail} (sep {-neg_sep:.2f}, mu-drift {drift:.4f})")
    print()
    print("`sep` is a clean-only proxy. Confirm on a LABELLED pool before")
    print("committing:  python scripts/score_pool.py --config <arm> ...")
    print()
    print("Apply it in configs/methods/tag.yaml (or the 7B arm):")
    print(f"  selection.tag.span_tokens: {W}")
    print(f"  selection.tag.tau: {tau}")
    if tail != "min":
        print(f"  selection.tag.tail_mode: {tail}")
    print(f"  selection.tag.target_zero_rate: {args.target_zero_rate}")

    if args.refit_out:
        _write_refit(args, ref, winner)


def _write_refit(args, ref, chosen) -> None:
    """Write a calibration artifact for one swept configuration, no GPU."""
    import torch

    cfg_raw, comp, fit = chosen
    out = Path(args.refit_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(ref)  # keep model_path / pool / counterfactual / tokens
    payload.update(
        {
            "delta_hat": comp["delta_hat"].cpu(),
            "delta_hat_centered": fit["delta_hat"].cpu(),
            "delta_bar": comp["delta_bar"].cpu(),
            "delta_min": comp["delta_min"].cpu(),
            "n_spans": comp["n_spans"].cpu(),
            "n_common": comp["n_common"].cpu(),
            "null": fit["null"].to_dict() if fit["null"] is not None else None,
            "scale": fit["scale"],
            "target_pct": args.target_pct,
            "target_q": args.target_q,
            "target_zero_rate": args.target_zero_rate,
            # The identity a training run checks itself against. It must
            # describe the SWEPT configuration, not the one the original
            # forward pass happened to use.
            "gate_config": cfg_raw.identity(),
            "refit_from": str(args.ref),
        }
    )
    torch.save(payload, out)
    print()
    print(f"Wrote refit reference -> {out}")
    print(f"  W = {cfg_raw.span_tokens} | s = {fit['scale']:.6f} | "
          f"clean zero-weight rate {100*fit['zero_rate']:.1f}% | "
          f"null {'off' if fit['null'] is None else 'on'}")
    print("Point the arm at it and recompute the gate:")
    print(f"  export TAG_GATE_REF_7B={out}")
    print("  rm -f $POOLS/composite20/tag_gate_*.pt")
    print("  bash scripts/precompute_gate.sh configs/experiments/lowq/tag_7b.yaml")


if __name__ == "__main__":
    main()
