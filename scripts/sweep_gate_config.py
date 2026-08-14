#!/usr/bin/env python
"""Choose W (and tau, and the tail statistic) from a saved calibration — no GPU.

    python scripts/sweep_gate_config.py --ref pools/clean_ref/delta_hat_7b.pt

The span width is a first-order hyper-parameter, not a detail. Delta^min is a
minimum over M = ceil(n/W) spans, so its null drifts downward as responses get
longer, while per-span noise shrinks like 1/sqrt(W): the two pull in opposite
directions and the balance decides both the clean false-veto rate and the
detection power. Picking W by re-running the calibration costs two full pool
forwards per candidate; re-deriving it from the cached per-token NLLs costs
seconds, which is the difference between choosing W on evidence and guessing.

What to read off the table:

  %pos    fraction of the CLEAN reference with Delta_hat > 0. Everything at or
          below zero is vetoed EXACTLY, by construction (Eq. 6), so this is a
          hard ceiling on how much clean data can survive at ANY scale s. If
          it is not comfortably above 90%, no calibration can rescue that W.
  s       the scale the calibration would derive. Non-positive means P10 of
          the clean reference is itself <= 0 and the run falls back to s=1.0,
          i.e. an uncalibrated gate.
  veto    clean false-veto rate at the derived s.
  bias    veto rate in the longest response quintile divided by the shortest.
          This is the length-confound number; near 1.0 is what you want.

The reference must have been written WITH token losses (the default since
calibrate_reliability.py grew --no-token-losses).
"""
from __future__ import annotations

import argparse
import math
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
    p.add_argument("--tau", default=None,
                   help="comma-separated tau values (default: the ref's own)")
    p.add_argument("--tail-modes", default="min",
                   help="comma-separated: min,quantile")
    p.add_argument("--tail-quantile", type=float, default=0.25)
    p.add_argument("--target-pct", type=float, default=0.10)
    p.add_argument("--target-q", type=float, default=0.8)
    args = p.parse_args()

    import torch
    from tads.core.gate import GateConfig, gate_components

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    if "token_true" not in ref:
        sys.exit(
            f"{args.ref} has no per-token losses, so nothing can be re-derived "
            f"without a forward pass.\nRegenerate it (the default now keeps "
            f"them):\n  python scripts/calibrate_reliability.py --mode tag ..."
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
    print()

    logit = math.log(args.target_q / (1.0 - args.target_q))
    resp_len = torch.minimum(n_true.long(), n_cf.long()).float()
    order = torch.argsort(resp_len)
    quint = torch.tensor_split(order, 5)

    Ws = [int(x) for x in args.span_tokens.split(",") if x.strip()]
    taus = ([float(x) for x in args.tau.split(",")] if args.tau
            else [float(base.get("tau", 0.5))])
    tails = [t.strip() for t in args.tail_modes.split(",") if t.strip()]

    print(f"{'W':>5} {'tau':>5} {'tail':>9} {'%pos':>7} {'s':>9} "
          f"{'veto':>7} {'bias':>6}  {'note':<34}")
    print("-" * 92)
    best = None
    for tail in tails:
        for tau in taus:
            for W in Ws:
                cfg = GateConfig(
                    span_tokens=W, tau=tau,
                    tau_mode=str(base.get("tau_mode", "per_token")),
                    min_span_tokens=int(base.get("min_span_tokens", 4)),
                    tail_mode=tail, tail_quantile=args.tail_quantile,
                    include_eos=bool(base.get("include_eos", False)),
                    c_trunc=float(base.get("c_trunc", 0.2)),
                    min_common_tokens=int(base.get("min_common_tokens", 8)),
                    scale=1.0,
                )
                comp = gate_components(tok_true, n_true, tok_cf, n_cf, cfg=cfg)
                dh = comp["delta_hat"]
                pos = float((dh > 0).float().mean())
                q = float(torch.quantile(dh, args.target_pct))
                s = q / logit
                note = ""
                if s <= 0:
                    note = "P10<=0: falls back to s=1, UNCALIBRATED"
                    s_eff = 1.0
                else:
                    s_eff = s
                G = torch.clamp(2 * torch.sigmoid(dh / s_eff) - 1, min=0.0)
                veto = float((G == 0).float().mean())
                v_first = float((G[quint[0]] == 0).float().mean())
                v_last = float((G[quint[-1]] == 0).float().mean())
                bias = (v_last / v_first) if v_first > 1e-6 else float("inf")
                if not note:
                    if bias > 3:
                        note = f"length-confounded ({100*v_first:.0f}%->{100*v_last:.0f}%)"
                    elif pos < 0.9:
                        note = "clean ceiling below 90%"
                    else:
                        note = "ok"
                        cand = (bias, veto)
                        if best is None or cand < best[0]:
                            best = (cand, W, tau, tail)
                print(f"{W:>5} {tau:>5.2f} {tail:>9} {100*pos:>6.1f}% "
                      f"{s:>9.4f} {100*veto:>6.1f}% {bias:>6.2f}  {note:<34}")

    print()
    if best is None:
        print("No configuration reached 'ok'. The two things to try, in order:")
        print("  1. a larger W — per-span noise falls as 1/sqrt(W), which is")
        print("     what pulls Delta^min back above zero on clean data;")
        print("  2. tail_mode=quantile, which stops one unlucky span from")
        print("     deciding the sample (it is Eq. 5 relaxed, so it is an")
        print("     honest ablation, not a fudge — report it as such).")
        print("If nothing works, the tail test is not viable on this backbone")
        print("and the honest move is Delta_bar alone, reported as such.")
    else:
        (bias, veto), W, tau, tail = best
        print(f"Best by length-bias then veto rate: W={W} tau={tau} tail={tail} "
              f"(bias {bias:.2f}, clean veto {100*veto:.1f}%)")
        print()
        print("Apply it in configs/methods/tag.yaml (or the 7B arm), then")
        print("RECALIBRATE at that W — s is a quantile of Delta_hat and its")
        print("distribution changes with the partition:")
        print(f"  tads.tag.span_tokens: {W}")
        print(f"  tads.tag.tau: {tau}")
        if tail != "min":
            print(f"  tads.tag.tail_mode: {tail}")


if __name__ == "__main__":
    main()
