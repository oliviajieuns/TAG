#!/usr/bin/env python
"""Why does the span minimum carry no signal? — CPU, seconds, no forward.

    python scripts/span_profile.py --ref $POOLS/clean_ref/delta_hat_7b.pt

Measured on the 7B pool, Delta_bar detects instruction-response mismatch at
13.8x the base rate while Delta^min detects nothing (scripts/gate_report.py).
That combination is strange on its face: a mismatched instruction should
degrade EVERY span, so the span-level contrast ought to inherit the
sequence-level signal rather than lose it.

The hypothesis this script tests is that the instruction's predictive
contribution DECAYS WITH POSITION. ell_k(y | u) conditions on the response
prefix y_<k as well as on u, so once the model has read enough of y it can
infer the topic and predict the remainder about as well without the correct
instruction as with it. If so, late spans have Delta_{i,m} ~ 0 for clean and
corrupted records alike -- and because Eq. 5 takes a MINIMUM, it selects
precisely those spans. The tail statistic would then be measuring "how much
does the instruction matter where it matters least", which is ~0 for
everything, and no threshold can separate on it.

Three outputs, in increasing order of how decisive they are:

  by-span-index profile   mean Delta_{i,m} and the mean per-token NLL under
                          both prompts, as a function of span index m. Under
                          the hypothesis the gap ell(y|u^-) - ell(y|u^+)
                          shrinks monotonically with m.
  argmin location         the distribution of arg min_m Delta_{i,m}, as a
                          share of the samples that HAVE a span at that
                          index. Uniform means the minimum is picking spans
                          on their merits; concentrated at high m means it is
                          picking on position, which is the failure.
  position-centred retry  the same argmin distribution after subtracting a
                          per-position null mu(m) estimated on this clean
                          reference. If centring flattens it, the fix is the
                          position analogue of Eq. 5' rather than dropping
                          Eqs. 4-5.

This reference is CLEAN, so it establishes the mechanism, not the detection
rate. Confirming that position-centring restores detection needs the dirty
pool's token losses (precompute with store_token_losses: true).
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
                   help="delta_hat_*.pt written WITH token losses")
    p.add_argument("--span-tokens", type=int, default=None,
                   help="W (default: the reference's own)")
    p.add_argument("--max-index", type=int, default=12,
                   help="report span indices 0..max-index, then a tail bucket")
    args = p.parse_args()

    import torch
    from tag.core.gate import (
        span_gains, spans_from_token_losses, valid_span_mask,
    )

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    if "token_true" not in ref:
        sys.exit(
            f"{args.ref} has no per-token losses — nothing to profile. "
            f"Regenerate with scripts/gpu_cloud/bootstrap.sh calibrate7b."
        )
    base = ref.get("gate_config") or {}
    W = int(args.span_tokens or base.get("span_tokens", 16))
    tau = float(base.get("tau", 0.5))
    tau_mode = str(base.get("tau_mode", "per_token"))
    min_span = int(base.get("min_span_tokens", 4))

    tok_true = ref["token_true"].float()
    n_true = ref["n_true"]
    tok_cf = ref["token_cf"][0].float()
    n_cf = ref["n_cf"][0]

    sp = spans_from_token_losses(tok_true, n_true, tok_cf, n_cf, span_tokens=W)
    gains = span_gains(sp["span_true"], sp["span_cf"], sp["span_len"])
    mask = valid_span_mask(sp["span_cf"], sp["span_len"], tau=tau,
                           tau_mode=tau_mode, min_span_tokens=min_span)
    n, m_max = gains.shape

    print(f"reference : {args.ref}   (CLEAN pool)")
    print(f"samples   : {n}   W={W} tau={tau}({tau_mode}) min_span={min_span}")
    print(f"span slots: {m_max}")
    print()

    # ---- 1. profile by span index --------------------------------------
    # Per-token means make the two prompts' losses comparable across spans of
    # different occupancy; the GAP between them is the instruction's
    # contribution at that position.
    ln = sp["span_len"].clamp(min=1).float()
    pt_true = sp["span_true"] / ln
    pt_cf = sp["span_cf"] / ln
    print("Instruction contribution by span index (valid spans only)")
    print(f"{'m':>4} {'n':>7} {'ell(y|u+)':>10} {'ell(y|u-)':>10} "
          f"{'gap':>8} {'mean D':>8} {'P05 D':>8}")
    print("-" * 62)
    lo = 0
    rows = list(range(min(args.max_index, m_max)))
    for m in rows:
        sel = mask[:, m]
        k = int(sel.sum())
        if k == 0:
            continue
        g = gains[sel, m]
        print(f"{m:>4} {k:>7} {float(pt_true[sel, m].mean()):>10.4f} "
              f"{float(pt_cf[sel, m].mean()):>10.4f} "
              f"{float((pt_cf[sel, m] - pt_true[sel, m]).mean()):>8.4f} "
              f"{float(g.mean()):>8.4f} {float(torch.quantile(g, 0.05)):>8.4f}")
    if m_max > len(rows):
        sel = mask[:, len(rows):].reshape(-1)
        gt = gains[:, len(rows):].reshape(-1)[sel]
        tt = pt_true[:, len(rows):].reshape(-1)[sel]
        cc = pt_cf[:, len(rows):].reshape(-1)[sel]
        if gt.numel():
            print(f"{'>=' + str(len(rows)):>4} {gt.numel():>7} "
                  f"{float(tt.mean()):>10.4f} {float(cc.mean()):>10.4f} "
                  f"{float((cc - tt).mean()):>8.4f} {float(gt.mean()):>8.4f} "
                  f"{float(torch.quantile(gt, 0.05)):>8.4f}")
    print()
    print("  gap = mean per-token ell(y|u^-) - ell(y|u^+): how many nats the")
    print("        TRUE instruction saves at that position. A gap that shrinks")
    print("        with m is the instruction being made redundant by the")
    print("        response prefix — the effect that would break a min.")
    print()

    # ---- 2. where does the minimum actually land? ----------------------
    # Normalised by how many samples HAVE a span at that index, otherwise
    # early indices win trivially (every sample has span 0, few have span 20).
    big = torch.full_like(gains, float("inf"))
    masked = torch.where(mask, gains, big)
    has_valid = mask.any(dim=1)
    argmin = masked.argmin(dim=1)

    def _argmin_profile(am: "torch.Tensor", title: str) -> None:
        print(title)
        print(f"{'m':>4} {'has span':>9} {'is argmin':>10} {'rate':>8}")
        print("-" * 35)
        for m in range(min(args.max_index, m_max)):
            avail = int((mask[:, m] & has_valid).sum())
            if avail == 0:
                continue
            hit = int(((am == m) & has_valid).sum())
            print(f"{m:>4} {avail:>9} {hit:>10} {100.0*hit/avail:>7.1f}%")
        tail_avail = int((mask[:, args.max_index:].any(dim=1) & has_valid).sum()) \
            if m_max > args.max_index else 0
        if tail_avail:
            hit = int(((am >= args.max_index) & has_valid).sum())
            print(f"{'>=' + str(args.max_index):>4} {tail_avail:>9} {hit:>10} "
                  f"{100.0*hit/tail_avail:>7.1f}%")
        print()

    _argmin_profile(
        argmin,
        "Where the minimum lands, RAW (rate = share of the samples that have\n"
        "a span at that index and whose minimum is there)",
    )

    # ---- 3. the position analogue of Eq. 5' -----------------------------
    # mu(m) = the same low quantile of Delta_{i,m} among CLEAN samples at that
    # position. Centring asks "is this span worse than clean spans HERE",
    # which is the question Eq. 5 meant to ask.
    alpha = float(ref.get("target_zero_rate", 0.05))
    mu_m = torch.zeros(m_max)
    for m in range(m_max):
        sel = mask[:, m]
        mu_m[m] = torch.quantile(gains[sel, m], alpha) if int(sel.sum()) > 20 else 0.0
    centred = gains - mu_m.unsqueeze(0)
    masked_c = torch.where(mask, centred, big)
    argmin_c = masked_c.argmin(dim=1)
    print(f"mu(m) at the {alpha:.0%} quantile of clean spans, by index:")
    print("  " + "  ".join(
        f"m{m}:{float(mu_m[m]):+.3f}" for m in range(min(8, m_max))
    ))
    print()
    _argmin_profile(
        argmin_c,
        "Where the minimum lands AFTER per-position centring (flat = the\n"
        "minimum is choosing on evidence rather than on position)",
    )

    d_bar = ref.get("delta_bar")
    if d_bar is not None:
        raw_min = torch.where(has_valid, masked.min(dim=1).values, d_bar.float())
        cen_min = torch.where(has_valid, masked_c.min(dim=1).values, d_bar.float())
        print("Clean-reference tail statistic")
        print(f"  Delta^min raw      mean {float(raw_min.mean()):+.4f}  "
              f"P05 {float(torch.quantile(raw_min, 0.05)):+.4f}")
        print(f"  Delta^min centred  mean {float(cen_min.mean()):+.4f}  "
              f"P05 {float(torch.quantile(cen_min, 0.05)):+.4f}")
        print(f"  Delta_bar          mean {float(d_bar.float().mean()):+.4f}")
        print()
        print("  Centring moves the clean tail to ~0 by construction. Whether")
        print("  it RESTORES DETECTION cannot be read here — this pool is all")
        print("  clean. That needs the dirty pool's token losses.")


if __name__ == "__main__":
    main()
