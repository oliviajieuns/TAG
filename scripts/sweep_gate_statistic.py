#!/usr/bin/env python
"""Which support statistic should Eq. 4-5 be? — CPU, seconds, no forward.

    python scripts/sweep_gate_statistic.py \\
        --gate $POOLS/composite20/tag_gate_qwen2.5-7b.pt \\
        --clean-ref $POOLS/clean_ref/delta_hat_7b.pt \\
        --manifest $POOLS/composite20/corruption_manifest.json

Requires the gate cache to have been written WITH token losses
(``store_token_losses: true``). With them every candidate summary of the
counterfactual contrast can be evaluated against the corruption labels for
free, so the choice is made on measurement instead of on argument.

Why this exists. scripts/span_profile.py established that the instruction's
predictive contribution is concentrated almost entirely in the first response
span: on Qwen2.5-7B the per-token nat gap runs 4.38 at m=0, 0.60 at m=1 and
0.05 by m>=12. A MINIMUM over spans therefore never selects span 0 — it has
the highest gain by construction — so Eq. 5 reads only the spans that carry
no signal. Among multi-span records the arg min sits at span 0 about 1% of
the time.

That rules out the minimum, but it does not by itself say what to replace it
with, and "drop the span idea" is only one of the options. The candidates
here span three families:

  whole-sequence   Delta_bar (Eq. 3) — the sum is dominated by span 0, which
                   is why it works, but it dilutes as responses get longer.
  prefix           the contrast restricted to the first n response tokens.
                   If support really is a prefix phenomenon these should
                   beat Delta_bar, and should not decay with length.
  tail             the minimum and low quantiles over spans, raw and after
                   per-position centring mu(m) estimated on the clean
                   reference. Included so the paper can report what the
                   span machinery actually delivered rather than dropping it
                   silently.

Read the per-type AP columns, not just the overall one: the corruption types
differ in kind, and a statistic that wins on mismatch while losing on noisy
is a different proposition from one that wins on both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def average_precision(detector, labels) -> float:
    import torch

    order = torch.argsort(detector, descending=True)
    y = labels[order].float()
    n_pos = float(y.sum().item())
    if n_pos == 0:
        return float("nan")
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float(((precision * y).sum() / n_pos).item())


def _prefix_gain(tok_true, tok_cf, n_common, n_prefix: int, eps: float = 1e-3):
    """1 - sum_{k<n} ell+ / sum_{k<n} ell-, over the first ``n_prefix`` tokens.

    Token-prefix rather than span-prefix: the decay measured by span_profile
    is a property of position, and a token window states the hypothesis
    directly without inheriting W.
    """
    import torch

    t = min(n_prefix, tok_true.size(1))
    lim = torch.minimum(n_common, torch.full_like(n_common, t))
    pos = torch.arange(tok_true.size(1)).unsqueeze(0)
    m = pos < lim.unsqueeze(1)
    s_true = (tok_true * m).sum(dim=1)
    s_cf = (tok_cf * m).sum(dim=1)
    return 1.0 - s_true / s_cf.clamp(min=eps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gate", required=True,
                   help="pool gate cache, written with store_token_losses: true")
    p.add_argument("--manifest", default=None, help="corruption_manifest.json")
    p.add_argument("--clean-ref", default=None,
                   help="clean-reference .pt, for the per-position null mu(m)")
    p.add_argument("--span-tokens", type=int, default=None)
    p.add_argument("--prefixes", default="8,16,32,64,128")
    p.add_argument("--quantiles", default="0.1,0.25")
    args = p.parse_args()

    import torch
    from tag.core.gate import span_gains, spans_from_token_losses, valid_span_mask

    cache = torch.load(args.gate, map_location="cpu", weights_only=False)
    if "token_true" not in cache or "token_cf" not in cache:
        sys.exit(
            f"{args.gate} has no per-token losses, so no statistic can be\n"
            f"re-derived. Recompute the gate with store_token_losses: true:\n"
            f"  (set it in the arm's YAML, then)\n"
            f"  bash scripts/precompute_gate.sh <config> <out>"
        )
    base_cfg = cache.get("config") or {}
    W = int(args.span_tokens or base_cfg.get("span_tokens", 16))
    tau = float(base_cfg.get("tau", 0.5))
    tau_mode = str(base_cfg.get("tau_mode", "per_token"))
    min_span = int(base_cfg.get("min_span_tokens", 4))

    tok_true = cache["token_true"].float()
    n_true = cache["n_true"]
    tok_cf = cache["token_cf"][0].float()
    n_cf = cache["n_cf"][0]
    n = tok_true.size(0)

    manifest_path = args.manifest
    if not manifest_path:
        ident = cache.get("identity") or {}
        pf = str(ident.get("pool_files") or "")
        if pf:
            guess = Path(pf.split(",")[0]).with_name("corruption_manifest.json")
            if guess.exists():
                manifest_path = str(guess)
    if not manifest_path:
        sys.exit("no --manifest and none inferable from the cache identity.")
    with open(manifest_path) as f:
        man = json.load(f)
    types: List[str] = ["clean"] * n
    for k, e in (man.get("entries") or {}).items():
        i = int(k)
        if 0 <= i < n:
            types[i] = str(e.get("type", "unknown"))
    dirty = torch.tensor([t != "clean" for t in types], dtype=torch.bool)
    base_rate = float(dirty.float().mean())

    print(f"gate     : {args.gate}")
    print(f"n        : {n}   dirty {100*base_rate:.1f}%   W={W}")
    print()

    sp = spans_from_token_losses(tok_true, n_true, tok_cf, n_cf, span_tokens=W)
    n_common = sp["n_common"]
    gains = span_gains(sp["span_true"], sp["span_cf"], sp["span_len"])
    mask = valid_span_mask(sp["span_cf"], sp["span_len"], tau=tau,
                           tau_mode=tau_mode, min_span_tokens=min_span)
    d_bar = 1.0 - sp["total_true"] / sp["total_cf"].clamp(min=1e-3)
    has_valid = mask.any(dim=1)
    big = torch.full_like(gains, float("inf"))

    cands: List[Tuple[str, "torch.Tensor"]] = [("Delta_bar (Eq. 3)", d_bar)]

    # Span 0 alone: where span_profile located essentially all of the signal.
    g0 = torch.where(mask[:, 0], gains[:, 0], d_bar)
    cands.append(("span 0 only", g0))

    for np_ in [int(x) for x in args.prefixes.split(",") if x.strip()]:
        cands.append((f"prefix {np_} tok", _prefix_gain(tok_true, tok_cf, n_common, np_)))

    masked = torch.where(mask, gains, big)
    d_min = torch.where(has_valid, masked.min(dim=1).values, d_bar)
    cands.append(("Delta^min (Eq. 5)", d_min))
    cands.append(("min(bar, Delta^min)", torch.minimum(d_bar, d_min)))

    # Per-position centring, mu(m) from the CLEAN reference so it cannot
    # absorb corruption signal (the same discipline as Eq. 5').
    if args.clean_ref:
        ref = torch.load(args.clean_ref, map_location="cpu", weights_only=False)
        if "token_true" in ref:
            rsp = spans_from_token_losses(
                ref["token_true"].float(), ref["n_true"],
                ref["token_cf"][0].float(), ref["n_cf"][0], span_tokens=W,
            )
            rg = span_gains(rsp["span_true"], rsp["span_cf"], rsp["span_len"])
            rm = valid_span_mask(rsp["span_cf"], rsp["span_len"], tau=tau,
                                 tau_mode=tau_mode, min_span_tokens=min_span)
            alpha = float(ref.get("target_zero_rate", 0.05))
            m_max = min(gains.size(1), rg.size(1))
            mu = torch.zeros(gains.size(1))
            for m in range(m_max):
                s = rm[:, m]
                if int(s.sum()) > 20:
                    mu[m] = torch.quantile(rg[s, m], alpha)
            cen = gains - mu.unsqueeze(0)
            cmask = torch.where(mask, cen, big)
            d_min_c = torch.where(has_valid, cmask.min(dim=1).values, d_bar)
            cands.append(("Delta^min pos-centred", d_min_c))
            cands.append(("min(bar, pos-centred)", torch.minimum(d_bar, d_min_c)))
        else:
            print("(--clean-ref has no token losses; skipping position centring)\n")

    for q in [float(x) for x in args.quantiles.split(",") if x.strip()]:
        tail = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            row = gains[i][mask[i]]
            tail[i] = torch.quantile(row, q) if row.numel() else float("inf")
        cands.append((f"span quantile {q:g}", torch.where(has_valid, tail, d_bar)))

    dirty_types = sorted({t for t in types if t != "clean"})
    clean_idx = [i for i, t in enumerate(types) if t == "clean"]
    by_type: Dict[str, List[int]] = {t: [] for t in dirty_types}
    for i, t in enumerate(types):
        if t != "clean":
            by_type[t].append(i)

    hdr = f"{'statistic':<24} {'AP all':>7}"
    for t in dirty_types:
        hdr += f" {t[:9]:>9}"
    print(hdr)
    print(f"{'(base rate)':<24} {base_rate:>7.3f}", end="")
    for t in dirty_types:
        b = len(by_type[t]) / (len(by_type[t]) + len(clean_idx))
        print(f" {b:>9.3f}", end="")
    print()
    print("-" * len(hdr))

    best = (None, -1.0)
    for name, v in cands:
        v = v.float().view(-1)
        v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
        ap_all = average_precision(-v, dirty)
        if ap_all > best[1]:
            best = (name, ap_all)
        row = f"{name:<24} {ap_all:>7.4f}"
        for t in dirty_types:
            sel = torch.tensor(by_type[t] + clean_idx, dtype=torch.long)
            lab = torch.zeros(sel.numel(), dtype=torch.bool)
            lab[: len(by_type[t])] = True
            row += f" {average_precision(-v[sel], lab):>9.4f}"
        print(row)

    print()
    print(f"Best overall: {best[0]} (AP {best[1]:.4f} vs base {base_rate:.4f}, "
          f"{best[1]/base_rate:.2f}x)")
    print()
    print("AP is rank-based, so these are independent of s and mu — they compare")
    print("the STATISTICS, not the calibrations. Whichever wins still has to be")
    print("calibrated (Eq. 5' centring + s) before it becomes a gate, and the")
    print("end-to-end arm is what settles the paper's claim.")


if __name__ == "__main__":
    main()
