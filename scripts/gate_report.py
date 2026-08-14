#!/usr/bin/env python
"""Does the gate separate corrupted data from clean? — CPU, seconds, no forward.

    python scripts/gate_report.py \\
        --gate $POOLS/composite20/tag_gate_qwen2.5-7b.pt \\
        --manifest $POOLS/composite20/corruption_manifest.json

Everything needed is already on disk after scripts/precompute_gate.sh: the
cache holds G for every pool record, and the manifest holds the ground-truth
corruption labels. So the first and most decisive question — *is G predictive
of dirtiness at all* — costs nothing and should be answered BEFORE spending
GPU hours on training arms.

This answers the gate in isolation. It does NOT answer "does G . R beat R",
which needs the dynamic reward R and therefore a forward pass
(scripts/score_pool.py). But if G shows no separation here, that comparison
cannot come out well either, and the cheap check has saved the expensive one.

What to read:

  AP(dirty)     average precision of -G as a dirty detector, against the
                base rate (= the AP a random detector gets). Above the base
                rate means G carries signal; at the base rate it carries none.
  floor block   the G == 0 samples. This is the set the gate refuses outright,
                so its purity is the sharpest statement the gate makes: a
                floor block that is mostly dirty is the gate working.
  dirty@K       corrupted fraction of the top-K by G alone, versus the pool's
                base rate. Selection in training ranks by G . R, not G, so
                this is an upper-bound-ish sanity check on the gate's own
                contribution rather than a prediction of the arm's result.
  by type       mean G per corruption type. The gate is built to catch
                instruction-response mismatch; it is NOT built to catch every
                corruption (a fluent wrong answer can still be well explained
                by its instruction). Types it misses are a finding to report,
                not necessarily a bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def average_precision(detector, labels) -> float:
    """AP for binary ``labels`` (1 = dirty) ranked by ``detector`` desc."""
    import torch

    order = torch.argsort(detector, descending=True)
    y = labels[order].float()
    n_pos = float(y.sum().item())
    if n_pos == 0:
        return float("nan")
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float(((precision * y).sum() / n_pos).item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gate", required=True, help="tag_gate_*.pt from precompute_gate.sh")
    p.add_argument("--manifest", default=None,
                   help="corruption_manifest.json (default: beside the pool "
                        "recorded in the cache identity)")
    p.add_argument("--ks", default="0.1",
                   help="comma-separated selection ratios for dirty@K")
    args = p.parse_args()

    import torch

    cache = torch.load(args.gate, map_location="cpu", weights_only=False)
    G = cache["gate"].float().view(-1)
    n = G.numel()

    manifest_path = args.manifest
    if not manifest_path:
        ident = cache.get("identity") or {}
        pool_files = str(ident.get("pool_files") or "")
        if pool_files:
            guess = Path(pool_files.split(",")[0]).with_name("corruption_manifest.json")
            if guess.exists():
                manifest_path = str(guess)
                print(f"(manifest from the cache's pool identity: {guess})")
    if not manifest_path:
        sys.exit(
            "no --manifest and none inferable from the cache identity; pass it "
            "explicitly. Without labels this script cannot say anything."
        )

    with open(manifest_path) as f:
        man = json.load(f)
    entries = man.get("entries") or {}

    types: List[str] = ["clean"] * n
    for k, e in entries.items():
        i = int(k)
        if 0 <= i < n:
            types[i] = str(e.get("type", "unknown"))
    dirty = torch.tensor([t != "clean" for t in types], dtype=torch.bool)
    base = float(dirty.float().mean().item())

    print(f"gate     : {args.gate}")
    print(f"manifest : {manifest_path}")
    print(f"n        : {n}   dirty {100*base:.1f}%  ({int(dirty.sum())})")
    cfg = cache.get("config") or {}
    print(f"config   : W={cfg.get('span_tokens')} tau={cfg.get('tau')} "
          f"s={cfg.get('scale')} null={'on' if cfg.get('null_correction') else 'OFF'}")
    print()

    # ---- weight distribution -------------------------------------------
    at_floor = G == 0
    print("G distribution")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  q{q:<5.2f} = {float(torch.quantile(G, q)):.4f}")
    print(f"  G == 0        : {100*float(at_floor.float().mean()):5.1f}%  "
          f"({int(at_floor.sum())})")
    print(f"  0 < G < 0.99  : {100*float(((G>0)&(G<0.99)).float().mean()):5.1f}%   "
          f"<- graded, not a mask")
    print(f"  G >= 0.99     : {100*float((G>=0.99).float().mean()):5.1f}%")
    print()

    # ---- the headline: does G know what is dirty? -----------------------
    ap = average_precision(-G, dirty)
    print("Separation")
    print(f"  AP(dirty | -G)      : {ap:.4f}   (base rate {base:.4f}"
          f" = a detector with no signal)")
    lift = ap / base if base > 0 else float("nan")
    print(f"  lift over base      : {lift:.2f}x")
    print(f"  mean G  clean       : {float(G[~dirty].mean()):.4f}")
    print(f"  mean G  dirty       : {float(G[dirty].mean()):.4f}")
    print()

    # The floor block is the gate's sharpest claim: these are refused outright.
    if int(at_floor.sum()) > 0:
        pur = float(dirty[at_floor].float().mean())
        print(f"  floor block (G==0)  : n={int(at_floor.sum())}, "
              f"{100*pur:.1f}% dirty  (pool base {100*base:.1f}%)")
        print(f"                        -> {pur/base:.2f}x enriched in corruption"
              if base > 0 else "")
        rec = float(dirty[at_floor].float().sum() / dirty.float().sum())
        print(f"                        catches {100*rec:.1f}% of all dirty samples")
    else:
        print("  floor block (G==0)  : EMPTY — the gate refuses nothing.")
    print()

    # ---- what a G-only selection would pick -----------------------------
    print("Selecting by G alone (training ranks by G . R, so this is the gate's"
          " own contribution, not the arm's result)")
    for ks in args.ks.split(","):
        k = float(ks.strip())
        kk = max(1, int(round(k * n)))
        top = torch.topk(G, kk).indices
        d = float(dirty[top].float().mean())
        print(f"  dirty@{k:<5.2f} = {100*d:5.1f}%   (base {100*base:.1f}%, "
              f"{'-' if d < base else '+'}{abs(100*(d-base)):.1f}pp)")
    print()

    # ---- per corruption type -------------------------------------------
    # The gate targets instruction-response mismatch. Types it does not move
    # are a reportable limitation, not automatically a defect.
    by_type: Dict[str, List[int]] = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    print(f"{'type':<16} {'n':>7} {'mean G':>8} {'G==0':>8}")
    print("-" * 43)
    for t in sorted(by_type, key=lambda x: (x != "clean", x)):
        idx = torch.tensor(by_type[t], dtype=torch.long)
        g = G[idx]
        print(f"{t:<16} {len(idx):>7} {float(g.mean()):>8.4f} "
              f"{100*float((g==0).float().mean()):>7.1f}%")


if __name__ == "__main__":
    main()
