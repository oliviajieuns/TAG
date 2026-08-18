#!/usr/bin/env python
"""What did the gate actually put at the floor? — read them and judge.

    python scripts/gate_examples.py --gate "$TAG_GATE_CACHE_Q7"

On a corrupted pool, gate_report.py scores G against known labels. Table 2's
pool has none, and the question "does the gate work here" still has to be
answered — so answer it the way it can be answered without labels: look at
the records G ranked lowest and decide whether they deserve it.

This is not a soft substitute for the labelled check. A clean corpus is only
clean in the sense that nobody corrupted it on purpose; Alpaca-GPT4 is
machine-generated and does contain pairs whose response barely follows its
instruction. G is a measure of exactly that, so the floor block is a
prediction about which records those are, and the prediction is legible.

Two columns matter and are printed separately:

  G/c    the counterfactual contrast alone — what Eqs. 2-6 decided
  c      the completeness heuristic, which is string matching and can be
         wrong on its own terms

A floor block that is all c=0.2 says the truncation heuristic is doing the
work, not the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="tag_gate_*.pt")
    ap.add_argument("--pool", default=None,
                    help="pool.json (default: from the cache's identity)")
    ap.add_argument("-k", type=int, default=10, help="records per group")
    ap.add_argument("--width", type=int, default=110, help="chars per field")
    ap.add_argument("--seed", type=int, default=0, help="for the random sample")
    args = ap.parse_args()

    if not str(args.gate).strip():
        sys.exit("--gate is empty — an unset environment variable? "
                 "TAG_ENV_RESET=1 source scripts/gpu_cloud/env.sh")
    cache = torch.load(args.gate, map_location="cpu", weights_only=False)
    G = cache["gate"].float().view(-1)
    c = cache.get("completeness")
    c = c.float().view(-1) if c is not None else torch.ones_like(G)

    pool_path = args.pool
    if not pool_path:
        ident = cache.get("identity") or {}
        pf = str(ident.get("pool_files") or "")
        pool_path = pf.split(",")[0] if pf else ""
    if not pool_path or not Path(pool_path).is_file():
        sys.exit(f"pool not found ({pool_path!r}); pass --pool explicitly")
    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)
    if len(pool) != G.numel():
        sys.exit(f"pool has {len(pool)} records, gate has {G.numel()} — "
                 f"different pools")

    print(f"gate : {args.gate}")
    print(f"pool : {pool_path}   n={len(pool)}")
    n_floor = int((G <= 0).sum())
    print(f"floor: {n_floor} records at G == 0 ({100*n_floor/len(pool):.1f}%)")
    print()

    def show(title: str, idx) -> None:
        print("=" * 78)
        print(title)
        print("=" * 78)
        for i in [int(x) for x in idx]:
            r = pool[i]
            gc = float(G[i] / c[i]) if float(c[i]) else float("nan")
            print(f"[{i}]  G={float(G[i]):.4f}  c={float(c[i]):.2f}  G/c={gc:.4f}")
            print(f"  instruction: {_clip(r.get('instruction'), args.width)}")
            if str(r.get("input") or "").strip():
                print(f"  input      : {_clip(r.get('input'), args.width)}")
            print(f"  output     : {_clip(r.get('output'), args.width)}")
            print()

    order = torch.argsort(G)
    show(f"LOWEST G — the {args.k} the gate most wants to drop", order[: args.k])
    g = torch.Generator().manual_seed(args.seed)
    mid = torch.randperm(len(pool), generator=g)[: args.k]
    show(f"RANDOM {args.k} — the comparison that makes the above mean something",
         mid)
    show(f"HIGHEST G — the {args.k} it most wants to keep", order[-args.k:])

    print("Read the first group against the second. If the lowest-G records")
    print("are visibly worse instruction-response pairs than a random draw,")
    print("the gate is measuring what it claims to on this pool. If they look")
    print("the same, it is not — whatever a corrupted pool showed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
