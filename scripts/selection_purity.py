#!/usr/bin/env python
"""What did each arm actually train on? — CPU, instant, no forward.

    python scripts/selection_purity.py \\
        --manifest $POOLS/composite20/corruption_manifest.json \\
        tag_prefix_7b=$OUTPUT_ROOT/lowq/tag_prefix_7b_seed42 \\
        legacy_7b=$OUTPUT_ROOT/lowq/legacy_7b_seed42

Every arm writes ``selected_indices_epoch{N}.json`` — the exact subset it
trained on that epoch. Cross it with the corruption manifest and you get
the one number the whole method is about: **how much corrupted data made it
into the training subset**, versus the pool it was drawn from.

This lands epochs before any eval does, and it is more diagnostic than the
eval: an arm whose selected subset is no cleaner than the pool cannot be
helped by the gate, whatever the downstream benchmark says. Conversely a
large drop here with no eval gain is also a finding — it says the
corruption that was removed was not the corruption that hurt.

Read it against two references, both printed:

  pool base   the corrupted fraction of the whole candidate pool. An arm at
              the base rate selected as if the labels did not exist.
  random      the same thing — random selection is unbiased — so any arm
              at base is indistinguishable from picking at random on this
              axis, however good its score looks.

The per-type table is where the story is. The gate is built to catch
instruction-response mismatch; it is NOT built to catch a fluent wrong
answer, which is genuinely well explained by its instruction. A type that
does not move is a limitation to report, not necessarily a bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_manifest(path: str) -> Tuple[int, List[str]]:
    """``(n_total, per-record type)`` with 'clean' for records not listed."""
    with open(path) as f:
        man = json.load(f)
    entries = man.get("entries") or {}
    n = int(man.get("n_total") or 0)
    if n <= 0:
        # Older manifests predate n_total. The highest listed index is a
        # LOWER bound on the pool size, so say so rather than quietly
        # reporting a base rate computed against the wrong denominator.
        n = (max(int(k) for k in entries) + 1) if entries else 0
        print(
            f"WARNING: manifest has no n_total; assuming n={n} from the "
            f"highest corrupted index. The base rate below is an OVER-estimate "
            f"if the pool has clean records past that point.",
            file=sys.stderr,
        )
    types = ["clean"] * n
    for k, e in entries.items():
        i = int(k)
        if 0 <= i < n:
            types[i] = str(e.get("type", "unknown"))
    return n, types


def resolve_run_dir(d: Path) -> Path:
    """Accept either a run dir or the experiment dir above it.

    train.py writes to ``<output_subdir>/runs/<tag>/`` and points
    ``<output_subdir>/_latest`` at the newest one, so the path a person has
    to hand (the one in the config) is one or two levels up from the
    selections. Follow the pointer rather than making them look it up.
    """
    if list(d.glob("selected_indices_epoch*.json")):
        return d
    latest = d / "_latest"
    if latest.exists():
        return latest.resolve()
    runs = d / "runs"
    if runs.is_dir():
        cands = [c for c in runs.iterdir() if c.is_dir()]
        if cands:
            return max(cands, key=lambda c: c.stat().st_mtime)
    return d


def find_epoch_files(run_dir: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in run_dir.glob("selected_indices_epoch*.json"):
        stem = p.stem.replace("selected_indices_epoch", "")
        if stem.isdigit():
            out.append((int(stem), p))
    return sorted(out)


def read_selection(path: Path, n: int) -> Optional[List[int]]:
    try:
        with open(path) as f:
            sel = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ! {path.name}: unreadable ({e})", file=sys.stderr)
        return None
    if not isinstance(sel, list) or not sel:
        print(f"  ! {path.name}: empty or not a list", file=sys.stderr)
        return None
    idx = [int(x) for x in sel]
    bad = [i for i in idx if not (0 <= i < n)]
    if bad:
        # A selection indexing past the pool means the run and the manifest
        # describe different pools — every number below would be fiction.
        print(
            f"  ! {path.name}: {len(bad)} index/indices outside [0, {n}) "
            f"(e.g. {bad[:3]}) — this selection is from a DIFFERENT pool than "
            f"the manifest. Skipping.",
            file=sys.stderr,
        )
        return None
    return idx


def compose(idx: List[int], types: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in idx:
        t = types[i]
        counts[t] = counts.get(t, 0) + 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+",
                   help="run dirs, optionally LABEL=DIR (label defaults to the "
                        "directory name)")
    p.add_argument("--manifest", required=True,
                   help="corruption_manifest.json for the pool the arms drew from")
    p.add_argument("--epoch", type=int, default=None,
                   help="only this epoch (default: every epoch found)")
    args = p.parse_args()

    n, types = load_manifest(args.manifest)
    if n == 0:
        sys.exit("manifest lists no records")
    pool_counts = compose(list(range(n)), types)
    base_dirty = 1.0 - pool_counts.get("clean", 0) / n
    all_types = sorted(t for t in pool_counts if t != "clean")

    print(f"manifest : {args.manifest}")
    print(f"pool     : n={n}   dirty {100*base_dirty:.1f}%  "
          f"({n - pool_counts.get('clean', 0)})")
    print()

    rows: List[Tuple[str, int, List[int]]] = []
    for spec in args.runs:
        label, _, d = spec.partition("=")
        if not d:
            d, label = label, Path(label).name
        run_dir = Path(d)
        if not run_dir.is_dir():
            print(f"! {label}: no such directory: {run_dir}", file=sys.stderr)
            continue
        run_dir = resolve_run_dir(run_dir)
        found = find_epoch_files(run_dir)
        if not found:
            print(f"! {label}: no selected_indices_epoch*.json yet in {run_dir}",
                  file=sys.stderr)
            continue
        for ep, path in found:
            if args.epoch is not None and ep != args.epoch:
                continue
            idx = read_selection(path, n)
            if idx is not None:
                rows.append((label, ep, idx))

    if not rows:
        sys.exit(
            "\nNo selections to report. The arms write "
            "selected_indices_epoch1.json at the END of the first scoring "
            "pass, so this is expected until then."
        )

    # ---- headline: dirty fraction of what was trained on ----
    # A pool with no corruption — Table 2's is one — has nothing to divide by.
    # The subset SIZES and the overlap below are still worth having, so report
    # those rather than failing.
    labelled = base_dirty > 0
    if labelled:
        print("Corrupted fraction of the SELECTED subset")
        print(f"{'arm':<22}{'ep':>3}{'selected':>10}{'dirty':>9}{'vs pool':>10}"
              f"{'vs base':>9}")
        print("-" * 63)
        for label, ep, idx in rows:
            c = compose(idx, types)
            k = len(idx)
            d = 1.0 - c.get("clean", 0) / k
            print(f"{label:<22}{ep:>3}{k:>10}{100*d:>8.1f}%"
                  f"{100*(d - base_dirty):>+9.1f}pp{d / base_dirty:>8.2f}x")
        print(f"{'(random / no signal)':<22}{'':>3}{'':>10}{100*base_dirty:>8.1f}%"
              f"{0.0:>+9.1f}pp{1.0:>8.2f}x")
    else:
        print("The pool carries no corruption labels, so there is no dirty")
        print("fraction to compare. What is still meaningful is how much the")
        print("arms' subsets DIFFER — see the overlap at the bottom.")
        print()
        print(f"{'arm':<22}{'ep':>3}{'selected':>10}")
        print("-" * 35)
        for label, ep, idx in rows:
            print(f"{label:<22}{ep:>3}{len(idx):>10}")

    # ---- per type: which corruption was actually removed ----
    if not labelled:
        _overlap(rows)
        return
    print()
    print("Per-type share of the selected subset, vs that type's share of the pool")
    print("(1.00x = the type is selected exactly as often as it occurs;")
    print(" below 1.00x = the arm avoided it)")
    header = f"{'arm':<22}{'ep':>3}" + "".join(f"{t[:11]:>12}" for t in all_types)
    print(header)
    print("-" * len(header))
    for label, ep, idx in rows:
        c = compose(idx, types)
        k = len(idx)
        cells = ""
        for t in all_types:
            share = c.get(t, 0) / k
            pool_share = pool_counts[t] / n
            cells += f"{share / pool_share:>11.2f}x" if pool_share else f"{'-':>12}"
        print(f"{label:<22}{ep:>3}{cells}")
    print(f"{'(pool composition)':<22}{'':>3}"
          + "".join(f"{100*pool_counts[t]/n:>11.1f}%" for t in all_types))

    _overlap(rows)


def _overlap(rows) -> None:
    # ---- overlap between arms at the same epoch ----
    by_epoch: Dict[int, List[Tuple[str, List[int]]]] = {}
    for label, ep, idx in rows:
        by_epoch.setdefault(ep, []).append((label, idx))
    pairs = [(ep, v) for ep, v in sorted(by_epoch.items()) if len(v) >= 2]
    if pairs:
        print()
        print("Subset overlap between arms (how different are they really?)")
        for ep, v in pairs:
            for a in range(len(v)):
                for b in range(a + 1, len(v)):
                    la, ia = v[a]
                    lb, ib = v[b]
                    inter = len(set(ia) & set(ib))
                    union = len(set(ia) | set(ib))
                    print(f"  epoch {ep}: {la} vs {lb} — {inter} shared "
                          f"({100*inter/min(len(ia), len(ib)):.1f}% of the smaller), "
                          f"Jaccard {inter/union:.3f}")


if __name__ == "__main__":
    main()
