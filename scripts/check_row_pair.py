#!/usr/bin/env python
"""Do two table rows differ by ONLY what they claim to differ by?

    python scripts/check_row_pair.py \\
        configs/experiments/main_7b/llama2/legacy_10.yaml \\
        configs/experiments/main_7b/llama2/tag_10.yaml

A row added under Table 2 asserts something specific: that its number and the
row above it were produced by the same pipeline on the same data with the same
hyperparameters, so the difference between them is the method. That assertion
is made in prose and broken in YAML — an inherited default that moved, an
optimizer override copied into one file and not the other, a pool pinned in
one row and read from the environment in the other.

This resolves both configs the way training resolves them, environment
interpolation and all, and reports every key that differs. Keys the pair is
ALLOWED to differ on are listed below; anything else is a finding.

Run it before launching the pair, and again before the numbers go in the
paper — the second time catches an environment that changed underneath you,
which is the failure this script exists for: gpu_cloud/env.sh points
ALPACA_DATA_FILES at the corrupted lowq pool, so a Table 2 row launched from
a lowq shell trains on corrupted data and says nothing about it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Keys whose whole subtree may legitimately differ between a gated row and its
# ungated control. Everything else must match.
_ALLOWED_PREFIXES = (
    "selection.score_mode",   # "tag" vs unset — the difference being claimed
    "selection.tag",          # the gate's own parameters
    "output_subdir",          # each row writes to its own directory
    "method",                 # some baselines name themselves differently
)

# A pool path containing any of these is the corrupted lowq pool, not the
# corpus Table 2 is defined on.
_CORRUPT_MARKERS = ("composite", "corrupt", "polluted", "lowq")


def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def allowed(key: str) -> bool:
    return any(key == p or key.startswith(p + ".") for p in _ALLOWED_PREFIXES)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", help="the row already in the table (e.g. legacy_10.yaml)")
    p.add_argument("candidate", help="the row being added (e.g. tag_10.yaml)")
    p.add_argument("--allow", action="append", default=[],
                   help="extra key (or key prefix) the pair may differ on; repeatable")
    args = p.parse_args()

    from tag.core.utils import load_config

    cfgs = {}
    for role, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not Path(path).is_file():
            print(f"no such config: {path}", file=sys.stderr)
            return 2
        cfgs[role] = flatten(load_config(path))

    extra = tuple(args.allow)
    a, b = cfgs["baseline"], cfgs["candidate"]
    keys = sorted(set(a) | set(b))

    problems: List[Tuple[str, Any, Any]] = []
    expected: List[Tuple[str, Any, Any]] = []
    for k in keys:
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va == vb:
            continue
        if allowed(k) or any(k == e or k.startswith(e + ".") for e in extra):
            expected.append((k, va, vb))
        else:
            problems.append((k, va, vb))

    name_a, name_b = Path(args.baseline).stem, Path(args.candidate).stem
    print(f"baseline  : {args.baseline}")
    print(f"candidate : {args.candidate}")
    print()

    if expected:
        print("Differences the pair is allowed to have (the method itself):")
        for k, va, vb in expected:
            print(f"  {k}")
            print(f"      {name_a}: {va!r}")
            print(f"      {name_b}: {vb!r}")
        print()

    # ---- the pool is checked on its own: equal is not enough, it must also
    #      be the right corpus ----
    pool_bad = False
    pa, pb = str(a.get("data_files", "")), str(b.get("data_files", ""))
    for role, val in ((name_a, pa), (name_b, pb)):
        if not val:
            print(f"POOL  {role}: data_files is EMPTY — the env var it reads is unset.")
            pool_bad = True
        elif any(m in val.lower() for m in _CORRUPT_MARKERS):
            print(f"POOL  {role}: data_files looks like a corrupted / lowq pool:")
            print(f"        {val}")
            print("      Table 2 is defined on the clean source corpus. Export")
            print("      ALPACA_DATA_FILES and TAG_MAIN_POOL to it before launching.")
            pool_bad = True
    if pa and pb and pa != pb:
        # data_files is in neither allow-list, so it is already in `problems`;
        # this just says why it matters.
        print("POOL  the two rows read DIFFERENT pools — the comparison is void.")
        pool_bad = True
    elif pa and not pool_bad:
        print(f"POOL  both rows: {pa}")
        if not Path(pa.split(",")[0]).exists():
            print("      WARNING: that path does not exist yet.")

    print()
    if problems:
        print(f"{len(problems)} unexpected difference(s) — the rows are NOT comparable:")
        for k, va, vb in problems:
            print(f"  {k}")
            print(f"      {name_a}: {va!r}")
            print(f"      {name_b}: {vb!r}")
        print()
        print("Fix the configs, or pass --allow <key> if a difference is intended")
        print("AND will be stated in the paper.")
        return 1
    if pool_bad:
        return 1
    print("The two rows differ only by the method. Comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
