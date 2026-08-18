#!/usr/bin/env python
"""Does each benchmark directory contain what its evaluator will ask for?

    python scripts/check_eval_data.py                       # the 8 of Table 2
    python scripts/check_eval_data.py --benchmarks mmlu,bbh

A directory existing is not the same as a benchmark being present, and the
gap between the two is expensive: a 7B eval that dies on the fifth benchmark
has already spent an hour. Worse, the layouts differ per mirror — MMLU's
parquet shards sit directly in the directory on one box and under ``all/``
on another, GSM8K hides its test split under ``main/``, MBPP under
``sanitized/`` — so pointing at the plausible-looking parent silently
produces "No test-*.parquet found" much later.

This checks the ACTUAL files each evaluator opens, and when the corpus is
one level away it says so with the exact export that fixes it.

The expectations below mirror the loaders in tag/evals/*.py. If one of those
changes its layout, this file is the other half of that change.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# bench -> (env var, [required glob relative to data_dir], human description)
# A benchmark passes when EVERY entry in its list matches at least one file.
_SPEC: Dict[str, Tuple[str, List[str], str]] = {
    # tag/evals/mmlu.py: os.listdir(data_dir), test-*.parquet and dev-*.parquet
    "mmlu": ("MMLU_DATA_DIR", ["test-*.parquet", "dev-*.parquet"],
             "HF cais/mmlu `all` split parquet shards"),
    # tag/evals/mmlu_pro.py
    "mmlu_pro": ("MMLU_PRO_DATA_DIR", ["test-*.parquet", "validation-*.parquet"],
                 "MMLU-Pro parquet shards"),
    # tag/evals/bbh.py: data_dir.glob("*.json"), with one level of nesting handled
    "bbh": ("BBH_DATA_DIR", ["*.json|*/*.json"], "one .json per BBH task"),
    # tag/evals/svamp.py
    "svamp": ("SVAMP_DATA_DIR", ["test-*.parquet|test.parquet"],
              "ChilleD/SVAMP test parquet"),
    # tag/evals/gsm8k.py: os.path.join(data_dir, "main", "test.parquet")
    "gsm8k": ("GSM8K_DATA_DIR", ["main/test.parquet|main/test-*.parquet"],
              "GSM8K `main` config test parquet"),
    # tag/evals/mbpp.py: os.path.join(data_dir, config) then {test,prompt}-*.parquet
    "mbpp": ("MBPP_DATA_DIR",
             ["sanitized/test-*.parquet", "sanitized/prompt-*.parquet"],
             "MBPP `sanitized` config test + prompt parquet"),
    # tag/evals/humaneval.py: os.path.join(data_dir, "HumanEval.jsonl.gz")
    "humaneval": ("HUMANEVAL_DATA_DIR", ["HumanEval.jsonl.gz"],
                  "HumanEval.jsonl.gz (also needs `pip install human-eval`)"),
    # tag/evals/tydiqa.py: globs the split files, tolerating a data/ level
    "tydiqa": ("TYDIQA_DATA_DIR",
               ["validation*.parquet|validation*.json*|data/validation*.parquet"],
               "TyDiQA validation split"),
    # tag/evals/xquad.py: os.path.join(data_dir, f"xquad.{lang}.json")
    "xquad": ("XQUAD_DATA_DIR", ["xquad.*.json"], "xquad.<lang>.json per language"),
}

TABLE2 = ["mmlu", "bbh", "svamp", "gsm8k", "mbpp", "humaneval", "tydiqa", "xquad"]

# Where else the same corpus might be sitting, relative to the configured dir.
_NEARBY = ["all", "data", "sanitized", "main", "test"]


def _matches(root: Path, pattern: str) -> bool:
    """``a|b`` means either alternative satisfies the requirement."""
    for alt in pattern.split("|"):
        try:
            if next(root.glob(alt), None) is not None:
                return True
        except (OSError, ValueError):
            continue
    return False


def _check_dir(root: Path, needs: Sequence[str]) -> List[str]:
    return [p for p in needs if not _matches(root, p)]


def _suggest(root: Path, needs: Sequence[str]) -> Optional[Path]:
    """A directory near ``root`` that DOES satisfy the requirement."""
    cands: List[Path] = [root / n for n in _NEARBY]
    if root.parent != root:
        cands.append(root.parent)
        cands.extend(root.parent / n for n in _NEARBY)
    for parent in (root, root.parent):
        try:
            cands.extend(c for c in parent.iterdir() if c.is_dir())
        except OSError:
            pass
    seen = set()
    for c in cands:
        if c in seen or not c.is_dir():
            continue
        seen.add(c)
        if not _check_dir(c, needs):
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default=",".join(TABLE2),
                   help=f"comma-separated (default: the Table 2 eight). "
                        f"Known: {','.join(sorted(_SPEC))}")
    p.add_argument("--quiet", action="store_true",
                   help="print only failures")
    args = p.parse_args()

    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in names if b not in _SPEC]
    if unknown:
        print(f"unknown benchmark(s): {', '.join(unknown)}; "
              f"known: {', '.join(sorted(_SPEC))}", file=sys.stderr)
        return 2

    bad = 0
    for b in names:
        var, needs, desc = _SPEC[b]
        raw = os.environ.get(var, "")
        if not raw:
            print(f"FAIL {b:<10} ${var} is unset — source scripts/gpu_cloud/env.sh")
            bad += 1
            continue
        root = Path(raw)
        if not root.is_dir():
            print(f"FAIL {b:<10} {raw}")
            print(f"     no such directory ({desc})")
            bad += 1
            continue
        missing = _check_dir(root, needs)
        if not missing:
            if not args.quiet:
                print(f"ok   {b:<10} {raw}")
            continue
        bad += 1
        print(f"FAIL {b:<10} {raw}")
        print(f"     missing: {', '.join(missing)}   ({desc})")
        # "missing" is not actionable on its own — the directory usually holds
        # SOMETHING, and what it holds says whether the corpus is one level
        # down, in another format, or genuinely absent.
        try:
            entries = sorted(e.name + ("/" if e.is_dir() else "")
                             for e in list(root.iterdir())[:200])
        except OSError as e:
            entries = [f"<unreadable: {e}>"]
        if entries:
            shown = ", ".join(entries[:8])
            more = f" (+{len(entries) - 8} more)" if len(entries) > 8 else ""
            print(f"     contains: {shown}{more}")
        else:
            print(f"     contains: <empty>")
        alt = _suggest(root, needs)
        if alt is not None:
            print(f"     found it here instead — fix with:")
            print(f"       export {var}={alt}")

    print()
    if bad:
        print(f"{bad} of {len(names)} benchmark(s) not usable. Eval would fail "
              f"on those, so it will refuse to start.")
        print("TAG_BENCH_ROOTS=/parent/of/corpora source scripts/gpu_cloud/env.sh")
        print("re-probes; scripts/download_<bench>.sh fetches what is truly absent.")
        return 1
    print(f"all {len(names)} benchmark(s) ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
