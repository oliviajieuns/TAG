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
#
# Datasets that ship one config PER SUBSET, where a recursive search happily
# returns the first subset it trips over: MMLU resolved to
# .../mmlu/computer_security, whose test-*.parquet satisfies every pattern
# while covering one subject out of 57. A wrong number is worse than a
# missing one.
#
# The test is NOT the directory's name. Requiring it to be called `all`
# rejected the concatenation this repo builds, which is correct data under
# a different name. What actually distinguishes a subset is that its
# SIBLINGS look just like it — 56 other subject directories, each satisfying
# the same markers. A directory that stands alone is the whole dataset,
# whatever it is called.
_SUBSET_PRONE: Tuple[str, ...] = ("mmlu", "mmlu_pro")
_SIBLING_THRESHOLD = 2

_SPEC: Dict[str, Tuple[str, List[str], str]] = {
    # tag/evals/mmlu.py: os.listdir(data_dir), test-*.parquet and dev-*.parquet
    "mmlu": ("MMLU_DATA_DIR", ["test-*.parquet", "dev-*.parquet"],
             "HF cais/mmlu `all` split parquet shards"),
    # tag/evals/mmlu_pro.py
    "mmlu_pro": ("MMLU_PRO_DATA_DIR", ["test-*.parquet", "validation-*.parquet"],
                 "MMLU-Pro parquet shards"),
    # tag/evals/bbh.py: data_dir.glob("*.json"), with one level of nesting handled
    # cot-prompts/ is not optional in practice: without it bbh.py scores a
    # direct-answer few-shot baseline instead of the CoT one Table 2 reports,
    # and says so in a warning that is easy to miss mid-run.
    "bbh": ("BBH_DATA_DIR", ["*.json|*/*.json", "cot-prompts/*.txt"],
            "per-task .json AND cot-prompts/*.txt "
            "(scripts/download_bbh.sh)"),
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
_SKIP = {".git", ".github", "__pycache__"}


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


# How far below a corpus root to look. The corpora here are BUNDLES —
# <corpus>/datasets/<org>/<name>/... — so the files a loader opens are
# several levels down, and checking only the top reported seven of eight
# benchmarks as absent on a box that has all of them.
_MAX_DEPTH = 6


def _is_one_of_many(d: Path, needs: Sequence[str]) -> bool:
    """Do at least ``_SIBLING_THRESHOLD`` sibling directories look the same?

    That is what makes a directory a per-subset config rather than the whole
    dataset, and it holds regardless of what any of them is named.
    """
    parent = d.parent
    if parent == d:
        return False
    n = 0
    try:
        for c in parent.iterdir():
            if c == d or not c.is_dir() or c.name in _SKIP:
                continue
            if not _check_dir(c, needs):
                n += 1
                if n >= _SIBLING_THRESHOLD:
                    return True
    except OSError:
        return False
    return False


def _config_ok(bench: str, d: Path, needs: Sequence[str]) -> bool:
    """Reject a hit that is one subset of a dataset split per subset."""
    if bench not in _SUBSET_PRONE:
        return True
    return not _is_one_of_many(d, needs)


def _suggest(root: Path, needs: Sequence[str], bench: str = "") -> Optional[Path]:
    """A directory at or below ``root`` that DOES satisfy the requirement.

    Searches for one of the marker files, then walks back UP from each hit:
    gsm8k's requirement is ``main/test.parquet``, so the directory that
    satisfies it is the grandparent of the file, not its parent.
    """
    seen: set = set()

    def _ok(c: Path) -> bool:
        if c in seen or not c.is_dir():
            return False
        seen.add(c)
        if bench and not _config_ok(bench, c, needs):
            return False
        return not _check_dir(c, needs)

    for n in _NEARBY:
        if _ok(root / n):
            return root / n

    for pattern in needs:
        for alt in pattern.split("|"):
            base = alt.rsplit("/", 1)[-1]
            try:
                hits = root.rglob(base)
            except OSError:
                continue
            for i, f in enumerate(hits):
                if i >= 200:
                    break
                try:
                    if len(f.relative_to(root).parts) > _MAX_DEPTH:
                        continue
                except ValueError:
                    continue
                d = f.parent
                for _ in range(3):
                    if _ok(d):
                        return d
                    if d.parent == d:
                        break
                    d = d.parent
    return None


def _show_layout(args) -> int:
    """One bounded listing per corpus: subdirectories and file extensions."""
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for root_s in [r.strip() for r in args.roots.split(",") if r.strip()]:
        root = Path(root_s)
        print(f"### {root}")
        for b in names:
            for cand in (root / b, root / b.replace("_", "-"),
                         root / b.replace("-", "_")):
                if cand.is_dir():
                    break
            else:
                print(f"  {b:<10} <no directory>")
                continue
            print(f"  {b:<10} {cand}")
            try:
                kids = sorted(c.name + "/" for c in cand.iterdir() if c.is_dir())
                tops = sorted(c.name for c in cand.iterdir() if c.is_file())
            except OSError as e:
                print(f"      <unreadable: {e}>")
                continue
            if kids:
                shown = ", ".join(kids[:12])
                print(f"      dirs : {shown}"
                      f"{f' (+{len(kids)-12})' if len(kids) > 12 else ''}")
            if tops:
                shown = ", ".join(tops[:8])
                print(f"      files: {shown}"
                      f"{f' (+{len(tops)-8})' if len(tops) > 8 else ''}")
            exts: Dict[str, int] = {}
            deep: Dict[str, int] = {}
            for i, f in enumerate(cand.rglob("*")):
                if i > 5000:
                    break
                if f.is_file():
                    exts[f.suffix or "<none>"] = exts.get(f.suffix or "<none>", 0) + 1
                    rel = f.relative_to(cand).parent.as_posix()
                    deep[rel] = deep.get(rel, 0) + 1
            if exts:
                top = sorted(exts.items(), key=lambda kv: -kv[1])[:5]
                print(f"      ext  : {', '.join(f'{k}x{v}' for k, v in top)}")
            busiest = sorted(deep.items(), key=lambda kv: -kv[1])[:3]
            for rel, n in busiest:
                if rel and rel != ".":
                    print(f"      {n:>5} files under {rel}")
        print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default=",".join(TABLE2),
                   help=f"comma-separated (default: the Table 2 eight). "
                        f"Known: {','.join(sorted(_SPEC))}")
    p.add_argument("--quiet", action="store_true",
                   help="print only failures")
    p.add_argument("--show-layout", action="store_true",
                   help="print what each corpus directory actually contains, "
                        "then exit. Use this instead of guessing when a "
                        "benchmark will not resolve.")
    p.add_argument("--roots", default="/group-volume/datasets",
                   help="with --show-layout: comma-separated corpus roots")
    args = p.parse_args()

    if args.show_layout:
        return _show_layout(args)

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
        # The subset guard has to apply to the CONFIGURED path too, not only
        # to the suggestion. MMLU_DATA_DIR pointing straight at
        # .../mmlu/computer_security satisfies every marker and covers one
        # subject of 57.
        if not missing and not _config_ok(b, root, needs):
            bad += 1
            print(f"FAIL {b:<10} {raw}")
            print(f"     this is ONE subset of {b} — sibling directories hold "
                  f"the same files, so this covers a fraction of the "
                  f"benchmark.")
            alt = _suggest(root.parent if root.parent != root else root, needs, b)
            if alt is not None:
                print(f"     use:  export {var}={alt}")
            else:
                print("     no whole-dataset copy on disk — build one:")
                print(f"       python scripts/prepare_eval_data.py --apply --only {b}")
            continue
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
            print("     contains: <empty>")
        alt = _suggest(root, needs, b)
        if alt is not None:
            print("     found it here instead — fix with:")
            print(f"       export {var}={alt}")

    print()
    if bad:
        print(f"{bad} of {len(names)} benchmark(s) not usable. Eval would fail "
              f"on those, so it will refuse to start.")
        print("Fixes, in the order to try them:")
        print("  1. scripts/download_<bench>.sh <dir> — humaneval, tydiqa, xquad,")
        print("     svamp, mbpp each have one, and each writes exactly the")
        print("     layout its evaluator opens. This is the intended path.")
        print("  2. scripts/prepare_eval_data.py --apply — mmlu and bbh have no")
        print("     downloader; this builds them from the clones on disk.")
        print("  3. TAG_BENCH_ROOTS=/parent/of/corpora source scripts/gpu_cloud/env.sh")
        print("     — if the corpus is on this box under a path not searched.")
        return 1
    print(f"all {len(names)} benchmark(s) ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
