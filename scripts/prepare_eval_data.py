#!/usr/bin/env python
"""Build MMLU and BBH from the HF clones on disk — the two with no downloader.

    python scripts/prepare_eval_data.py            # look, convert nothing
    python scripts/prepare_eval_data.py --apply

This script covers MMLU and BBH and nothing else, on purpose. Every other
benchmark has scripts/download_<bench>.sh, which fetches from the canonical
upstream and writes exactly the layout its evaluator opens; converting a
local clone instead would be both redundant and a DIFFERENT artifact — the
official google/xquad JSON is what tag/evals/xquad.py was written against.
Asking for one of those here prints the downloader to use and exits.

MMLU and BBH have no downloader. setup_env.sh assumes both are already on
the cluster, and the clones that are there do not match:

    mmlu   57 per-subject configs and no `all`  -> one concatenated pair
    bbh    <task>/test-*.parquet                -> <task>.json {examples:[...]}

No evaluator is changed. They define how each benchmark is scored and the
published numbers came out of them; editing a loader to accept a new layout
risks moving a score for reasons unrelated to the method.

Output goes to a directory of our own (default $TAG_WORKSPACE/eval-data),
never into the shared corpus tree. env.sh searches it first.

MMLU deserves a note: this clone has the 57 subject configs and no `all`, so
`all` is rebuilt by concatenating them in sorted subject order. That IS what
the upstream `all` config is, but the row order differs from upstream's, and
anything that depends on order (a --limit truncation) would differ with it.
Use --limit only for smoke tests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SKIP_DIRS = {".git", ".github", "__pycache__"}


def _subdirs(d: Path) -> List[Path]:
    return sorted(c for c in d.iterdir() if c.is_dir() and c.name not in _SKIP_DIRS)


def _parquets(d: Path, prefix: str = "") -> List[Path]:
    return sorted(f for f in d.glob(f"{prefix}*.parquet") if f.is_file())


def _read_concat(files: List[Path]):
    import pandas as pd
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# one function per benchmark: (src, out) -> summary string, raises on failure
# --------------------------------------------------------------------------

def prep_mmlu(src: Path, out: Path) -> str:
    """Rebuild the `all` config from the per-subject ones."""
    import pandas as pd

    subjects = [d for d in _subdirs(src) if _parquets(d)]
    if not subjects:
        raise ValueError(f"no subject directories with parquet under {src}")
    if any(d.name == "all" for d in subjects):
        subjects = [d for d in subjects if d.name == "all"]

    for split, want in (("test", "test"), ("dev", "dev")):
        frames = []
        for d in subjects:
            fs = _parquets(d, f"{want}-") or _parquets(d, f"{want}.")
            if not fs:
                continue
            df = _read_concat(fs)
            if "subject" not in df.columns:
                df = df.assign(subject=d.name)
            frames.append(df)
        if not frames:
            raise ValueError(f"no {split} parquet under any subject of {src}")
        allf = pd.concat(frames, ignore_index=True)
        missing = [c for c in ("question", "choices", "answer") if c not in allf.columns]
        if missing:
            raise ValueError(f"mmlu {split}: missing column(s) {missing}; "
                             f"has {sorted(allf.columns)}")
        allf.to_parquet(out / f"{split}-00000-of-00001.parquet")
    n_test = len(pd.read_parquet(out / "test-00000-of-00001.parquet"))
    n_dev = len(pd.read_parquet(out / "dev-00000-of-00001.parquet"))
    return f"{len(subjects)} subject(s) -> test {n_test}, dev {n_dev}"


def prep_bbh(src: Path, out: Path) -> str:
    """One {'examples': [{'input','target'}]} JSON per task."""
    tasks = [d for d in _subdirs(src) if _parquets(d)]
    if not tasks:
        raise ValueError(f"no task directories with parquet under {src}")
    total = 0
    for d in tasks:
        df = _read_concat(_parquets(d))
        missing = [c for c in ("input", "target") if c not in df.columns]
        if missing:
            raise ValueError(f"bbh/{d.name}: missing column(s) {missing}; "
                             f"has {sorted(df.columns)}")
        examples = [{"input": str(r["input"]), "target": str(r["target"])}
                    for _, r in df.iterrows()]
        (out / f"{d.name}.json").write_text(
            json.dumps({"examples": examples}, ensure_ascii=False))
        total += len(examples)
    return f"{len(tasks)} task(s), {total} examples"


_PREP = {"mmlu": prep_mmlu, "bbh": prep_bbh}

# BBH built from the HF parquet has no cot-prompts/, and the evaluator then
# falls back to a direct-answer baseline that is not what Table 2 reports.
# scripts/download_bbh.sh fetches the upstream repo, which has both.
_PREFER_DOWNLOADER = {"bbh": "scripts/download_bbh.sh"}

# Everything else already has a way in that is better than converting a
# clone, and duplicating it here would mean two paths to keep in agreement.
_HANDLED_ELSEWHERE = {
    "humaneval": "scripts/download_humaneval.sh",
    "tydiqa": "scripts/download_tydiqa.sh",
    "xquad": "scripts/download_xquad.sh",
    "svamp": "scripts/download_svamp.sh (already readable on this box)",
    "mbpp": "scripts/download_mbpp.sh (already readable on this box)",
    "gsm8k": "already readable on this box",
}


def main() -> int:
    ws = os.environ.get("TAG_WORKSPACE", "")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/group-volume/datasets",
                    help="corpus root holding the HF clones")
    ap.add_argument("--out", default=(f"{ws}/eval-data" if ws else ""),
                    help="where to write the converted layout "
                         "(default $TAG_WORKSPACE/eval-data)")
    ap.add_argument("--only", default="", help="comma-separated benchmark names")
    ap.add_argument("--apply", action="store_true", help="actually write")
    args = ap.parse_args()

    if not args.out:
        print("--out is required (or export TAG_WORKSPACE)", file=sys.stderr)
        return 2
    src_root, out_root = Path(args.src), Path(args.out)
    if not src_root.is_dir():
        print(f"no such --src: {src_root}", file=sys.stderr)
        return 2

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(_PREP)
    for n in names:
        if n in _HANDLED_ELSEWHERE:
            print(f"{n}: use {_HANDLED_ELSEWHERE[n]} — it writes the layout "
                  f"the evaluator opens, from upstream.", file=sys.stderr)
            return 2
    for n in names:
        if n in _PREFER_DOWNLOADER:
            print(f"NOTE {n}: {_PREFER_DOWNLOADER[n]} fetches the upstream "
                  f"repo, which also carries cot-prompts/. Building it from "
                  f"the HF parquet here leaves those out, and the evaluator "
                  f"then scores a non-CoT baseline that Table 2 does not use.")
    unknown = [n for n in names if n not in _PREP]
    if unknown:
        print(f"unknown benchmark(s): {unknown}; this script builds "
              f"{sorted(_PREP)}.", file=sys.stderr)
        return 2

    print(f"src : {src_root}")
    print(f"out : {out_root}")
    print()

    if not args.apply:
        for n in names:
            s = src_root / n
            print(f"{'would convert' if s.is_dir() else 'MISSING src  '}  {n:<10} {s}")
        print()
        print("Dry run — nothing written. Re-run with --apply.")
        return 0

    bad = 0
    for n in names:
        src = src_root / n
        if not src.is_dir():
            print(f"[{n}] no such source: {src}", file=sys.stderr)
            bad += 1
            continue
        print(f"[{n}] {src}")
        tmp = out_root / f"{n}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            summary = _PREP[n](src, tmp)
        except Exception as e:  # noqa: BLE001 — one benchmark must not stop the rest
            print(f"   FAILED: {e}", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            bad += 1
            continue
        final = out_root / n
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        print(f"   {summary}")
        print(f"   -> {final}")

    print()
    print(f"{len(names) - bad}/{len(names)} converted.")
    print()
    print("Verify, then run the eval:")
    print(f"  TAG_BENCH_ROOTS={out_root} source scripts/gpu_cloud/env.sh")
    print("  python scripts/check_eval_data.py")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
