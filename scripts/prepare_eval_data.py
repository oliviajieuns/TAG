#!/usr/bin/env python
"""Convert the HF-clone benchmark corpora into the layout each evaluator reads.

    python scripts/prepare_eval_data.py            # look, convert nothing
    python scripts/prepare_eval_data.py --apply

The corpora on /group-volume/datasets are git clones of HF dataset repos.
Five of the eight are laid out differently from what tag/evals/*.py opens:

    mmlu       57 per-subject configs, no `all`  -> one concatenated pair
    bbh        <task>/test-*.parquet             -> <task>.json {examples:[...]}
    humaneval  openai_humaneval/*.parquet        -> HumanEval.jsonl.gz
    tydiqa     dev/*.jsonl, one per language     -> one flat validation.jsonl
    xquad      xquad.<lang>/*.parquet            -> xquad.<lang>.json (SQuAD)

The evaluators are deliberately NOT changed. They define how each benchmark
is scored, and the numbers already in the paper came out of them; editing a
loader to accept a new layout risks moving a score for reasons that have
nothing to do with the method. Converting the data leaves scoring identical.

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
import gzip
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def prep_humaneval(src: Path, out: Path) -> str:
    """HumanEval.jsonl.gz, the file openai/human-eval publishes."""
    cand = [src / "openai_humaneval", src]
    files: List[Path] = []
    for c in cand:
        if c.is_dir():
            files = _parquets(c)
            if files:
                break
    if not files:
        raise ValueError(f"no parquet under {src} or {src}/openai_humaneval")
    df = _read_concat(files)
    need = ("task_id", "prompt", "entry_point", "test")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"humaneval: missing column(s) {missing}; "
                         f"has {sorted(df.columns)}")
    keep = [c for c in ("task_id", "prompt", "canonical_solution", "test",
                        "entry_point") if c in df.columns]
    with gzip.open(out / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as f:
        for _, r in df[keep].iterrows():
            f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
    return f"{len(df)} problems"


def prep_tydiqa(src: Path, out: Path) -> str:
    """One flat validation.jsonl across every language.

    The clone splits the dev set one file per language, and the evaluator
    takes a single dev file — pointing it at the directory would silently
    score one language out of nine. Parsing is delegated to the evaluator's
    own reader so the records mean exactly what it expects, and the
    language is recovered downstream from each QA id.
    """
    from tag.evals.tydiqa import _parse_squad_file

    dev_dir = src / "dev" if (src / "dev").is_dir() else src
    files = sorted(f for f in dev_dir.rglob("*")
                   if f.is_file() and f.suffix in (".json", ".jsonl")
                   and not any(p in _SKIP_DIRS for p in f.parts)
                   and "dataset_infos" not in f.name)
    if not files:
        raise ValueError(f"no json/jsonl under {dev_dir}")
    records: List[Dict[str, Any]] = []
    used = 0
    for f in files:
        try:
            recs = _parse_squad_file(str(f))
        except Exception as e:  # noqa: BLE001 — a stray file is not fatal
            print(f"      ! skipping {f.name}: {e}", file=sys.stderr)
            continue
        if recs:
            records.extend(recs)
            used += 1
    if not records:
        raise ValueError(f"every file under {dev_dir} parsed to zero records")
    with open(out / "validation.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs = sorted({str(r.get("language", "?")) for r in records})
    return f"{used} file(s), {len(records)} questions, {len(langs)} language(s): {','.join(langs)}"


def prep_xquad(src: Path, out: Path) -> str:
    """xquad.<lang>.json, SQuAD-nested, one per language."""
    langs = [d for d in _subdirs(src) if d.name.startswith("xquad.") and _parquets(d)]
    if not langs:
        raise ValueError(f"no xquad.<lang> directories with parquet under {src}")
    total = 0
    for d in langs:
        lang = d.name.split(".", 1)[1]
        df = _read_concat(_parquets(d))
        missing = [c for c in ("context", "question", "answers") if c not in df.columns]
        if missing:
            raise ValueError(f"xquad/{lang}: missing column(s) {missing}; "
                             f"has {sorted(df.columns)}")
        by_ctx: Dict[str, List[Dict[str, Any]]] = {}
        for _, r in df.iterrows():
            ans = r["answers"]
            texts = list(ans.get("text", [])) if isinstance(ans, dict) else list(ans)
            starts = list(ans.get("answer_start", [])) if isinstance(ans, dict) else []
            qas = by_ctx.setdefault(str(r["context"]), [])
            qas.append({
                "id": str(r.get("id", f"{lang}-{len(qas)}")),
                "question": str(r["question"]),
                "answers": [
                    {"text": str(t),
                     "answer_start": int(starts[i]) if i < len(starts) else 0}
                    for i, t in enumerate(texts)
                ],
            })
        doc = {"data": [{"title": f"xquad.{lang}", "paragraphs": [
            {"context": c, "qas": q} for c, q in by_ctx.items()]}]}
        (out / f"xquad.{lang}.json").write_text(
            json.dumps(doc, ensure_ascii=False))
        total += len(df)
    return f"{len(langs)} language(s), {total} questions"


_PREP = {
    "mmlu": prep_mmlu,
    "bbh": prep_bbh,
    "humaneval": prep_humaneval,
    "tydiqa": prep_tydiqa,
    "xquad": prep_xquad,
}
# Already in a shape the evaluators read; converting them would only add a
# second copy to keep in sync.
_ALREADY_OK = ("svamp", "gsm8k", "mbpp")


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
    unknown = [n for n in names if n not in _PREP]
    if unknown:
        print(f"unknown benchmark(s): {unknown}; known: {sorted(_PREP)} "
              f"(already fine and not converted: {list(_ALREADY_OK)})",
              file=sys.stderr)
        return 2

    print(f"src : {src_root}")
    print(f"out : {out_root}")
    print(f"not converted (already readable): {', '.join(_ALREADY_OK)}")
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
