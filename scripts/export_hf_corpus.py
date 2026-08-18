#!/usr/bin/env python
"""Materialise an HF `datasets` cache as the JSON file the pipeline reads.

    python scripts/export_hf_corpus.py \\
        --cache-dir /group-volume/data/hf_home/datasets/liangxin___alpaca_gpt4 \\
        --out "$ALPACA_GPT4_JSON"

Alpaca-GPT4 is on this cluster only as an arrow cache under HF_HOME. Every
consumer here — make_corrupted_pool.py, build_alpaca_dataset, the manifest's
corpus fingerprint — reads .json / .jsonl / .parquet, so the corpus has to be
written out once. Doing it once, to a fixed path, also makes it nameable: the
sha256 printed at the end is what a paper should cite as the corpus, since
"Alpaca-GPT4" alone does not distinguish the several mirrors in circulation.

Only the three fields the SFT prompt uses are kept (instruction, input,
output) and their order is preserved, so record i here is record i in the
cache. Anything else in the cache is dropped rather than carried along
silently — a stray column that survived into the pool would change the
tokenised text without appearing in any config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_FIELDS = ("instruction", "input", "output")


def _find_arrow(cache_dir: Path) -> List[Path]:
    files = sorted(cache_dir.rglob("*.arrow"))
    if not files:
        raise SystemExit(
            f"no .arrow files under {cache_dir}. Point --cache-dir at the "
            f"dataset directory inside HF_HOME/datasets (the one containing "
            f"dataset_info.json)."
        )
    return files


def _rows_from_arrow(files: List[Path]) -> List[Dict[str, Any]]:
    try:
        import pyarrow as pa  # noqa: F401
        from datasets import Dataset, concatenate_datasets
    except ImportError as e:
        raise SystemExit(
            f"needs `datasets` + `pyarrow` to read the cache ({e}). "
            f"They are already required by the training pipeline."
        )
    parts = []
    for f in files:
        try:
            parts.append(Dataset.from_file(str(f)))
        except Exception as e:  # noqa: BLE001 — a stray arrow file is not fatal
            print(f"  skipping {f.name}: {e}", file=sys.stderr)
    if not parts:
        raise SystemExit("every .arrow file failed to load")
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    cols = set(ds.column_names)
    missing = [c for c in ("instruction", "output") if c not in cols]
    if missing:
        raise SystemExit(
            f"cache is missing required column(s) {missing}; has {sorted(cols)}. "
            f"This does not look like an Alpaca-shaped corpus."
        )
    out: List[Dict[str, Any]] = []
    for r in ds:
        out.append({k: (r.get(k) or "") for k in _FIELDS if k in cols or k == "input"})
    return out


def _describe(out: Path, rows: List[Dict[str, Any]]) -> None:
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    n = len(rows)
    n_with_input = sum(1 for r in rows if (r.get("input") or "").strip())
    print(f"  path         : {out}")
    print(f"  records      : {n}")
    print(f"  with 'input' : {n_with_input} ({100*n_with_input/max(n,1):.1f}%)")
    print(f"  fields       : {sorted(rows[0]) if n else '<empty>'}")
    print(f"  sha256       : {h.hexdigest()}")
    if n:
        first = rows[0]
        print(f"  first record : instruction={first.get('instruction','')[:60]!r}")
    print()
    print("Cite THIS hash as the corpus — 'Alpaca-GPT4' names several mirrors")
    print("that differ in record count and cleaning.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="",
                   help="HF datasets cache dir for the corpus (contains dataset_info.json)")
    p.add_argument("--out", required=True, help="destination .json")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing destination")
    p.add_argument("--inspect", action="store_true",
                   help="describe an existing --out file and exit, writing "
                        "nothing. Use this before reaching for --force: the "
                        "question is whether the corpus already there is the "
                        "one you want, and the record count and hash answer it.")
    args = p.parse_args()

    cache = Path(args.cache_dir)
    out = Path(args.out)

    if args.inspect:
        if not out.exists():
            print(f"{out} does not exist yet.")
            return 1
        try:
            with open(out, encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"{out} is not readable JSON: {e}", file=sys.stderr)
            return 1
        _describe(out, rows)
        return 0

    if out.exists() and not args.force:
        # Overwriting the corpus under a finished pool would silently
        # invalidate every artifact fingerprinted against it.
        print(f"{out} already exists. Pass --force only if you are sure nothing "
              f"downstream was built from the current one.", file=sys.stderr)
        return 2

    if not args.cache_dir:
        print("--cache-dir is required unless --inspect is given", file=sys.stderr)
        return 2
    files = _find_arrow(cache)
    print(f"reading {len(files)} arrow file(s) under {cache}")
    rows = _rows_from_arrow(files)
    if not rows:
        raise SystemExit("cache produced zero records")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    tmp.replace(out)

    print(f"wrote {out}")
    _describe(out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
