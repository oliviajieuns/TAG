"""Dump the actual schema + a row sample for each downloaded eval dataset.

Use this when an evaluator complains about a schema mismatch — the output
tells you whether the data you have is canonical-shape, an HF
auto-converted parquet branch variant, a different mirror, or a
completely wrong dataset.

Usage:
    source scripts/setup_env.sh   # exports the *_DATA_DIR vars
    python scripts/inspect_eval_data.py

Or override per-bench paths:
    MMLU_PRO_DATA_DIR=/your/path python scripts/inspect_eval_data.py mmlu_pro
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


def _preview_cell(v):
    """Render a cell value for compact display — full strings, list/array
    head + length, dict keys."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        head = list(v)[:2]
        return f"<{type(v).__name__} len={len(v)}: {head}...>"
    if isinstance(v, dict):
        return f"<dict keys={sorted(v.keys())[:3]}...>"
    if hasattr(v, "__len__") and not isinstance(v, str):
        try:
            head = list(v)[:2]
            return f"<{type(v).__name__} len={len(v)}: {head}...>"
        except Exception:
            return f"<{type(v).__name__}>"
    if isinstance(v, str) and len(v) > 120:
        return v[:120] + "..."
    return v


def _dump_parquet(path: Path, label: str) -> None:
    if not path.exists():
        print(f"  [missing] {label}: {path}")
        return
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [error] {label}: {path}\n      → {type(e).__name__}: {e}")
        return
    print(f"  [ok] {label}: {path}")
    print(f"      shape:   {df.shape}")
    print(f"      columns: {sorted(df.columns)}")
    print(f"      dtypes:  {dict(df.dtypes.astype(str))}")
    if not df.empty:
        sample = {k: _preview_cell(v) for k, v in df.iloc[0].to_dict().items()}
        print(f"      row[0]:  {sample}")


def inspect_mmlu_pro():
    print("=" * 70)
    print("MMLU-Pro")
    print("=" * 70)
    d = Path(os.environ.get("MMLU_PRO_DATA_DIR", "/group-volume/IT-datasets/mmlu_pro"))
    print(f"  dir: {d}")
    if not d.is_dir():
        print("  [missing] directory not found")
        return
    for stem in ("test", "validation"):
        candidates = sorted(d.glob(f"{stem}*.parquet"))
        if not candidates:
            print(f"  [missing] no {stem}-*.parquet under {d}")
            continue
        for p in candidates:
            _dump_parquet(p, f"{stem}")


def inspect_svamp():
    print("=" * 70)
    print("SVAMP")
    print("=" * 70)
    d = Path(os.environ.get("SVAMP_DATA_DIR", "/group-volume/IT-datasets/svamp"))
    print(f"  dir: {d}")
    if not d.is_dir():
        print("  [missing] directory not found")
        return
    candidates = sorted(d.glob("test*.parquet")) + sorted(d.glob("*.parquet"))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        _dump_parquet(p, "test")


def inspect_mbpp():
    print("=" * 70)
    print("MBPP")
    print("=" * 70)
    d = Path(os.environ.get("MBPP_DATA_DIR", "/group-volume/IT-datasets/mbpp"))
    print(f"  dir: {d}")
    if not d.is_dir():
        print("  [missing] directory not found")
        return
    for cfg in ("sanitized", "full"):
        sub = d / cfg
        if not sub.is_dir():
            print(f"  [skip] no {cfg}/ subdir")
            continue
        for stem in ("test", "prompt"):
            candidates = sorted(sub.glob(f"{stem}*.parquet"))
            if not candidates:
                print(f"  [missing] no {cfg}/{stem}-*.parquet")
                continue
            for p in candidates:
                _dump_parquet(p, f"{cfg}/{stem}")


def inspect_xquad():
    print("=" * 70)
    print("XQuAD")
    print("=" * 70)
    d = Path(os.environ.get("XQUAD_DATA_DIR", "/group-volume/IT-datasets/xquad"))
    print(f"  dir: {d}")
    if not d.is_dir():
        print("  [missing] directory not found")
        return
    json_files = sorted(d.glob("xquad.*.json"))
    if not json_files:
        print(f"  [missing] no xquad.<lang>.json under {d}")
        return
    print(f"  found {len(json_files)} language file(s): "
          f"{[p.name for p in json_files]}")
    # Inspect the first one (typically English) for structure.
    p = next((q for q in json_files if q.stem.endswith(".en")), json_files[0])
    try:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"  [error] {p}: {type(e).__name__}: {e}")
        return
    print(f"  [ok] {p.name}")
    print(f"      top-level keys: {list(doc.keys())}")
    if "data" in doc and doc["data"]:
        article = doc["data"][0]
        print(f"      data[0] keys: {list(article.keys())}")
        if "paragraphs" in article and article["paragraphs"]:
            para = article["paragraphs"][0]
            print(f"      data[0].paragraphs[0] keys: {list(para.keys())}")
            if "qas" in para and para["qas"]:
                qa = para["qas"][0]
                print(f"      qa[0] keys: {list(qa.keys())}")
                ans = qa.get("answers", [])
                if ans:
                    print(f"      qa[0].answers[0] keys: {list(ans[0].keys())}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["mmlu_pro", "svamp", "mbpp", "xquad"]
    fns = {
        "mmlu_pro": inspect_mmlu_pro,
        "svamp": inspect_svamp,
        "mbpp": inspect_mbpp,
        "xquad": inspect_xquad,
    }
    for t in targets:
        fn = fns.get(t)
        if fn is None:
            print(f"[skip] unknown target: {t!r} (known: {sorted(fns)})")
            continue
        fn()
        print()


if __name__ == "__main__":
    main()
