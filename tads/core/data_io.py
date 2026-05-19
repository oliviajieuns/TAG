"""Schema-agnostic raw-record loader for Alpaca-GPT4 / instruction-data files.

Supports the same three formats `datasets.load_dataset` auto-detects from
extension:
    .json     → JSON list (or top-level dict, but we reject that).
    .jsonl    → one JSON object per line.
    .parquet  → pandas/pyarrow.

Single helper used by every baseline that needs to recover raw text after
the tokenised HF Dataset is built — `tokenize_alpaca` discards the raw
strings, but SelectIT / Q2Q / AlpaGasus need them for scoring/matching,
and NAIT's `_ensure_seed_file` needs them to build the seed JSON. Reading
through this module guarantees per-index alignment with the tokenised
side (both sort the same glob and `load_dataset('json'/'parquet')`
preserves on-disk record order).
"""
from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _read_json_or_jsonl(path: Path) -> List[dict]:
    """JSON-list vs JSONL auto-detect by first non-whitespace char."""
    with open(path) as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise TypeError(
                    f"{path}: expected JSON list, got {type(data).__name__}"
                )
            return data
        out: List[dict] = []
        for lineno, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (file {path}, line {lineno})", e.doc, e.pos,
                )
        return out


def _read_parquet(path: Path) -> List[dict]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"Reading {path} as parquet requires pandas + pyarrow "
            f"(both already in requirements.txt). Got ImportError: {e}"
        ) from e
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def read_records(path: str) -> List[dict]:
    """Load a list of dicts from one file, auto-detecting JSON/JSONL/Parquet."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        return _read_parquet(p)
    if suf in (".json", ".jsonl", ".ndjson"):
        return _read_json_or_jsonl(p)
    # Unknown extension — fall back to text sniff (handles e.g. `.txt`
    # mirrors that are actually JSONL). Parquet is a binary magic; sniff
    # via the first byte to avoid misclassifying that as JSONL.
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == b"PAR1":
        return _read_parquet(p)
    return _read_json_or_jsonl(p)


def read_records_glob(glob_spec: str) -> List[dict]:
    """Resolve a glob and concat records from each matched file (sorted order)."""
    matches = sorted(_glob.glob(glob_spec))
    if not matches:
        raise FileNotFoundError(
            f"data_io: glob {glob_spec!r} matched no files."
        )
    out: List[dict] = []
    for p in matches:
        out.extend(read_records(p))
    return out
