"""LIMA dataset loading + tokenisation.

GAIR/lima ships rows as `{"conversations": [user_text, assistant_text], "source": str}`
(strings inside a 2-element list, not ShareGPT dicts). We map that into the
{instruction, input, output} schema `tokenize_alpaca` already understands so
the rest of our SFT path works unchanged.

Loading order:
    1. `data_files` (CLI / cfg) if given      → local parquet / json / jsonl.
    2. HF hub `GAIR/lima` (gated; needs `huggingface-cli login` + dataset
       access acceptance at https://huggingface.co/datasets/GAIR/lima).
       Refuses to network-fetch when HF_DATASETS_OFFLINE=1; in that case
       set LIMA_DATA_FILES to a local mirror.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset

from tads.data.sft_prompts import tokenize_alpaca

logger = logging.getLogger(__name__)


def _lima_record_to_alpaca(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a LIMA row into {instruction, input, output}."""
    if "conversations" in rec and isinstance(rec["conversations"], list):
        c = rec["conversations"]
        if len(c) < 2:
            raise ValueError(f"LIMA conversations needs >=2 turns; got {len(c)}")
        # Element may be raw string OR {"value": str}; accept both.
        def _txt(v):
            return v if isinstance(v, str) else v.get("value", v.get("content", ""))
        return {
            "instruction": _txt(c[0]),
            "input": "",
            "output": _txt(c[1]),
        }
    if "instruction" in rec and ("output" in rec or "response" in rec):
        return {
            "instruction": rec["instruction"],
            "input": rec.get("input", "") or "",
            "output": rec.get("output") or rec.get("response"),
        }
    raise KeyError(
        f"Cannot map LIMA record to Alpaca schema; keys={list(rec.keys())}"
    )


def _load_local_lima_records(spec: str) -> list:
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(f"LIMA: data_files glob {spec!r} matched no files.")
    records: list = []
    for p in matches:
        low = p.lower()
        if low.endswith(".jsonl"):
            with open(p) as h:
                for ln in h:
                    ln = ln.strip()
                    if ln:
                        records.append(json.loads(ln))
        elif low.endswith(".json"):
            with open(p) as h:
                data = json.load(h)
            records.extend(data if isinstance(data, list) else [data])
        else:
            raise ValueError(
                f"LIMA: unsupported file extension on {p} — only .json / .jsonl."
            )
    return records


def build_lima_dataset(
    tokenizer,
    cache_dir: str,
    max_seq_len: int = 512,
    *,
    data_files: Optional[str] = None,
    prompt_style: str = "alpaca_default",
    num_proc: int = 2,
):
    """Return a tokenised, response-masked LIMA dataset (HF Dataset).

    Args:
        tokenizer / cache_dir / max_seq_len / prompt_style / num_proc: same
            semantics as `tads.data.alpaca.build_alpaca_dataset`.
        data_files: optional local file or glob (.json / .jsonl). If absent,
            falls back to HF hub `GAIR/lima` (gated).
    """
    os.makedirs(cache_dir, exist_ok=True)

    if data_files:
        records = _load_local_lima_records(str(data_files))
        normalised = [_lima_record_to_alpaca(r) for r in records]
        logger.info("LIMA: loaded %d records from %s", len(normalised), data_files)
        raw = Dataset.from_list(normalised)
    else:
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        if offline:
            raise RuntimeError(
                "LIMA: HF_DATASETS_OFFLINE=1 and no LIMA_DATA_FILES set. "
                "Either export HF_DATASETS_OFFLINE=0 to fetch GAIR/lima, or "
                "download a local mirror and set LIMA_DATA_FILES."
            )
        logger.info("LIMA: fetching GAIR/lima from HF hub (gated dataset)")
        raw = load_dataset("GAIR/lima", split="train", cache_dir=cache_dir)
        # Schema-normalise rows on the HF Dataset directly.
        raw = raw.map(_lima_record_to_alpaca, remove_columns=raw.column_names)

    # Reuse the canonical tokenisation path so prompt_style behaves
    # identically to Alpaca / NAIT / SelectIT runs.
    return raw.map(
        lambda ex: tokenize_alpaca(ex, tokenizer, max_seq_len, prompt_style=prompt_style),
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="tokenize_lima",
    )
