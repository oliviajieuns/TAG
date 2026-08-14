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

from tag.data.sft_prompts import tokenize_alpaca

logger = logging.getLogger(__name__)


def _lima_record_to_alpaca(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a LIMA row into {instruction, input, output}.

    Accepts the following input layouts:
      A. LIMA canonical: {"conversations": [user_str, asst_str], "source": ...}
      B. ShareGPT-style: {"conversations": [{"value": str}, ...]}
      C. ChatML / OpenAI: {"conversations": [{"role": "user", "content": str}, ...]}
         (or under key "messages" — same shape)
      D. Flat Alpaca:    {"instruction": str, "input"?: str, "output"|"response": str}
      E. prompt-pair:    {"prompt"|"text": str, "completion"|"response"|"output": str}

    Raises KeyError with the offending record's full key list + a small
    JSON preview so the user can see the schema in the error traceback
    (no need to add separate print statements).
    """
    # (A)/(B)/(C) — `conversations` or `messages` key with a list value.
    conv_key = (
        "conversations" if "conversations" in rec and isinstance(rec["conversations"], list)
        else "messages" if "messages" in rec and isinstance(rec["messages"], list)
        else None
    )
    if conv_key is not None:
        c = rec[conv_key]
        if len(c) < 2:
            raise ValueError(
                f"LIMA {conv_key} needs >=2 turns; got {len(c)}. record={rec!r:.200}"
            )

        def _txt(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                # ShareGPT / ChatML / OpenAI / random-mirror variations.
                for k in ("value", "content", "text", "message"):
                    if k in v and isinstance(v[k], str):
                        return v[k]
                return ""
            return str(v)

        return {
            "instruction": _txt(c[0]),
            "input": "",
            "output": _txt(c[1]),
        }
    # (D) Flat Alpaca-style.
    if "instruction" in rec and ("output" in rec or "response" in rec):
        return {
            "instruction": rec["instruction"],
            "input": rec.get("input", "") or "",
            "output": rec.get("output") or rec.get("response"),
        }
    # (E) prompt-completion pair.
    _prompt_key = next((k for k in ("prompt", "text") if k in rec), None)
    _comp_key = next((k for k in ("completion", "response", "output") if k in rec), None)
    if _prompt_key and _comp_key:
        return {
            "instruction": rec[_prompt_key],
            "input": "",
            "output": rec[_comp_key],
        }
    # Nothing matched — surface schema + a short preview so the user sees
    # exactly what shape we got in the error message (no separate debug
    # print needed — the traceback itself is the diagnostic).
    import json as _json
    preview = _json.dumps(rec, ensure_ascii=False)[:300]
    raise KeyError(
        f"LIMA record schema not recognised.\n"
        f"  keys = {list(rec.keys())}\n"
        f"  first 300 chars = {preview}\n"
        f"Supported: 'conversations' / 'messages' (list), 'instruction'+'output',\n"
        f"or 'prompt'/'text' + 'completion'/'response'/'output'. If your mirror\n"
        f"uses a different key naming, normalise it once (jq / sed) before\n"
        f"setting LIMA_DATA_FILES."
    )


def _load_local_lima_records(spec: str) -> list:
    """Load via the shared helper — supports JSON-list / JSONL / Parquet
    (audit fix: previously json/jsonl only, parquet mirrors raised
    ValueError even when the file was present on disk)."""
    from tag.core.data_io import read_records_glob
    return read_records_glob(spec)


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
            semantics as `tag.data.alpaca.build_alpaca_dataset`.
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
                "LIMA: HF_DATASETS_OFFLINE=1 and `data_files` is None — "
                "the local-mirror path resolved to no value, so we'd have "
                "to hit the HF hub, but offline mode blocks that.\n"
                f"  $LIMA_DATA_FILES         = {os.environ.get('LIMA_DATA_FILES')!r}\n"
                f"  $HF_DATASETS_OFFLINE     = {os.environ.get('HF_DATASETS_OFFLINE')!r}\n"
                "Fix one of:\n"
                "  1) pass `--data_files /path/to/lima.jsonl` on the training\n"
                "     command (most reliable — bypasses env entirely)\n"
                "  2) `export LIMA_DATA_FILES=/path/to/lima.jsonl` in the SAME\n"
                "     shell session as the python invocation (nohup / new\n"
                "     tmux session need their own export)\n"
                "  3) `HF_DATASETS_OFFLINE=0` inline + huggingface-cli login\n"
                "     to fetch from HF hub at runtime"
            )
        logger.info("LIMA: fetching GAIR/lima from HF hub (gated dataset)")
        try:
            raw = load_dataset("GAIR/lima", split="train", cache_dir=cache_dir)
        except Exception as e:
            # Surface the friendly hint above the raw datasets traceback;
            # the gated 401 is otherwise opaque to users new to HF auth.
            raise RuntimeError(
                "LIMA: failed to fetch GAIR/lima from HF hub. This dataset is "
                "GATED — you must (1) `huggingface-cli login` with a personal "
                "access token, and (2) visit https://huggingface.co/datasets/GAIR/lima "
                "and click 'Agree and access'. Alternatively, download a local "
                "mirror and set LIMA_DATA_FILES=/path/to/lima.{json,jsonl}.\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
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
