"""Alpaca instruction-tuning dataset loading and tokenisation.

Supports both Hugging Face hub names (``tatsu-lab/alpaca``) and local
Parquet files. Tokenises with the prompt-style-aware
:func:`tads.data.sft_prompts.tokenize_alpaca` so that the training-time
formatting matches the model family used at evaluation time.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .sft_prompts import tokenize_alpaca

logger = logging.getLogger(__name__)


def _resolve_data_files(spec: str) -> Optional[str]:
    """Expand globs / verify the literal path. Returns the resolved spec or None.

    - If ``spec`` contains a glob metachar, expand and return the (comma-joined)
      list of matches; None if zero matches.
    - If ``spec`` is a concrete path that exists, return it unchanged.
    - If ``spec`` is a concrete path that does NOT exist, try sibling globs
      (``*.json``, ``*.jsonl``, ``*.parquet``) in the same directory as a
      fallback — this rescues the common case where the file has a hashed
      suffix (HF dataset shards).
    """
    if not spec:
        return None
    if any(ch in spec for ch in "*?["):
        matches = sorted(glob.glob(spec))
        if matches:
            return ",".join(matches)
        return None
    if os.path.exists(spec):
        return spec
    parent = os.path.dirname(spec) or "."
    if os.path.isdir(parent):
        for pat in ("*.json", "*.jsonl", "*.parquet"):
            matches = sorted(glob.glob(os.path.join(parent, pat)))
            if matches:
                logger.warning(
                    "ALPACA_DATA_FILES=%r not found; using sibling fallback %s (%d files)",
                    spec, os.path.join(parent, pat), len(matches),
                )
                return ",".join(matches)
    return None


def verify_response_marker(tokenizer) -> List[int]:
    """Encode ``### Response:\\n`` and warn if it isn't a recoverable substring.

    Kept as a diagnostic — the tokenisation in :func:`tokenize_alpaca` no
    longer relies on marker search, but logging the marker is useful when
    debugging unfamiliar tokenisers.
    """
    marker = tokenizer.encode("### Response:\n", add_special_tokens=False)
    test = tokenizer.encode(
        "### Instruction:\nfoo\n\n### Response:\nbar",
        add_special_tokens=False,
    )
    found = any(
        test[j : j + len(marker)] == marker
        for j in range(len(test) - len(marker))
    )
    if found:
        logger.info("Response marker verified | marker=%s", marker)
    else:
        logger.warning(
            "Response marker NOT found | marker=%s "
            "(prompt/response split tokenisation is used regardless)",
            marker,
        )
    return marker


def build_alpaca_dataset(
    tokenizer,
    cache_dir: str,
    max_seq_len: int = 512,
    *,
    dataset_name: Optional[str] = "tatsu-lab/alpaca",
    data_files: Optional[str] = None,
    prompt_style: str = "alpaca_default",
    num_proc: int = 4,
):
    """Return a tokenised, response-masked Alpaca dataset (HF Dataset).

    Args:
        tokenizer: HF tokenizer with ``pad_token``/``eos_token`` set.
        cache_dir: HF datasets cache directory.
        max_seq_len: pad / truncate to this length.
        dataset_name: HF hub dataset id; ignored if ``data_files`` is given.
        data_files: local parquet path; takes precedence over ``dataset_name``.
        prompt_style: passed to :func:`tokenize_alpaca`.
        num_proc: ``Dataset.map`` parallel workers.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Normalise empty-string overrides (e.g. from `${oc.env:VAR,}`) to None.
    data_files = data_files or None
    dataset_name = dataset_name or None

    if data_files:
        resolved = _resolve_data_files(str(data_files))
        if not resolved:
            env_val = os.environ.get("ALPACA_DATA_FILES", "<unset>")
            raise FileNotFoundError(
                f"Alpaca local file(s) not found.\n"
                f"  requested data_files = {data_files!r}\n"
                f"  ALPACA_DATA_FILES env = {env_val!r}\n"
                f"Fix one of:\n"
                f"  1. export ALPACA_DATA_FILES=/abs/path/to/file_or_glob.json "
                f"   (then re-run; nothing else to do)\n"
                f"  2. edit scripts/setup_env.sh and source it again\n"
                f"  3. unset ALPACA_DATA_FILES to fall back to HF hub "
                f"({dataset_name or 'liangxin/Alpaca_GPT4'})"
            )
        # Auto-detect HF `load_dataset` builder by the first matched file's extension.
        sample = resolved.split(",")[0].lower()
        if sample.endswith((".json", ".jsonl")):
            fmt = "json"
        elif sample.endswith(".csv"):
            fmt = "csv"
        elif sample.endswith((".txt", ".text")):
            fmt = "text"
        else:
            fmt = "parquet"
        # Pretty-print: collapse to count if many files (glob shards).
        n_files = resolved.count(",") + 1
        display = resolved if n_files <= 3 else f"{sample} … (+{n_files - 1} more)"
        logger.info(
            "Loading Alpaca from local file(s): %s | format=%s | n_files=%d",
            display, fmt, n_files,
        )
        raw = load_dataset(
            fmt,
            data_files=resolved.split(",") if "," in resolved else resolved,
            split="train",
        )
    elif dataset_name:
        logger.info("Loading Alpaca from HF hub: %s", dataset_name)
        raw = load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    else:
        raise ValueError(
            "Neither `data_files` nor `dataset_name` is set. "
            "Set ALPACA_DATA_FILES env var (local parquet / json / jsonl / csv) "
            "or ALPACA_DATASET_NAME (HF hub) — or set them in the YAML config."
        )

    verify_response_marker(tokenizer)

    def _tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_alpaca(
            example,
            tokenizer,
            max_seq_len=max_seq_len,
            prompt_style=prompt_style,
        )

    ds = raw.map(
        _tokenize,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"Tokenising Alpaca ({prompt_style})",
    )
    ds.set_format("torch")
    logger.info(
        "Alpaca dataset built | n=%d | max_seq_len=%d | style=%s",
        len(ds), max_seq_len, prompt_style,
    )
    return ds
