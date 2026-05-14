"""Alpaca instruction-tuning dataset loading and tokenisation.

Supports both Hugging Face hub names (``tatsu-lab/alpaca``) and local
Parquet files. Tokenises with the prompt-style-aware
:func:`tads.data.sft_prompts.tokenize_alpaca` so that the training-time
formatting matches the model family used at evaluation time.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .sft_prompts import tokenize_alpaca

logger = logging.getLogger(__name__)


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
        logger.info("Loading Alpaca from local file(s): %s", data_files)
        raw = load_dataset("parquet", data_files=data_files, split="train")
    elif dataset_name:
        logger.info("Loading Alpaca from HF hub: %s", dataset_name)
        raw = load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    else:
        raise ValueError(
            "Neither `data_files` nor `dataset_name` is set. "
            "Set ALPACA_DATA_FILES env var (local parquet) "
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
