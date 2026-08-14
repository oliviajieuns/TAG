"""Alpaca instruction-tuning dataset loading and tokenisation.

Supports both Hugging Face hub names (``tatsu-lab/alpaca``) and local
Parquet files. Tokenises with the prompt-style-aware
:func:`tag.data.sft_prompts.tokenize_alpaca` so that the training-time
formatting matches the model family used at evaluation time.
"""
from __future__ import annotations

import contextlib
import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .sft_prompts import tokenize_alpaca

logger = logging.getLogger(__name__)


# Coarse inter-process lock around the dataset-build sequence. Concurrent
# training launches that share a cache_dir — multiple DDP ranks within one
# job, OR several `python -m tag.train --config <...>` invocations fired
# in parallel against the same model_key/prompt_style — would otherwise
# race on (a) HF `load_dataset` reading the raw shards into Arrow, and
# (b) `.map(_tokenize, num_proc=4)` writing the tokenised cache file.
# HF datasets uses its own filelocks internally, but they're per-file and
# don't cover the fingerprint→path resolution window; symptoms we've
# observed:
#   - "ArrowInvalid: ... cache file truncated" when two workers wrote the
#     same shard simultaneously,
#   - "DatasetGenerationError: ... .incomplete left behind" when one
#     process was killed mid-write,
#   - intermittent silent stalls inside .map's multiprocess fork.
# This module-level lock serialises only the BUILD phase. Once the cache
# is warm, every subsequent call falls through .map's cache-hit path
# (microseconds) so the lock cost amortises to zero across runs.
_LOCK_TIMEOUT_SEC = 4 * 60 * 60  # 4h ceiling — first-time tokenise of 52K
                                 # Alpaca samples with num_proc=4 is ~2min
                                 # on a fast filesystem; the cap exists to
                                 # surface a wedged lock instead of hanging.


def _dataset_build_lock(cache_dir: str):
    """Return a context manager that holds an advisory lock on cache_dir.

    Uses ``filelock.FileLock`` (transitive dep via ``datasets``); falls
    back to a no-op context if the lib isn't importable so a partial
    install never blocks training. The lockfile lives inside cache_dir
    so it shares the FS's lock semantics with the data it protects —
    notably, NFS hosts that disable fcntl byte-range locks will surface
    that via filelock's own diagnostic, not via mysterious arrow errors.
    """
    try:
        from filelock import FileLock  # type: ignore
    except Exception as e:  # filelock missing / broken install
        logger.warning(
            "filelock unavailable (%s); proceeding without cross-process "
            "serialisation. Concurrent dataset builds on the same cache "
            "may race on first-time tokenisation.", e,
        )
        return contextlib.nullcontext()

    lock_path = os.path.join(cache_dir, ".tag_dataset_build.lock")
    return _LoggingFileLock(FileLock(lock_path, timeout=_LOCK_TIMEOUT_SEC), lock_path)


class _LoggingFileLock:
    """Wrap FileLock to log acquire / release with elapsed wall time.

    When several experiments race for first-time tokenisation, the loser
    can wait minutes inside acquire() with no log output, which looks
    indistinguishable from a hang. Emit a single info line on entry and
    on exit so the wait is visible.
    """

    def __init__(self, lock, path: str):
        self._lock = lock
        self._path = path
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        logger.info("Acquiring dataset-build lock: %s", self._path)
        self._lock.acquire()
        waited = time.time() - self._t0
        if waited > 1.0:
            logger.info(
                "Acquired dataset-build lock after %.1fs wait", waited,
            )
        else:
            logger.info("Acquired dataset-build lock (no contention)")
        return self

    def __exit__(self, *exc):
        try:
            self._lock.release()
        finally:
            logger.info(
                "Released dataset-build lock | held=%.1fs",
                time.time() - self._t0,
            )


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
        # Sibling fallback: the canonical case is an HF re-download where the
        # filename hash changed (e.g. train-00000-of-00001-<hash>.json). Look
        # for files whose basename STEM matches the requested basename's
        # leading word — NOT "every .json in this directory", which would
        # silently sweep in valid.json / test.json / cached preprocessing
        # artifacts and contaminate the training set.
        want_stem = os.path.splitext(os.path.basename(spec))[0]
        # Take the first hyphen-separated token, e.g. "train" out of
        # "train-00000-of-00001-deadbeef". Falls back to the whole stem
        # when the filename has no hyphen.
        head = want_stem.split("-", 1)[0] if "-" in want_stem else want_stem
        for pat in (f"{head}*.json", f"{head}*.jsonl", f"{head}*.parquet"):
            matches = sorted(glob.glob(os.path.join(parent, pat)))
            if matches:
                logger.warning(
                    "ALPACA_DATA_FILES=%r not found; using sibling fallback "
                    "matching basename prefix %r → %d file(s): %s",
                    spec, head, len(matches),
                    matches if len(matches) <= 3 else f"{matches[:3]} +({len(matches) - 3} more)",
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

    # Serialise concurrent builders that share this cache_dir. The lock is
    # released as soon as we return; cache-hit calls after the first build
    # acquire and release in microseconds and never block.
    with _dataset_build_lock(cache_dir):
        return _build_alpaca_dataset_locked(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            dataset_name=dataset_name,
            data_files=data_files,
            prompt_style=prompt_style,
            num_proc=num_proc,
        )


def _build_alpaca_dataset_locked(
    tokenizer,
    cache_dir: str,
    max_seq_len: int,
    *,
    dataset_name: Optional[str],
    data_files: Optional[str],
    prompt_style: str,
    num_proc: int,
):
    """Inner builder — caller already holds the cross-process build lock."""
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
            cache_dir=cache_dir,
        )
    elif dataset_name:
        # Refuse to silently hit the HF hub when the trainer is in its default
        # offline mode (see tag.train.main). Cluster nodes typically have no
        # outbound HTTPS, and the resulting download attempt corrupts the HF
        # cache lockfiles, surfacing as an opaque cache error several minutes
        # later. Force the user to either (a) set ALPACA_DATA_FILES, or (b)
        # explicitly opt into the network by exporting HF_DATASETS_OFFLINE=0.
        offline = (
            os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
            or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        )
        if offline:
            raise FileNotFoundError(
                "ALPACA_DATA_FILES is unset / unresolved AND HF_DATASETS_OFFLINE=1.\n"
                "Refusing to download " + repr(dataset_name) + " from the HF hub.\n"
                "Fix one of:\n"
                "  1. export ALPACA_DATA_FILES=/abs/path/to/file_or_glob.json (preferred)\n"
                "  2. export HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 (re-enables hub access)"
            )
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

    # Cache-bypass switch: when TAG_FRESH_DATA_CACHE=1, force-re-tokenise
    # rather than reusing the HF `Dataset.map` fingerprint cache. The
    # fingerprint is derived from (raw dataset hash, _tokenize closure
    # bytes), and a code change inside tokenize_alpaca SHOULD bust it —
    # but in practice equivalent-bytes closure variations (e.g., default
    # arg drift, import-order changes that move the byte-encoded code
    # object) have served stale tokenisations and silently distorted
    # training. Set the env var to force a fresh pass; otherwise the
    # cache is used as before (saves ~1-2 minutes on Alpaca-52K).
    _fresh_cache = os.environ.get("TAG_FRESH_DATA_CACHE", "0") == "1"
    ds = raw.map(
        _tokenize,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"Tokenising Alpaca ({prompt_style})",
        load_from_cache_file=not _fresh_cache,
    )
    ds.set_format("torch")
    logger.info(
        "Alpaca dataset built | n=%d | max_seq_len=%d | style=%s",
        len(ds), max_seq_len, prompt_style,
    )
    return ds
