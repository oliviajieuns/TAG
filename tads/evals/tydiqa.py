"""TyDiQA evaluator (Exact-Match on Gold-Passage dev split, 5-shot).

Matches the NAIT paper (Appendix D) setup: paper-faithful 5-shot
gold-passage with demonstrations sampled from the **same language** as
the test example. Demonstrations are loaded from
``tydiqa-goldp-v1.1-train.json``; if absent, the evaluator falls back
to 0-shot and logs a clear warning.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import tydiqa_generation_prefix

logger = logging.getLogger(__name__)


# TyDiQA gold-passage QA ids look like:
#   - HF parquet:                 ``arabic-2387335860751143628-1``  (single dash)
#   - Legacy GCS JSON (v1.1):     ``english--<hash>-<i>-<j>``      (double dash)
# The legacy double-dash pattern was hard-coded here, so every parquet row
# fell through to "unknown" — and the per-language 5-shot demo lookup found
# nothing, dropping every example silently to 0-shot. Accept either form.
_LANG_PAT = re.compile(r"^([a-z]+)-")


# Translation table that DELETES all ASCII punctuation (mapping to None
# rather than a space). SQuAD 1.1's canonical `remove_punc` concatenates
# without separators, so "U.S.A." → "USA" — the same convention TyDiQA-GoldP
# is evaluated under. Replacing with a space instead would split tokens at
# punctuation boundaries and turn "U.S.A." into "u s a", which mismatches
# "USA" and diverges from the standard.
# string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ — ASCII-only on
# purpose, so language-specific punctuation in Arabic / Telugu / Bengali
# (which carries semantic meaning) is preserved.
_ASCII_PUNCT_STRIP = str.maketrans({c: None for c in string.punctuation})
# English-only article pattern. TyDiQA's other 8 languages don't have direct
# equivalents, and bare "a"/"an"/"the" as standalone foreign-language tokens
# is rare enough that this is effectively a no-op on non-English rows.
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize(s: str) -> str:
    """SQuAD 1.1-style answer normalisation (TyDiQA-GoldP convention).

    Order matches the canonical Rajpurkar et al. script:
      lowercase → strip ASCII punctuation → remove English articles →
      collapse whitespace → NFC.

    NFC matters for the non-Latin TyDiQA languages (Bengali, Arabic, Korean,
    Telugu, …) where the same visual answer can be encoded pre-composed
    (NFC) or decomposed (NFD). Without NFC, gold and prediction can be
    byte-different but visually identical, producing false-negative EM
    matches.

    Punctuation + article stripping match the de-facto extractive-QA
    metric. Previously omitted, which made "1920s" vs "1920s." (and the
    English "the X" / "X" pair) fail EM despite being the same answer —
    not just rare edge cases: TyDiQA's gold spans frequently embed
    parens and punctuation that the generation model wouldn't echo
    verbatim. Lifts EM by 5-15pt depending on the model.
    """
    s = s.lower()
    s = s.translate(_ASCII_PUNCT_STRIP)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return unicodedata.normalize("NFC", s)


def _exact_match(pred: str, gold_list: List[str]) -> bool:
    pn = _normalize(pred)
    return any(pn == _normalize(g) for g in gold_list)


def _f1(pred: str, gold_list: List[str]) -> float:
    """SQuAD token-overlap F1 — max over all gold references.

    This is the metric NAIT (ICLR 2026) Table 2 reports for TyDiQA-GoldP
    (the 39.48 baseline number), open-instruct uses via
    `evaluate.load("squad")`, and the original TyDiQA paper (Clark et al.
    2020) specifies as the primary GoldP metric.

    Token split is whitespace after `_normalize`. For CJK/Thai with no
    spaces this is coarser than ideal, but it matches the canonical SQuAD
    F1 implementation so cross-paper comparison stays valid.
    """
    pred_tokens = _normalize(pred).split()
    best = 0.0
    for g in gold_list:
        if not g:
            continue
        gold_tokens = _normalize(g).split()
        if not pred_tokens or not gold_tokens:
            score = 1.0 if pred_tokens == gold_tokens else 0.0
        else:
            common: Dict[str, int] = {}
            for t in pred_tokens:
                common[t] = common.get(t, 0) + 1
            num_same = 0
            for t in gold_tokens:
                if common.get(t, 0) > 0:
                    common[t] -= 1
                    num_same += 1
            if num_same == 0:
                score = 0.0
            else:
                precision = num_same / len(pred_tokens)
                recall = num_same / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        if score > best:
            best = score
    return best


def _language_of(qa_id: Optional[str]) -> str:
    """Extract the language prefix from a TyDiQA QA id."""
    if not qa_id:
        return "unknown"
    m = _LANG_PAT.match(qa_id)
    return m.group(1) if m else "unknown"


def _normalize_answers_field(ans: Any) -> List[str]:
    """Return the list of gold answer strings from any of the layouts the
    TyDiQA / SQuAD ecosystem uses for the ``answers`` field.

    Handles:
      - dict ``{"text": [...], "answer_start": [...]}`` (SQuAD / HF parquet)
      - dict ``{"text": "single"}`` (some converters emit scalar)
      - list of dicts ``[{"text": "...", "answer_start": N}, ...]`` (SQuAD 2.0)
      - list of strings ``["a", "b"]`` (flat array converters)
      - None / empty → []
    """
    if isinstance(ans, dict):
        t = ans.get("text", [])
        if isinstance(t, str):
            return [t] if t else []
        if t is None:
            return []
        return [str(x) for x in t]
    if isinstance(ans, list):
        if not ans:
            return []
        if isinstance(ans[0], str):
            return [str(x) for x in ans]
        return [str(a.get("text", "")) for a in ans if isinstance(a, dict)]
    return []


def _parse_squad_file(path: str) -> List[Dict[str, Any]]:
    """Parse a TyDiQA JSON dev/train file.

    Accepts three layouts, auto-detected from the top-level shape:

    1. **Canonical SQuAD nested** — ``{"data": [{"paragraphs":
       [{"context": "...", "qas": [{"id": ..., "question": ...,
       "answers": ...}]}]}]}``. This is the v1.1 GCS layout.

    2. **Flat JSON array** — ``[{"id": ..., "question": ..., "context":
       ..., "answers": ...}, ...]``. Some converters (and a few re-hosts)
       publish TyDiQA-GoldP this way. The previous parser called
       ``raw.get("data", [])`` here and crashed with AttributeError
       ("list has no attribute 'get'"); add explicit branch.

    3. **JSONL** — one JSON object per line, same per-record schema as (2).
       Detected by JSONDecodeError on the whole-file parse + a successful
       line-by-line re-parse.
    """
    with open(path) as f:
        text = f.read()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL fallback before giving up.
        records: List[Dict[str, Any]] = []
        for ln, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"TyDiQA file {path!r} is neither valid JSON nor JSONL "
                    f"(line {ln} failed: {exc}).",
                ) from exc
        raw = records  # falls through to the flat-array branch below

    examples: List[Dict[str, Any]] = []

    if isinstance(raw, dict) and "data" in raw:
        # (1) Canonical SQuAD nested layout.
        for article in raw.get("data", []) or []:
            for para in article.get("paragraphs", []) or []:
                for qa in para.get("qas", []) or []:
                    examples.append({
                        "id": qa.get("id"),
                        "question": qa.get("question"),
                        "context": para.get("context", ""),
                        "answers": {"text": _normalize_answers_field(qa.get("answers"))},
                        "language": _language_of(qa.get("id")),
                    })
        return examples

    if isinstance(raw, list):
        # (2) / (3) Flat array of per-QA records. Each record carries its own
        # context (no paragraphs wrapper). Required keys are graceful — we
        # default missing context/question to empty string rather than
        # crashing so partial data still produces a metric (and per-record
        # failures show up as wrong predictions, not a parse abort).
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            qa_id = rec.get("id") or rec.get("qid") or rec.get("example_id")
            examples.append({
                "id": qa_id,
                "question": str(rec.get("question", "")),
                "context": str(rec.get("context") or rec.get("passage") or ""),
                "answers": {"text": _normalize_answers_field(rec.get("answers"))},
                "language": _language_of(qa_id),
            })
        return examples

    raise ValueError(
        f"TyDiQA file {path!r} has unrecognised top-level type "
        f"{type(raw).__name__!r}. Expected:\n"
        f"  - dict with `data` key (SQuAD v1.1 nested), or\n"
        f"  - list of QA records (flat JSON array), or\n"
        f"  - JSONL (one record per line)."
    )


def _parse_parquet_file(path: str) -> List[Dict[str, Any]]:
    """Parse the HuggingFace `google-research-datasets/tydiqa` Gold Passage
    parquet shard. Schema: ``id, title, context, question, answers={text,
    answer_start}`` — same field meanings as the legacy SQuAD JSON.

    The legacy v1.1 GCS JSON URLs now return HTTP 403 (the bucket revoked
    anonymous read), so the parquet on HF is the de-facto source of truth.
    """
    # pandas / pyarrow are not in the project's core deps because the
    # trainer side doesn't need them. We import lazily so installing them
    # only matters when running TyDiQA eval against parquet.
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Reading TyDiQA parquet requires pandas + pyarrow. Install with:\n"
            "  pip install pandas pyarrow\n"
            "Or convert to JSON once via scripts/download_tydiqa.sh and the\n"
            "evaluator will fall back to the SQuAD JSON path."
        ) from exc
    import pandas as pd
    df = pd.read_parquet(path)
    examples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ans = row.get("answers")
        # parquet answers is a dict with numpy arrays — normalise to list[str].
        # Don't use `texts_raw or []` here: numpy arrays raise
        # "truth value ambiguous" on bool conversion. Iterate directly and
        # treat the None case explicitly.
        if isinstance(ans, dict):
            texts_raw = ans.get("text")
            if texts_raw is None:
                texts = []
            else:
                try:
                    texts = [str(t) for t in texts_raw]
                except TypeError:
                    texts = []
        else:
            texts = []
        qa_id = row.get("id")
        examples.append({
            "id": qa_id,
            "question": str(row.get("question", "")),
            "context": str(row.get("context", "")),
            "answers": {"text": texts},
            "language": _language_of(qa_id),
        })
    return examples


def _parse_tydiqa_file(path: str) -> List[Dict[str, Any]]:
    """Dispatch to the parquet or JSON / JSONL parser based on extension.

    ``.parquet`` → HF parquet path. Anything else (``.json``, ``.jsonl``,
    extension-less) → `_parse_squad_file` which auto-detects the SQuAD
    nested vs flat-array vs JSONL layout.
    """
    if path.endswith(".parquet"):
        return _parse_parquet_file(path)
    return _parse_squad_file(path)


_DEV_KEYWORDS = ("validation", "valid", "dev")
_TRAIN_KEYWORDS = ("train",)


def _classify_split(basename: str) -> Optional[str]:
    """Return ``"dev"`` / ``"train"`` based on filename keyword, else None.

    Order matters: check ``train`` BEFORE ``valid`` because some hashed
    HF shard names contain "train" without a leading separator.
    Case-insensitive.
    """
    name = basename.lower()
    # Strip the extension(s) so we only match keywords in the stem
    # (e.g. "validation-00000-of-00001.parquet" stem == "validation-00000-of-00001").
    for ext in (".parquet", ".jsonl", ".json"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    for kw in _TRAIN_KEYWORDS:
        if kw in name:
            return "train"
    for kw in _DEV_KEYWORDS:
        if kw in name:
            return "dev"
    return None


def _glob_split_files(data_dir: str, ext: str) -> Tuple[Optional[str], Optional[str]]:
    """Walk ``data_dir`` looking for files matching ``*<ext>`` and classify
    each by filename keyword. Returns ``(dev_path, train_path)``.

    Searches the top level AND one sub-directory deep (HF datasets cache
    layouts sometimes nest as ``<repo>/data/*.parquet``). The first match
    of each kind wins.
    """
    import glob as _glob
    dev_path: Optional[str] = None
    train_path: Optional[str] = None

    patterns = [
        os.path.join(data_dir, f"*{ext}"),
        os.path.join(data_dir, "*", f"*{ext}"),
        os.path.join(data_dir, "*", "*", f"*{ext}"),
    ]
    for pat in patterns:
        for p in sorted(_glob.glob(pat)):
            kind = _classify_split(os.path.basename(p))
            if kind == "dev" and dev_path is None:
                dev_path = p
            elif kind == "train" and train_path is None:
                train_path = p
        if dev_path is not None and train_path is not None:
            break
    return dev_path, train_path


def _resolve_split_paths(data_dir: str) -> Tuple[str, str, str]:
    """Return (dev_path, train_path, base_dir).

    Accepts either:
      - a direct .json / .jsonl / .parquet path (used as dev_path; train
        mate is looked up in the same directory using the matching
        extension and keyword), or
      - a directory containing TyDiQA files in any of the following
        common layouts (preferred order: parquet > legacy JSON):
            * canonical HF parquet — ``validation-00000-of-00001.parquet`` /
              ``train-00000-of-00001.parquet``,
            * any parquet whose filename contains ``validation``/``valid``/``dev``
              vs ``train`` keyword (e.g. ``dev.parquet`` + ``train.parquet``,
              or ``tydiqa-dev-<hash>.parquet``),
            * legacy v1.1 SQuAD JSON — ``tydiqa-goldp-v1.1-dev.json`` /
              ``tydiqa-goldp-v1.1-train.json``,
            * fall-back glob: any JSON / JSONL file matching the dev/train
              keyword.

    The previous resolver required EXACT canonical filenames and silently
    fell through to the legacy JSON branch when parquet was present but
    named differently (`validation.parquet`, `dev.parquet`, …) — and then
    crashed on a missing JSON. This broader matcher picks parquet first
    whenever any parquet with a recognisable keyword is on disk.
    """
    if data_dir.endswith((".json", ".jsonl", ".parquet")):
        dev_path = data_dir
        base_dir = os.path.dirname(data_dir) or "."
        if data_dir.endswith(".parquet"):
            # Try canonical name first, then keyword-glob within the same dir.
            train_canon = os.path.join(base_dir, "train-00000-of-00001.parquet")
            if os.path.exists(train_canon):
                train_path = train_canon
            else:
                _, train_path = _glob_split_files(base_dir, ".parquet")
                if train_path is None:
                    train_path = train_canon  # leave the missing-file diag to load time
        elif data_dir.endswith(".jsonl"):
            stem = os.path.basename(data_dir).replace("dev", "train", 1)
            train_path = os.path.join(base_dir, stem)
        else:
            train_path = os.path.join(base_dir, "tydiqa-goldp-v1.1-train.json")
        return dev_path, train_path, base_dir

    # Directory mode. Prefer parquet, then legacy canonical JSON, then
    # broad keyword glob.
    canonical_parquet = (
        os.path.join(data_dir, "validation-00000-of-00001.parquet"),
        os.path.join(data_dir, "train-00000-of-00001.parquet"),
    )
    if os.path.exists(canonical_parquet[0]):
        # Canonical names — keep train-canonical even if missing; load
        # time will surface a clear "no demos → 0-shot" path.
        return canonical_parquet[0], canonical_parquet[1], data_dir

    # Try parquet glob (any filename containing dev/train keywords).
    p_dev, p_train = _glob_split_files(data_dir, ".parquet")
    if p_dev is not None:
        logger.info(
            "TyDiQA: matched parquet by keyword (dev=%s, train=%s) — "
            "exact canonical filenames not found, but a recognisable "
            "parquet pair was present.",
            os.path.basename(p_dev),
            os.path.basename(p_train) if p_train else "(none — will run 0-shot)",
        )
        return p_dev, (p_train or canonical_parquet[1]), data_dir

    # Legacy canonical JSON.
    legacy_json = (
        os.path.join(data_dir, "tydiqa-goldp-v1.1-dev.json"),
        os.path.join(data_dir, "tydiqa-goldp-v1.1-train.json"),
    )
    if os.path.exists(legacy_json[0]):
        return legacy_json[0], legacy_json[1], data_dir

    # Broad JSON / JSONL glob as last resort.
    for ext in (".jsonl", ".json"):
        j_dev, j_train = _glob_split_files(data_dir, ext)
        if j_dev is not None:
            logger.warning(
                "TyDiQA: matched %s by keyword (dev=%s, train=%s) — "
                "neither canonical parquet nor legacy SQuAD JSON found. "
                "If a parquet file exists, please verify its filename "
                "contains 'validation'/'valid'/'dev' or 'train' and re-run.",
                ext, os.path.basename(j_dev),
                os.path.basename(j_train) if j_train else "(none — will run 0-shot)",
            )
            return j_dev, (j_train or legacy_json[1]), data_dir

    tried = "\n  ".join((
        canonical_parquet[0],
        f"{data_dir}/*.parquet  (keyword glob: validation/valid/dev/train)",
        legacy_json[0],
        f"{data_dir}/*.json  (keyword glob)",
        f"{data_dir}/*.jsonl  (keyword glob)",
    ))
    # Surface enough state to pinpoint the failure cause: missing dir,
    # missing files, wrong env var, wrong permissions. The previous
    # message only showed candidate paths, which left the user guessing
    # whether the dir even existed.
    diag_lines = [
        f"  TYDIQA_DATA_DIR env  : {os.environ.get('TYDIQA_DATA_DIR', '<unset>')!r}",
        f"  data_dir arg        : {data_dir!r}",
        f"  data_dir is_dir     : {os.path.isdir(data_dir)}",
    ]
    if os.path.isdir(data_dir):
        try:
            contents = sorted(os.listdir(data_dir))
            shown = contents[:10] + (["…"] if len(contents) > 10 else [])
            diag_lines.append(f"  data_dir contents   : {shown}")
        except PermissionError as _e:
            diag_lines.append(f"  data_dir contents   : <PermissionError: {_e}>")
    else:
        parent = os.path.dirname(data_dir.rstrip("/")) or "."
        diag_lines.append(f"  parent dir          : {parent!r} (exists={os.path.isdir(parent)})")
    diagnostics = "\n".join(diag_lines)
    raise FileNotFoundError(
        f"TyDiQA dev split not found under {data_dir!r}.\n\n"
        f"Tried these candidate paths:\n  {tried}\n\n"
        f"Diagnostics:\n{diagnostics}\n\n"
        f"Fix:\n"
        f"  bash scripts/download_tydiqa.sh {data_dir}\n\n"
        f"That fetches both splits from "
        f"huggingface.co/datasets/google-research-datasets/tydiqa "
        f"(the legacy storage.googleapis.com URLs now return HTTP 403).\n"
        f"If `download_tydiqa.sh` itself failed, run it manually and "
        f"check that the cluster has outbound HTTPS to huggingface.co "
        f"and that you have write permission on {data_dir!r}."
    )


def _load_demos_by_language(
    train_path: str,
    n_fewshot: int,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return ``{language: [(context, question, answer), ...]}`` (first
    ``n_fewshot`` train examples per language with non-empty gold)."""
    if not os.path.exists(train_path):
        return {}
    train = _parse_tydiqa_file(train_path)
    by_lang: Dict[str, List[Tuple[str, str, str]]] = {}
    for ex in train:
        gold = (ex["answers"].get("text") or [""])[0]
        if not gold.strip():
            continue
        bucket = by_lang.setdefault(ex["language"], [])
        if len(bucket) < n_fewshot:
            bucket.append((ex["context"], ex["question"], gold))
        # All buckets full → done.
        if all(len(v) >= n_fewshot for v in by_lang.values()) and len(by_lang) >= 9:
            # 9 = number of TyDiQA gold-passage languages; loose check.
            pass  # don't break — we may not have seen all languages yet.
    return by_lang


@register("tydiqa")
class TyDiQAEvaluator(BenchmarkEvaluator):

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",
        data_dir: Optional[str] = None,
        max_new_tokens: int = 100,
        n_fewshot: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "TyDiQA: `data_dir` is required (path to a directory "
                "containing tydiqa-goldp-v1.1-dev.json).",
            )

        # Resolve dev/train paths, handling both the HF parquet layout
        # (current) and the legacy v1.1 JSON layout. Raises a clear
        # FileNotFoundError listing every checked location on miss.
        dev_file, train_file, base_dir = _resolve_split_paths(data_dir)

        examples = _parse_tydiqa_file(dev_file)

        # Content audit BEFORE running any model forward. A file can
        # successfully parse as JSON/parquet but contain the wrong dataset
        # (e.g. SQuAD v1.1 itself, MS MARCO, or a flat array of OPP problems).
        # NAIT-faithful TyDiQA GoldP has these characteristics:
        #   - non-empty list of records,
        #   - each record carries a non-empty `question`,
        #   - majority have at least one gold answer,
        #   - record `id` starts with one of the 9 TyDiQA-GoldP language
        #     prefixes (arabic / bengali / english / finnish / indonesian /
        #     korean / russian / swahili / telugu).
        # If any of these fail markedly, abort with a clear message.
        if not examples:
            raise ValueError(
                f"TyDiQA file {dev_file!r} parsed but produced 0 records — "
                f"either empty file or unrecognised schema. Refusing to run "
                f"a benchmark on an empty dataset.",
            )
        _has_q = sum(1 for ex in examples if ex.get("question"))
        _has_a = sum(1 for ex in examples if ex.get("answers", {}).get("text"))
        _expected_langs = {
            "arabic", "bengali", "english", "finnish", "indonesian",
            "korean", "russian", "swahili", "telugu",
        }
        _seen_langs = {ex.get("language", "unknown") for ex in examples}
        _known = _seen_langs & _expected_langs
        # Hard aborts only for "this is clearly empty/wrong data" cases.
        # Soft warnings for "paper-faithfulness deviates" cases — we'd rather
        # produce a partial score than 0 because of an overly strict audit.
        if _has_q < 0.50 * len(examples):
            raise ValueError(
                f"TyDiQA file {dev_file!r}: only {_has_q}/{len(examples)} "
                f"records have a non-empty `question` field — the dataset "
                f"doesn't look like a QA dataset at all. Refusing to run.",
            )
        elif _has_q < 0.95 * len(examples):
            logger.warning(
                "TyDiQA: only %d/%d records have a question field. Score "
                "will average over the populated subset only.",
                _has_q, len(examples),
            )
        if _has_a < 0.10 * len(examples):
            raise ValueError(
                f"TyDiQA file {dev_file!r}: only {_has_a}/{len(examples)} "
                f"records have any gold answer text — without gold answers "
                f"EM can't be measured. Refusing to run.",
            )
        elif _has_a < 0.50 * len(examples):
            logger.warning(
                "TyDiQA: only %d/%d records have gold answers (TyDiQA GoldP "
                "validation usually has answers on ~all rows). Score will "
                "skip answerless rows.",
                _has_a, len(examples),
            )
        if not _known:
            # Soft warning instead of abort — non-canonical id formats (e.g.
            # numeric ids from a custom dump) are common and shouldn't prevent
            # measurement. NAIT comparability suffers but a score is better
            # than a 0 from an aborted run.
            logger.warning(
                "TyDiQA: no record id starts with a known TyDiQA-GoldP language "
                "prefix (expected one of %s). Got prefixes: %s. Same-language "
                "5-shot demo lookup will all miss → 0-shot fallback per sample.",
                sorted(_expected_langs), sorted(_seen_langs)[:10],
            )
        elif len(_known) < 5:
            logger.warning(
                "TyDiQA: only %d/9 known languages present in dev (%s) — "
                "score will be over a partial language set and won't be "
                "directly comparable to NAIT Table 2 (which reports the "
                "full 9-language average).",
                len(_known), sorted(_known),
            )
        logger.info(
            "TyDiQA schema OK: %d records | %d with question | %d with answer "
            "| known langs: %s",
            len(examples), _has_q, _has_a, sorted(_known),
        )

        if limit is not None:
            examples = examples[:limit]

        # Truncation safety: TyDiQA prompt ends with the test "Context:\n...\n
        # Question:\n...\nAnswer:" block. With the HF tokenizer's default
        # `truncation_side='right'`, an over-long 5-shot prompt would have
        # the TEST context+question+Answer: prefix truncated, silently
        # destroying the prediction. Left-truncation drops a same-language
        # demo from the FRONT instead.
        tokenizer.truncation_side = "left"

        # Load same-language demonstrations from train.json.
        demos_by_lang = _load_demos_by_language(train_file, n_fewshot) if n_fewshot > 0 else {}
        fewshot_fallback_reason: Optional[str] = None
        if n_fewshot > 0 and not demos_by_lang:
            fewshot_fallback_reason = (
                f"train.json not found at {train_file} — running 0-shot. "
                f"NAIT paper Table 2 reports 5-shot; 0-shot typically scores "
                f"10-15pt lower EM, so this run is NOT directly comparable. "
                f"Download from "
                f"https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json "
                f"to enable 5-shot."
            )
            logger.error("TyDiQA FALLBACK: %s", fewshot_fallback_reason)
            effective_fewshot = 0
        else:
            effective_fewshot = n_fewshot
            logger.info(
                "TyDiQA: %d examples | limit=%s | n_fewshot=%d | langs_with_demos=%s",
                len(examples), limit, effective_fewshot,
                {lang: len(d) for lang, d in demos_by_lang.items()},
            )

        # Use the caller's prompt_style as-is. The earlier code force-mapped
        # alpaca_default → llama_user_assistant on the assumption that NAIT's
        # paper layout was always `<|user|>/<|assistant|>`. That mapping
        # actively HURT accuracy on models SFT'd with the Alpaca template
        # (the new LLAMA-TUNE default for llama2 — see configs/models/
        # llama2-7b.yaml): those models never saw `<|user|>` as a
        # response prefix, so their answers came out as fragmentary / wrong-
        # format text, tanking EM by ~5-15pt. The `tydiqa_generation_prefix`
        # function in tads/data/sft_prompts.py already has an
        # `alpaca_default` branch (Alpaca SFT template) that matches what
        # the model saw at train time — let it through unchanged.
        prefix_style = prompt_style

        correct = 0
        f1_sum = 0.0
        results = []
        # Count samples that fell back to 0-shot because their language
        # couldn't be parsed (qa_id didn't match `lang--...`) or no demos
        # were available for that language. Visible in the summary so a
        # custom dev.json doesn't silently downgrade to 0-shot.
        n_silent_zero_shot = 0
        for i, ex in enumerate(examples):
            gold = ex["answers"].get("text") or ["No answer"]
            ex_lang = ex.get("language", "unknown")
            demos = demos_by_lang.get(ex_lang) if effective_fewshot else None
            if effective_fewshot and not demos:
                n_silent_zero_shot += 1
            prompt = tydiqa_generation_prefix(
                ex.get("context", ""), ex["question"],
                prompt_style=prefix_style,
                demos=demos,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            # Slice on input_ids length, NOT on the decoded prompt string.
            # tokenizer(...) auto-prepends BOS (e.g. <s> for Llama / Mistral)
            # AND decode(skip_special_tokens=True) removes it again; the same
            # round-trip also normalises whitespace and renders some special
            # tokens differently. The result is that
            # decode(out[0], skip_special_tokens=True)[:len(prompt)] is NOT
            # the original prompt — slicing by prompt length lopped off
            # characters from the generated answer too, which silently
            # turned every EM into a near-empty / leading-fragment match and
            # produced the all-zero EM the user reported.
            prompt_tok_len = inputs["input_ids"].shape[1]
            gen_ids = out[0, prompt_tok_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # TyDiQA gold answers are short extractive spans. Take just the
            # first line so trailing prose / next "Question:" demonstrations
            # the model continued into don't poison the EM compare.
            if pred:
                pred = pred.splitlines()[0].strip()

            ok = _exact_match(pred, gold)
            f1 = _f1(pred, gold)
            correct += int(ok)
            f1_sum += f1
            results.append({
                "question": ex["question"],
                "language": ex.get("language", "unknown"),
                "prediction": pred,
                "correct": ok,
                "f1": round(f1, 4),
            })

            # Release per-example CUDA tensors so the next 5-shot prompt
            # (~1.5k input + up to max_new_tokens generated) doesn't peak
            # at 2× KV cache. Mirrors bbh.py / gsm8k.py / humaneval.py.
            # ~5500 examples × ~10ms = ~55s overhead vs the OOM /
            # fragmentation hang risk under concurrent multi-GPU eval.
            # gen_ids is a slice-view of out, so it keeps the underlying
            # storage alive even after `del out` — must drop it too for
            # empty_cache() to actually reclaim the KV block.
            del inputs, out, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 100 == 0:
                logger.info(
                    "  Progress: %d/%d | EM: %.4f | F1: %.4f",
                    i + 1, len(examples),
                    correct / (i + 1), f1_sum / (i + 1),
                )

        em_score = correct / len(examples) if examples else 0.0
        f1_score = f1_sum / len(examples) if examples else 0.0
        # NAIT / Clark et al. / open-instruct all report F1 as the headline
        # TyDiQA-GoldP metric; EM is secondary. Expose F1 as `accuracy` so
        # the cross-bench score-board reader picks up the paper-canonical
        # number without bench-specific branching.
        accuracy = f1_score
        # Per-language EM + F1 breakdown (paper-style).
        per_lang: Dict[str, Dict[str, Any]] = {}
        for r in results:
            bucket = per_lang.setdefault(
                r["language"],
                {"correct": 0, "f1_sum": 0.0, "total": 0},
            )
            bucket["total"] += 1
            bucket["correct"] += int(r["correct"])
            bucket["f1_sum"] += float(r["f1"])
        per_lang_acc = {
            lang: {
                **b,
                "accuracy_em": b["correct"] / b["total"] if b["total"] else 0.0,
                "accuracy_f1": b["f1_sum"] / b["total"] if b["total"] else 0.0,
            }
            for lang, b in per_lang.items()
        }
        if effective_fewshot and n_silent_zero_shot > 0:
            logger.warning(
                "TyDiQA: %d / %d samples (%.1f%%) fell back to 0-shot because "
                "no same-language demos were available — their qa_id didn't "
                "match `lang--...` or the train.json lacked that language.",
                n_silent_zero_shot, len(examples),
                100.0 * n_silent_zero_shot / max(1, len(examples)),
            )
        summary = {
            # `accuracy` aliases F1 — paper canonical (NAIT Table 2 39.48,
            # Clark et al. 2020, open-instruct headline). EM stays as
            # secondary diagnostic under `accuracy_em`.
            "accuracy": accuracy,
            "accuracy_f1": f1_score,
            "accuracy_em": em_score,
            "f1": f1_score,
            "em": em_score,
            "correct": correct,
            "total": len(examples),
            "n_fewshot": effective_fewshot,
            "n_fewshot_requested": n_fewshot,
            # Per-sample 0-shot fallback count (language unparseable or
            # train.json had no demos for that language).
            "n_silent_zero_shot": n_silent_zero_shot,
            # If train.json was missing and we silently dropped to 0-shot,
            # surface that in the JSON itself — downstream paper-comparison
            # tooling can then refuse to treat this number as 5-shot.
            "fewshot_fallback": fewshot_fallback_reason,
            "paper_faithful": (
                fewshot_fallback_reason is None and n_silent_zero_shot == 0
            ),
            "benchmark": "tydiqa",
            "per_language": per_lang_acc,
            "per_question": results,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        if fewshot_fallback_reason is not None:
            logger.error(
                "TyDiQA F1: %.4f | EM: %.4f (%d/%d) | NOT paper-faithful (0-shot fallback)",
                f1_score, em_score, correct, len(examples),
            )
        else:
            logger.info(
                "TyDiQA F1: %.4f | EM: %.4f (%d/%d) | n_fewshot=%d",
                f1_score, em_score, correct, len(examples), effective_fewshot,
            )
        return summary
