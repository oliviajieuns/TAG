"""XQuAD evaluator — cross-lingual SQuAD-v1, 5-shot extractive QA.

Setup
-----
- **All 11 NAIT languages** (ar, de, el, en, es, hi, ru, th, tr, vi, zh)
  loaded from per-language JSON files (``xquad.<lang>.json``).
  Romanian (`ro`) is also included if present locally — `--languages`
  CLI override lets you drop it for paper-exact comparison.
- **F1 macro-average over languages** is the headline metric (the code
  aliases ``accuracy = macro_f1``); EM macro is also computed and kept
  in the JSON for diagnostics. NAIT Table 2 / Artetxe et al. 2020 both
  use macro-F1 for XQuAD so a tiny language can't be swamped by a
  larger one. Each XQuAD language ships 1,190 questions (the same
  English SQuAD test split, professionally translated), so per-language
  sizes are uniform; macro keeps us paper-consistent regardless.
- **5-shot demonstrations** are drawn from the SAME language's first
  N items (and the eval skips those N). Matches the lm-eval-harness
  cross-lingual SQuAD convention.
- **Single-line generation + exact-match grading**, identical to TyDiQA
  (normalisation: NFC + lower + punctuation strip).

Input file shape (one per language):
    {"data": [{"paragraphs": [{"context": str, "qas": [{"question": str,
                                                        "answers": [{"text": str, ...}, ...],
                                                        "id": str}, ...]}, ...]}, ...]}

This is the canonical SQuAD-v1 JSON, identical between XQuAD and the
SQuAD-v1 dev set. We flatten paragraph→qa rows up-front for simplicity.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


from ._gen import gen_batch_size, generate_texts
from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import tydiqa_generation_prefix

logger = logging.getLogger(__name__)


# NAIT Table 2 baseline list. Order is alphabetic for log readability.
NAIT_LANGUAGES = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]
# Romanian was added later by Google; include only if locally present and
# the user didn't opt out via --languages.
OPTIONAL_LANGUAGES = ["ro"]


# ---- text normalisation + EM (SQuAD canonical, NFC + lower + punct strip) ----
# Canonical SQuAD-v1 normalize_answer (Rajpurkar et al. 2016):
#   1. lowercase
#   2. strip punctuation
#   3. drop English articles (`a` / `an` / `the`) as whole words
#   4. collapse whitespace
# Steps 1, 2, 4 used to be implemented but step 3 was missing — costing
# ~3–5pt of English EM and matching paper-faithful SQuAD scoring.
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _is_punct(ch: str) -> bool:
    """Unicode-aware punctuation predicate. `string.punctuation` only covers
    ASCII (`!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~`); non-English XQuAD answers
    can end with `。` / `,` / `？` / `；` / `」` etc. which we DO want to
    strip before EM compare. Use `unicodedata.category` so every Unicode
    `P*` class (punctuation) is included regardless of script.
    """
    return unicodedata.category(ch).startswith("P")


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").strip().lower()
    # Replace punctuation with whitespace (not delete) so adjacent words
    # don't get concatenated — e.g. "John,Mary" → "john mary" not "johnmary".
    s = "".join(" " if _is_punct(ch) else ch for ch in s)
    # Drop English articles. Affects only English (and a few Romance demos),
    # but cheap to apply globally — non-Latin scripts don't contain the
    # word boundaries this regex matches.
    s = _ARTICLE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Common Alpaca-SFT / chat-style prose preambles a model emits before the
# short extractive span. Stripping these BEFORE EM compare turns
# "the answer is <X>" into "<X>" — recovering EM mass that would otherwise
# collapse to 0 against the gold span.
_PRED_PREAMBLE_RES = [
    re.compile(r"^\s*the\s+answer\s+(?:is|would\s+be)\s*[:,]?\s*", re.IGNORECASE),
    re.compile(r"^\s*answer\s*[:=]\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:it|that)\s+is\s*[:,]?\s*", re.IGNORECASE),
    re.compile(r"^\s*based\s+on\s+the\s+(?:passage|context|text)[^,]*,\s*", re.IGNORECASE),
    re.compile(r"^\s*according\s+to\s+the\s+(?:passage|context|text)[^,]*,\s*", re.IGNORECASE),
]


def _strip_pred_preamble(pred: str) -> str:
    """Drop leading prose like 'The answer is ' so EM compares against the
    short span the model actually meant to give. Idempotent."""
    for _ in range(3):  # cap iterations — guards against pathological loops
        new = pred
        for r in _PRED_PREAMBLE_RES:
            new = r.sub("", new)
        if new == pred:
            break
        pred = new
    return pred.strip()


def _exact_match(pred: str, golds: List[str]) -> bool:
    pn = _normalize(pred)
    return any(pn == _normalize(g) for g in golds if g)


def _f1(pred: str, golds: List[str]) -> float:
    """Token-overlap F1 (SQuAD canonical). Max over all gold references.
    Reported alongside EM as a diagnostic: when EM is low but F1 is high,
    the model is producing correct content with surface differences
    (different word order, partial match, extra article) — i.e. a
    normalisation issue rather than a model-knowledge issue.
    """
    pred_tokens = _normalize(pred).split()
    best = 0.0
    for g in golds:
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


def _load_xquad_file(path: str, language: str) -> List[Dict[str, Any]]:
    """Flatten one XQuAD JSON file into [(language, context, question, gold_list), ...]."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    out: List[Dict[str, Any]] = []
    for article in doc.get("data", []):
        for para in article.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                gold_texts = [
                    (a or {}).get("text", "") for a in answers if isinstance(a, dict)
                ]
                gold_texts = [g for g in gold_texts if g]
                out.append({
                    "language": language,
                    "context": ctx,
                    "question": qa.get("question", ""),
                    "id": qa.get("id", ""),
                    "answers": gold_texts or ["No answer"],
                })
    return out


def _build_demos(
    items: List[Dict[str, Any]], n: int,
) -> List[Tuple[str, str, str]]:
    """Pick first ``n`` items as (context, question, answer-text) tuples."""
    demos: List[Tuple[str, str, str]] = []
    for it in items[:n]:
        ans = (it["answers"] or [""])[0]
        demos.append((it["context"], it["question"], ans))
    return demos


@register("xquad")
class XQuADEvaluator(BenchmarkEvaluator):
    """XQuAD 11-language (plus optional ``ro``), 5-shot, F1 macro-avg
    over languages (EM macro is also reported as a diagnostic)."""

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
        # Audit A2: bumped 50 → 80. XQuAD answers in Thai/Chinese/Vietnamese
        # can exceed 50 subword tokens with the Llama tokenizer; the stop_strings
        # short-circuit short answers before 80 so wall-time impact is small
        # (~5% per lang) while long-span EM stops silently failing.
        max_new_tokens: int = 80,
        n_fewshot: int = 5,
        max_input_tokens: int = 2048,
        languages: Optional[List[str]] = None,
        # empty_cache every N items rather than every iteration. Per-item
        # KV cache for a ~50-token generation is tiny (~5–10 MB) so
        # fragmentation isn't a real risk between calls — sync + cudaFree
        # calls were adding ~5–15 ms × 13K items = up to ~3 min of pure
        # cleanup overhead. Bump to every 50 items so we still drain
        # occasionally but the per-step cost amortises away. Pure memory
        # management; zero impact on EM/F1 values.
        empty_cache_interval: int = 50,
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "XQuAD: `data_dir` is required (directory of xquad.<lang>.json "
                "files from google-deepmind/xquad).",
            )

        # Resolve language list: explicit override > NAIT-11 + locally-present
        # optional. Romanian is opt-in by being present; user can force
        # NAIT-exact comparison by passing languages=NAIT_LANGUAGES.
        if languages is None:
            wanted = list(NAIT_LANGUAGES)
            for opt_lang in OPTIONAL_LANGUAGES:
                if os.path.exists(os.path.join(data_dir, f"xquad.{opt_lang}.json")):
                    wanted.append(opt_lang)
        else:
            wanted = [str(l).strip() for l in languages if str(l).strip()]

        per_lang_items: Dict[str, List[Dict[str, Any]]] = {}
        missing: List[str] = []
        for lang in wanted:
            path = os.path.join(data_dir, f"xquad.{lang}.json")
            if not os.path.exists(path):
                missing.append(lang)
                continue
            per_lang_items[lang] = _load_xquad_file(path, lang)

        if not per_lang_items:
            raise FileNotFoundError(
                f"XQuAD: no xquad.<lang>.json found in {data_dir}. "
                f"Run `bash scripts/download_xquad.sh {data_dir}` first. "
                f"Wanted: {wanted}",
            )
        if missing:
            logger.warning(
                "XQuAD: %d languages missing from %s — skipping: %s",
                len(missing), data_dir, missing,
            )

        # Decide prefix_style. tydiqa_generation_prefix supports the same
        # 5-shot SQuAD-style template we want here, so reuse it directly.
        # For Mistral it auto-omits manual <s> (tokenizer adds BOS).
        prefix_style = prompt_style or "alpaca_default"

        # Left-truncation (same rationale as MMLU / GSM8K / TyDiQA).
        tokenizer.truncation_side = "left"

        per_lang_correct: Dict[str, int] = {}
        per_lang_total: Dict[str, int] = {}
        per_lang_f1_sum: Dict[str, float] = {}
        per_question: List[Dict[str, Any]] = []
        # Kept as a progress/telemetry counter. Allocator hygiene moved
        # into generate_texts(), which drains between batches;
        # `empty_cache_interval` is accepted for call-site compatibility and
        # no longer drives anything here.
        _seen_global = 0

        # Audit A5: pull demos from the English split for all non-English target
        # languages (NAIT/Artetxe cross-lingual convention: English exemplar →
        # target-language test). Previously each language used its OWN test
        # split's first n_fewshot items as demos, which (a) shrunk the eval set
        # to (1190 - n_fewshot)/lang and (b) deviated from the standard
        # cross-lingual eval setup. English itself still uses its own first
        # n_fewshot rows as demos.
        _en_items = per_lang_items.get("en")
        shared_en_demos = (
            _build_demos(_en_items, n_fewshot)
            if _en_items is not None and len(_en_items) > n_fewshot
            else None
        )
        if shared_en_demos is None:
            logger.warning(
                "XQuAD: xquad.en unavailable or too short — falling back to "
                "per-language demos from each test split (legacy behaviour, "
                "not paper-faithful for cross-lingual evaluation)."
            )

        for lang, items in per_lang_items.items():
            if shared_en_demos is not None and lang != "en":
                demos = shared_en_demos
                test_items = items  # full eval set; demos come from xquad.en
            else:
                # English itself, or fallback when xquad.en is missing.
                if len(items) <= n_fewshot:
                    logger.warning(
                        "XQuAD/%s: only %d items, skipping (need > n_fewshot=%d)",
                        lang, len(items), n_fewshot,
                    )
                    continue
                demos = _build_demos(items, n_fewshot)
                test_items = items[n_fewshot:]
            if limit is not None:
                test_items = test_items[:limit]

            # One generate() per example was the whole cost of this
            # benchmark (14k questions across 11 languages). Prompts are
            # built up front and decoded in batches; the prompt text, the
            # stop strings and every post-processing step below are
            # unchanged. See tag/evals/_gen.py.
            prompts = [
                tydiqa_generation_prefix(
                    ex["context"], ex["question"],
                    prompt_style=prefix_style, demos=demos,
                )
                for ex in test_items
            ]
            # Stop on the next demo's header — the model often hallucinates
            # a continuation like "\nContext: <next>" or "\nQuestion:".
            # The post-decode trim below stays as belt-and-braces for any
            # separator variant that slips through.
            raw_preds = generate_texts(
                model, tokenizer, prompts, device=device,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                stop_strings=[
                    "\nContext:", "\n\nContext:",
                    "\nQuestion:", "\n\nQuestion:",
                    "\nAnswer:", "\n\nAnswer:",
                ],
                progress_every=500, progress_label=f"XQuAD/{lang}",
            score_key=lambda r: _normalize(_strip_pred_preamble(
                r.strip().splitlines()[0] if r.strip() else "")),
            )

            n_correct = 0
            n_total = 0
            for ex, pred in zip(test_items, raw_preds):
                pred = pred.strip()
                # XQuAD gold is a short extractive span.
                for sep in ("\nContext:", "\nQuestion:", "\nAnswer:",
                            "\n\nContext:", "\n\nQuestion:", "\n\nAnswer:"):
                    idx = pred.find(sep)
                    if idx != -1:
                        pred = pred[:idx]
                        break
                if pred:
                    pred = pred.splitlines()[0].strip()
                # Drop leading "The answer is " / "Answer: " / "It is " /
                # "Based on the passage, " etc. prose preambles. Alpaca-SFT
                # models frequently emit these even when the prompt ends with
                # "Answer:" so the gold-side EM compare fails on a correctly-
                # known span hidden behind the preamble.
                pred = _strip_pred_preamble(pred)

                ok = _exact_match(pred, ex["answers"])
                f1 = _f1(pred, ex["answers"])
                n_correct += int(ok)
                n_total += 1
                per_lang_f1_sum[lang] = per_lang_f1_sum.get(lang, 0.0) + f1
                per_question.append({
                    "language": lang,
                    "id": ex["id"],
                    "question": ex["question"],
                    "prediction": pred,
                    "correct": ok,
                    "f1": round(f1, 4),
                })
                _seen_global += 1

            per_lang_correct[lang] = n_correct
            per_lang_total[lang] = n_total
            lang_em = n_correct / n_total if n_total else 0.0
            lang_f1 = (per_lang_f1_sum.get(lang, 0.0) / n_total) if n_total else 0.0
            logger.info(
                "  XQuAD/%s: EM=%.4f | F1=%.4f (%d/%d)",
                lang, lang_em, lang_f1, n_correct, n_total,
            )

        # NAIT-faithful macro: mean over per-language EM. Micro is also kept
        # for diagnostic.
        per_lang_em = {
            l: per_lang_correct[l] / per_lang_total[l] if per_lang_total[l] else 0.0
            for l in per_lang_total
        }
        per_lang_f1 = {
            l: per_lang_f1_sum.get(l, 0.0) / per_lang_total[l] if per_lang_total[l] else 0.0
            for l in per_lang_total
        }
        macro_em = (
            sum(per_lang_em.values()) / len(per_lang_em) if per_lang_em else 0.0
        )
        macro_f1 = (
            sum(per_lang_f1.values()) / len(per_lang_f1) if per_lang_f1 else 0.0
        )
        tot_correct = sum(per_lang_correct.values())
        tot_total = sum(per_lang_total.values())
        micro_em = tot_correct / tot_total if tot_total else 0.0

        summary = {
            # `accuracy` aliases macro F1 — paper canonical headline metric
            # for XQuAD (Artetxe et al. 2020; NAIT Table 2). EM stays as
            # a secondary diagnostic. Keeping the `accuracy` key name so
            # the score-board reader picks the right bench-level number
            # without bench-specific branching.
            "accuracy": macro_f1,
            "accuracy_f1": macro_f1,
            "accuracy_em": macro_em,
            "macro_em": macro_em,
            "macro_f1": macro_f1,
            "micro_em": micro_em,
            "per_language_f1": per_lang_f1,
            "total_correct": tot_correct,
            "total_questions": tot_total,
            "languages": sorted(per_lang_total),
            "missing_languages": missing,
            "n_fewshot": n_fewshot,
            "per_language": per_lang_em,
            "per_question": per_question,
            "benchmark": "xquad",
            # The batch size prompts were decoded at. Greedy decoding is
            # padding-invariant in exact arithmetic but not in float, so a
            # long chain-of-thought can fork on a near-tie; recording the
            # batch size is what makes this number reproducible.
            "generation_batch_size": gen_batch_size(),
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "XQuAD macro EM: %.4f | macro F1: %.4f | micro EM: %.4f | "
            "langs=%d | total=%d",
            macro_em, macro_f1, micro_em, len(per_lang_total), tot_total,
        )
        return summary
