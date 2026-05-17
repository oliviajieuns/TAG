"""XQuAD evaluator — cross-lingual SQuAD-v1, 5-shot extractive QA.

Setup
-----
- **All 11 NAIT languages** (ar, de, el, en, es, hi, ru, th, tr, vi, zh)
  loaded from per-language JSON files (``xquad.<lang>.json``).
  Romanian (`ro`) is also included if present locally — `--languages`
  CLI override lets you drop it for paper-exact comparison.
- **EM macro-average over languages** — NAIT Table 2 reports XQuAD as a
  macro mean of per-language EM scores so a tiny language can't be
  swamped by a larger one. Each XQuAD language ships 1,190 questions
  (the same English SQuAD test split, professionally translated), so
  the per-language sizes are uniform anyway, but macro keeps us paper-
  consistent.
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

import torch

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import tydiqa_generation_prefix

logger = logging.getLogger(__name__)


# NAIT Table 2 baseline list. Order is alphabetic for log readability.
NAIT_LANGUAGES = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]
# Romanian was added later by Google; include only if locally present and
# the user didn't opt out via --languages.
OPTIONAL_LANGUAGES = ["ro"]


# ---- text normalisation + EM (SQuAD canonical, NFC + lower + punct strip) ----
_PUNCT = set(string.punctuation)


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").strip().lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _exact_match(pred: str, golds: List[str]) -> bool:
    pn = _normalize(pred)
    return any(pn == _normalize(g) for g in golds if g)


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
    """XQuAD 11-language, 5-shot, EM macro-avg over languages."""

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
        max_new_tokens: int = 50,
        n_fewshot: int = 5,
        max_input_tokens: int = 2048,
        languages: Optional[List[str]] = None,
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
        per_question: List[Dict[str, Any]] = []

        for lang, items in per_lang_items.items():
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

            n_correct = 0
            n_total = 0
            for ex in test_items:
                prompt = tydiqa_generation_prefix(
                    ex["context"], ex["question"],
                    prompt_style=prefix_style,
                    demos=demos,
                )
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_input_tokens,
                ).to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                prompt_tok_len = inputs["input_ids"].shape[1]
                gen_ids = out[0, prompt_tok_len:]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                # XQuAD gold is a short extractive span. Take only the first
                # line so trailing prose / next "Question:" hallucinations
                # don't poison EM.
                if pred:
                    pred = pred.splitlines()[0].strip()

                ok = _exact_match(pred, ex["answers"])
                n_correct += int(ok)
                n_total += 1
                per_question.append({
                    "language": lang,
                    "id": ex["id"],
                    "question": ex["question"],
                    "prediction": pred,
                    "correct": ok,
                })

                del inputs, out, gen_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_lang_correct[lang] = n_correct
            per_lang_total[lang] = n_total
            lang_em = n_correct / n_total if n_total else 0.0
            logger.info(
                "  XQuAD/%s: %.4f (%d/%d)", lang, lang_em, n_correct, n_total,
            )

        # NAIT-faithful macro: mean over per-language EM. Micro is also kept
        # for diagnostic.
        per_lang_em = {
            l: per_lang_correct[l] / per_lang_total[l] if per_lang_total[l] else 0.0
            for l in per_lang_total
        }
        macro_em = (
            sum(per_lang_em.values()) / len(per_lang_em) if per_lang_em else 0.0
        )
        tot_correct = sum(per_lang_correct.values())
        tot_total = sum(per_lang_total.values())
        micro_em = tot_correct / tot_total if tot_total else 0.0

        summary = {
            # `accuracy` aliases macro EM for score-board uniformity.
            "accuracy": macro_em,
            "accuracy_em": macro_em,           # alias 2 (TyDiQA-style key)
            "macro_em": macro_em,
            "micro_em": micro_em,
            "total_correct": tot_correct,
            "total_questions": tot_total,
            "languages": sorted(per_lang_total),
            "missing_languages": missing,
            "n_fewshot": n_fewshot,
            "per_language": per_lang_em,
            "per_question": per_question,
            "benchmark": "xquad",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "XQuAD macro EM: %.4f | micro EM: %.4f | langs=%d | total=%d",
            macro_em, micro_em, len(per_lang_total), tot_total,
        )
        return summary
