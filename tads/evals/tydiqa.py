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
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import tydiqa_generation_prefix

logger = logging.getLogger(__name__)


_LANG_PAT = re.compile(r"^([a-z]+)--")


def _normalize(s: str) -> str:
    """Lowercase + whitespace collapse + Unicode NFC normalization.

    NFC matters for the non-Latin TyDiQA languages (Bengali, Arabic, Korean,
    Telugu, …) where the same visual answer can be encoded in either
    pre-composed (NFC) or decomposed (NFD) form. Without normalising, gold
    and prediction can be byte-different but visually identical, producing
    false-negative EM matches.
    """
    s = unicodedata.normalize("NFC", s.strip().lower())
    return re.sub(r"\s+", " ", s)


def _exact_match(pred: str, gold_list: List[str]) -> bool:
    pn = _normalize(pred)
    return any(pn == _normalize(g) for g in gold_list)


def _language_of(qa_id: Optional[str]) -> str:
    """TyDiQA gold-passage QA ids look like ``english--<hash>-<i>-<j>``."""
    if not qa_id:
        return "unknown"
    m = _LANG_PAT.match(qa_id)
    return m.group(1) if m else "unknown"


def _parse_squad_file(path: str) -> List[Dict[str, Any]]:
    """Parse a SQuAD-style TyDiQA file (train or dev share the schema)."""
    with open(path) as f:
        raw = json.load(f)
    examples = []
    for article in raw.get("data", []):
        for para in article.get("paragraphs", []):
            for qa in para.get("qas", []):
                ans = qa.get("answers", {})
                if isinstance(ans, dict):
                    texts = ans.get("text", []) or []
                elif isinstance(ans, list):
                    if ans and isinstance(ans[0], str):
                        texts = ans
                    else:
                        texts = [a.get("text") for a in ans if isinstance(a, dict)]
                else:
                    texts = []
                examples.append({
                    "id": qa.get("id"),
                    "question": qa.get("question"),
                    "context": para.get("context", ""),
                    "answers": {"text": texts},
                    "language": _language_of(qa.get("id")),
                })
    return examples


def _load_demos_by_language(
    train_path: str,
    n_fewshot: int,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return ``{language: [(context, question, answer), ...]}`` (first
    ``n_fewshot`` train examples per language with non-empty gold)."""
    if not os.path.exists(train_path):
        return {}
    train = _parse_squad_file(train_path)
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

        # Resolve dev/train paths. Accept either a directory or a direct .json path.
        if data_dir.endswith(".json"):
            dev_file = data_dir
            base_dir = os.path.dirname(data_dir) or "."
        else:
            dev_file = os.path.join(data_dir, "tydiqa-goldp-v1.1-dev.json")
            base_dir = data_dir
        train_file = os.path.join(base_dir, "tydiqa-goldp-v1.1-train.json")

        if not os.path.exists(dev_file):
            summary = {"accuracy_em": 0.0, "benchmark": "tydiqa", "error": "data not found"}
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.error("TyDiQA dev not found at %s", dev_file)
            return summary

        examples = _parse_squad_file(dev_file)
        if limit is not None:
            examples = examples[:limit]

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

        # Map alpaca_default to llama_user_assistant for the generation prefix
        # (paper convention; the prefix only affects the generation framing).
        prefix_style = (
            "llama_user_assistant" if prompt_style == "alpaca_default" else prompt_style
        )

        correct = 0
        results = []
        for i, ex in enumerate(examples):
            gold = ex["answers"].get("text") or ["No answer"]
            demos = demos_by_lang.get(ex.get("language", "unknown")) if effective_fewshot else None
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
            )
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = pred[len(prompt):].strip()

            ok = _exact_match(pred, gold)
            correct += int(ok)
            results.append({
                "question": ex["question"],
                "language": ex.get("language", "unknown"),
                "prediction": pred,
                "correct": ok,
            })
            if (i + 1) % 100 == 0:
                logger.info(
                    "  Progress: %d/%d | EM: %.4f",
                    i + 1, len(examples), correct / (i + 1),
                )

        accuracy = correct / len(examples) if examples else 0.0
        # Per-language accuracy breakdown (paper-style).
        per_lang: Dict[str, Dict[str, int]] = {}
        for r in results:
            bucket = per_lang.setdefault(r["language"], {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(r["correct"])
        per_lang_acc = {
            lang: {**b, "accuracy": b["correct"] / b["total"] if b["total"] else 0.0}
            for lang, b in per_lang.items()
        }
        summary = {
            "accuracy_em": accuracy,
            "correct": correct,
            "total": len(examples),
            "n_fewshot": effective_fewshot,
            "n_fewshot_requested": n_fewshot,
            # If train.json was missing and we silently dropped to 0-shot,
            # surface that in the JSON itself — downstream paper-comparison
            # tooling can then refuse to treat this number as 5-shot.
            "fewshot_fallback": fewshot_fallback_reason,
            "paper_faithful": fewshot_fallback_reason is None,
            "benchmark": "tydiqa",
            "per_language": per_lang_acc,
            "per_question": results,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        if fewshot_fallback_reason is not None:
            logger.error(
                "TyDiQA EM: %.4f (%d/%d) | NOT paper-faithful (0-shot fallback)",
                accuracy, correct, len(examples),
            )
        else:
            logger.info(
                "TyDiQA EM: %.4f (%d/%d) | n_fewshot=%d",
                accuracy, correct, len(examples), effective_fewshot,
            )
        return summary
