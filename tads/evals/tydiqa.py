"""TyDiQA evaluator (Exact-Match on English Gold-Passage dev split).

Note on few-shot count: this evaluator is currently 0-shot. The NAIT paper
(Appendix D) uses 5-shot gold-passage. To match paper Table 2 numbers,
prepend 5 demonstration QA pairs to each `tydiqa_generation_prefix` call.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import tydiqa_generation_prefix

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"\s+", " ", s)


def _exact_match(pred: str, gold_list: List[str]) -> bool:
    pn = _normalize(pred)
    return any(pn == _normalize(g) for g in gold_list)


def _parse_dev_file(dev_file: str):
    with open(dev_file) as f:
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
                })
    return examples


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
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "TyDiQA: `data_dir` is required (path to a directory "
                "containing tydiqa-goldp-v1.1-dev.json).",
            )

        dev_file = (
            data_dir
            if data_dir.endswith(".json")
            else os.path.join(data_dir, "tydiqa-goldp-v1.1-dev.json")
        )
        if not os.path.exists(dev_file):
            summary = {"accuracy_em": 0.0, "benchmark": "tydiqa", "error": "data not found"}
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.error("TyDiQA dev not found at %s", dev_file)
            return summary

        examples = _parse_dev_file(dev_file)
        if limit is not None:
            examples = examples[:limit]
        logger.info("TyDiQA: %d examples | limit=%s", len(examples), limit)

        # Map alpaca_default to llama_user_assistant for the generation prefix
        # (paper convention; the prefix only affects the generation framing).
        prefix_style = (
            "llama_user_assistant" if prompt_style == "alpaca_default" else prompt_style
        )

        correct = 0
        results = []
        for i, ex in enumerate(examples):
            gold = ex["answers"].get("text") or ["No answer"]
            prompt = tydiqa_generation_prefix(
                ex.get("context", ""), ex["question"], prompt_style=prefix_style,
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
                "prediction": pred,
                "correct": ok,
            })
            if (i + 1) % 100 == 0:
                logger.info(
                    "  Progress: %d/%d | EM: %.4f",
                    i + 1, len(examples), correct / (i + 1),
                )

        accuracy = correct / len(examples) if examples else 0.0
        summary = {
            "accuracy_em": accuracy,
            "correct": correct,
            "total": len(examples),
            "benchmark": "tydiqa",
            "per_question": results,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("TyDiQA EM: %.4f (%d/%d)", accuracy, correct, len(examples))
        return summary
