"""GSM8K evaluator (8-shot CoT, `The answer is X` extraction)."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import build_cot_prompt_prefix

logger = logging.getLogger(__name__)

ANSWER_PATTERN_HASH = re.compile(r"####\s*([^\n]+)", re.MULTILINE)
ANSWER_PATTERN_COT = re.compile(
    r"The answer is\s*([\-0-9.,$]+)\s*\.?", re.IGNORECASE | re.MULTILINE,
)


def _extract_predicted_number(text: str) -> str:
    # Self-correcting responses commonly say "The answer is 7. Wait, let me
    # recompute... The answer is 12." — .search() would have returned 7, the
    # wrong one. Use findall + last match so the FINAL declared answer wins,
    # matching how the canonical lm-eval-harness GSM8K extractor behaves.
    matches = ANSWER_PATTERN_COT.findall(text)
    if matches:
        return matches[-1].strip()
    matches = ANSWER_PATTERN_HASH.findall(text)
    if matches:
        return matches[-1].strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1].strip() if numbers else ""


def _normalize_answer(text: str) -> str:
    # Strip trailing punctuation too (the COT regex can capture a final "."
    # or "," when the answer ends a sentence, e.g. "The answer is 8."). The
    # downstream float comparison handles this, but string-equality on the
    # fast path was failing on otherwise-correct predictions.
    return (
        text.replace(",", "")
        .replace(" ", "")
        .replace("$", "")
        .replace("%", "")
        .rstrip(".")
        .strip()
    )


def _grade(response: str, ground_truth: str) -> bool:
    pred = _normalize_answer(_extract_predicted_number(response))
    gt = _normalize_answer(_extract_predicted_number(ground_truth))
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return False


@register("gsm8k")
class GSM8KEvaluator(BenchmarkEvaluator):
    """GSM8K with 8-shot Chain-of-Thought prompt (paper / harness aligned)."""

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
        max_new_tokens: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        # IMPORTANT: do NOT use `datasets.load_dataset` here. Without an
        # explicit cache_dir it writes to ~/.cache/huggingface/datasets,
        # which (a) lives on user-volume (small + private), (b) races
        # across concurrent evaluations, and (c) leaves stale arrow files
        # behind. pandas.read_parquet is a one-shot read with no cache.
        import pandas as pd

        if data_dir is None:
            raise ValueError("GSM8K: `data_dir` is required (path to gsm8k root).")

        # Locate test parquet.
        test_path = os.path.join(data_dir, "main", "test.parquet")
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, "main", "test-00000-of-00001.parquet")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"GSM8K test parquet not found under {data_dir}/main/. "
                f"Expected `test.parquet` or `test-00000-of-00001.parquet`."
            )
        df = pd.read_parquet(test_path)
        # GSM8K canonical schema. Validate up-front so we fail with a
        # specific message instead of a downstream KeyError on ex["question"].
        _missing = [c for c in ("question", "answer") if c not in df.columns]
        if _missing:
            raise ValueError(
                f"GSM8K parquet at {test_path} missing required column(s) "
                f"{_missing}. Got columns: {list(df.columns)}",
            )
        test_data = df.to_dict("records")
        if limit is not None:
            test_data = test_data[:limit]
        logger.info("GSM8K: %d examples | limit=%s", len(test_data), limit)

        correct = 0
        results = []
        for i, ex in enumerate(test_data):
            prompt = build_cot_prompt_prefix(ex["question"])
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
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            ok = _grade(response, ex["answer"])
            correct += int(ok)
            results.append({
                "question": ex["question"],
                "predicted": response,
                "correct": ok,
            })
            if (i + 1) % 100 == 0:
                logger.info(
                    "  Progress: %d/%d | Acc: %.4f",
                    i + 1, len(test_data), correct / (i + 1),
                )

        accuracy = correct / len(test_data) if test_data else 0.0
        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_data),
            "benchmark": "gsm8k",
            "per_question": results,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "GSM8K Accuracy: %.4f (%d/%d)", accuracy, correct, len(test_data),
        )
        return summary
