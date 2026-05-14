"""MMLU evaluator (5-shot, logit-based choice scoring)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


CHOICES = ["A", "B", "C", "D"]
MMLU_TEMPLATE = """{question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer:"""


def _format_subject(subject: str) -> str:
    return subject.replace("_", " ").title()


def _build_few_shot_prompt(dev_examples, test_example, subject):
    subj = _format_subject(subject)
    parts = [
        f"The following are multiple choice questions (with answers) about {subj}.\n\n",
    ]
    for ex in dev_examples:
        parts.append(
            MMLU_TEMPLATE.format(
                question=ex["question"],
                A=ex["choices"][0], B=ex["choices"][1],
                C=ex["choices"][2], D=ex["choices"][3],
            ).strip()
            + f" {CHOICES[ex['answer']]}\n\n"
        )
    parts.append(MMLU_TEMPLATE.format(
        question=test_example["question"],
        A=test_example["choices"][0], B=test_example["choices"][1],
        C=test_example["choices"][2], D=test_example["choices"][3],
    ))
    return "".join(parts)


def _get_choice_token_ids(tokenizer):
    ids = []
    for c in CHOICES:
        for candidate in (f" {c}", c):
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
                break
    return ids


@register("mmlu")
class MMLUEvaluator(BenchmarkEvaluator):
    """MMLU 5-shot, choose argmax over (A,B,C,D) next-token logits."""

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
        num_fewshot: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        import pandas as pd

        if data_dir is None:
            raise ValueError(
                "MMLU: `data_dir` is required (point to a directory of "
                "parquet files containing the MMLU `all` split).",
            )

        choice_ids = _get_choice_token_ids(tokenizer)

        # IMPORTANT: load *only* the test split. The "all" config on HF ships
        # multiple parquet files (dev, validation, auxiliary_train, test);
        # naively loading every .parquet contaminates the test pool.
        test_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("test-") and f.endswith(".parquet")
        )
        if not test_files:
            raise FileNotFoundError(
                f"No `test-*.parquet` files found in {data_dir}. "
                "Expected the MMLU `all` directory.",
            )
        dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in test_files]
        test_df = pd.concat(dfs, ignore_index=True)
        dev_df = pd.read_parquet(os.path.join(data_dir, "dev-00000-of-00001.parquet"))

        subjects = test_df["subject"].unique()
        logger.info("MMLU: %d subjects | limit=%s", len(subjects), limit)

        results = []
        for subject in subjects:
            dev_examples = dev_df[dev_df["subject"] == subject].to_dict("records")[:num_fewshot]
            test_examples = test_df[test_df["subject"] == subject].to_dict("records")
            if limit is not None:
                test_examples = test_examples[:limit]

            correct, total = 0, 0
            for ex in test_examples:
                prompt = _build_few_shot_prompt(dev_examples, ex, subject)
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048,
                ).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                logits = out.logits[0, -1, :]
                pred = logits[choice_ids].argmax().item()
                correct += int(pred == ex["answer"])
                total += 1

            acc = correct / total if total else 0.0
            results.append({
                "subject": subject, "accuracy": acc,
                "correct": correct, "total": total,
            })
            logger.info("  %s: %.4f", subject, acc)

        total_correct = sum(r["correct"] for r in results)
        total_total = sum(r["total"] for r in results)
        overall = total_correct / total_total if total_total else 0.0
        summary = {
            "overall_accuracy": overall,
            "total_correct": total_correct,
            "total_questions": total_total,
            "num_subjects": len(results),
            "per_subject": results,
            "benchmark": "mmlu",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MMLU overall: %.4f", overall)
        return summary
