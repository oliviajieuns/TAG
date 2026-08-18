"""SVAMP evaluator — 8-shot CoT, last-number EM.

SVAMP (Patel et al., 2021) is a math word-problem benchmark with 1,000 test
items, structurally similar to GSM8K but with crafted variations that
expose surface-level shortcut learning. NAIT Table 2 evaluates SVAMP with
the same 8-shot CoT prompt + "The answer is X" extraction as GSM8K — so we
reuse the GSM8K CoT prefix builder and grader verbatim. The only material
difference is the input shape: SVAMP's "question" is composed from two
fields (``Body`` + ``Question``), not a single string.

Schema (ChilleD/SVAMP):
    Body / Question / Equation / Answer / Type / ID
The combined question text fed to the model is ``Body + " " + Question``.
The gold answer is a number (int or float) — we stringify for the same
last-number extractor GSM8K uses.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional


from ._gen import gen_batch_size, generate_texts
from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import build_cot_prompt_prefix
from .gsm8k import _extract_predicted_number, _grade, _normalize_answer

logger = logging.getLogger(__name__)


def _combine_question(row: Dict[str, Any]) -> str:
    """SVAMP problems are split into a story Body and a Question line.
    Concatenate with a space so the resulting text reads as one
    natural-language question. Fall back to either field alone if the
    other is missing (some HF mirrors collapse the two)."""
    body = (row.get("Body") or row.get("body") or "").strip()
    question = (row.get("Question") or row.get("question") or "").strip()
    if body and question:
        return f"{body} {question}"
    return body or question


def _gold_answer(row: Dict[str, Any]) -> str:
    """Render the Answer field as a normalised string the GSM8K grader can
    parse. SVAMP stores numeric answers (int or float); cast to str."""
    ans = row.get("Answer", row.get("answer"))
    if ans is None:
        return ""
    # Drop ".0" tail when the answer is an int-valued float — gold "5.0"
    # would otherwise fail string-EM against a model that wrote "5".
    if isinstance(ans, float) and ans.is_integer():
        return str(int(ans))
    return str(ans)


@register("svamp")
class SVAMPEvaluator(BenchmarkEvaluator):
    """SVAMP with 8-shot CoT (NAIT Table 2 — same setup as GSM8K)."""

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",  # unused (SVAMP uses GSM8K CoT)
        data_dir: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        import pandas as pd

        if data_dir is None:
            raise ValueError(
                "SVAMP: `data_dir` is required (path to svamp/ dir containing "
                "test-*.parquet from ChilleD/SVAMP)."
            )

        # SVAMP test parquet location — match download_svamp.sh layout.
        cand_paths = [
            os.path.join(data_dir, "test-00000-of-00001.parquet"),
            os.path.join(data_dir, "test.parquet"),
        ]
        test_path = next((p for p in cand_paths if os.path.exists(p)), None)
        if test_path is None:
            raise FileNotFoundError(
                f"SVAMP test parquet not found in {data_dir}. "
                f"Tried: {cand_paths}. Run `bash scripts/download_svamp.sh "
                f"{data_dir}` first.",
            )
        df = pd.read_parquet(test_path)

        # Tolerate both casings: HF parquet sometimes ships title-case
        # (Body / Question / Answer), sometimes lowercase.
        cols_lower = {c.lower() for c in df.columns}
        if "body" not in cols_lower or "question" not in cols_lower or "answer" not in cols_lower:
            raise ValueError(
                f"SVAMP parquet at {test_path} has wrong schema. "
                f"Expected Body / Question / Answer (any case). "
                f"Got columns: {list(df.columns)}",
            )
        test_data = df.to_dict("records")
        if limit is not None:
            test_data = test_data[:limit]
        logger.info("SVAMP: %d examples | limit=%s", len(test_data), limit)

        # Same left-truncation rationale as GSM8K.
        tokenizer.truncation_side = "left"

        # Batched greedy decoding — see tag/evals/_gen.py. Prompt text,
        # stop strings and grading are unchanged.
        questions = [_combine_question(ex) for ex in test_data]
        prompts = [build_cot_prompt_prefix(q) for q in questions]
        responses = generate_texts(
            model, tokenizer, prompts, device=device,
            max_new_tokens=max_new_tokens, max_input_tokens=2048,
            stop_strings=["\nQ:", "\n\nQ:", "Question:", "\n\nQuestion:"],
            progress_every=200, progress_label="SVAMP",
            score_key=lambda r: _normalize_answer(
                _extract_predicted_number(r.strip())),
        )

        correct = 0
        results = []
        for i, (ex, question, response) in enumerate(
            zip(test_data, questions, responses)
        ):
            gold = _gold_answer(ex)
            response = response.strip()
            # Trim hallucinated next-demo (safety net besides stop_strings).
            for stop in ("\n\nQ:", "\nQ:", "\n\nQuestion:", "\nQuestion:"):
                idx = response.find(stop)
                if idx != -1:
                    response = response[:idx]
                    break

            ok = _grade(response, gold)
            correct += int(ok)
            results.append({
                "id": ex.get("ID") or ex.get("id") or i,
                "question": question,
                "gold": gold,
                "predicted": response,
                "correct": ok,
            })

        accuracy = correct / len(test_data) if test_data else 0.0
        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_data),
            "benchmark": "svamp",
            # The batch size prompts were decoded at. Greedy decoding is
            # padding-invariant in exact arithmetic but not in float, so a
            # long chain-of-thought can fork on a near-tie; recording the
            # batch size is what makes this number reproducible.
            "generation_batch_size": gen_batch_size(),
            "per_question": results,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "SVAMP Accuracy: %.4f (%d/%d)", accuracy, correct, len(test_data),
        )
        return summary
