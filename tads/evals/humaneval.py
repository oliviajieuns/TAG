"""HumanEval evaluator (pass@1 / pass@10 via the human-eval harness)."""
from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import humaneval_generation_prefix

logger = logging.getLogger(__name__)


@register("humaneval")
class HumanEvalEvaluator(BenchmarkEvaluator):
    """HumanEval functional correctness.

    Requires the ``human_eval`` package and a path to ``HumanEval.jsonl.gz``.
    """

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
        n_samples: int = 10,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "HumanEval: `data_dir` is required (path to HumanEval.jsonl.gz "
                "or its parent directory).",
            )

        if os.path.isdir(data_dir):
            data_path = os.path.join(data_dir, "HumanEval.jsonl.gz")
        else:
            data_path = data_dir

        with gzip.open(data_path, "rt") as f:
            problems = [json.loads(line) for line in f]
        if limit is not None:
            problems = problems[:limit]
        logger.info("HumanEval: %d problems | limit=%s", len(problems), limit)

        # Paper §D: pass@10 with temperature=0.8, top_p=0.95.
        # We draw n_samples completions per problem. When n_samples == 1 we
        # silently fall back to greedy (do_sample=False) for cheap dry-runs.
        use_sampling = n_samples > 1
        completions: Dict[str, list] = {}
        for i, problem in enumerate(problems):
            prefix = humaneval_generation_prefix(
                problem["prompt"], prompt_style=prompt_style,
            )
            inputs = tokenizer(
                prefix, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)
            gen_kwargs = dict(max_new_tokens=max_new_tokens)
            if use_sampling:
                gen_kwargs.update(
                    do_sample=True, temperature=temperature, top_p=top_p,
                    num_return_sequences=n_samples,
                )
            else:
                gen_kwargs.update(do_sample=False, temperature=0.0)
            out = model.generate(**inputs, **gen_kwargs)
            for j in range(out.shape[0]):
                completion = tokenizer.decode(out[j], skip_special_tokens=True)
                completion = completion[len(prefix):].strip()
                completions.setdefault(problem["task_id"], []).append(completion)
            if (i + 1) % 20 == 0:
                logger.info(
                    "  Progress: %d/%d (n_samples=%d)",
                    i + 1, len(problems), n_samples,
                )

        # Hand off to the official harness.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as temp_file:
            for task_id, comps in completions.items():
                for c in comps:
                    temp_file.write(
                        json.dumps({"task_id": task_id, "completion": c}) + "\n"
                    )
            temp_path = temp_file.name

        try:
            from human_eval.evaluation import evaluate_functional_correctness
            pass_at_k = evaluate_functional_correctness(
                sample_file=temp_path, k=[1, 10], timeout=10, n_workers=4,
            )
        finally:
            os.unlink(temp_path)

        summary = {
            "pass@1": pass_at_k.get("pass@1", 0.0),
            "pass@10": pass_at_k.get("pass@10", 0.0),
            "num_problems": len(problems),
            "benchmark": "humaneval",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("HumanEval Pass@1: %.4f", summary["pass@1"])
        return summary
