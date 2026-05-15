"""HumanEval evaluator (pass@1 / pass@10 via the human-eval harness)."""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional

import torch

from .base import BenchmarkEvaluator, register
from ..data.sft_prompts import humaneval_generation_prefix

logger = logging.getLogger(__name__)


# Stop sequences for HumanEval completion truncation. The model is asked to
# fill in ONE function body, but greedy / sampled decoding will happily
# continue past the function end into the next def / class / module-level
# code, top-level prints, or markdown fences from chat-style outputs.
# Concatenating that trailing garbage to the prompt and exec()'ing it
# raises SyntaxError or redefines functions in ways that break the harness
# test cases — so every problem fails and pass@1 collapses to 0-ε.
# Cut at the FIRST occurrence of any of these substrings. Matches the
# stop set used by bigcode-eval-harness and the original codex paper.
_HUMANEVAL_STOP_SEQUENCES = (
    "\nclass ",
    "\ndef ",
    "\n#",
    "\nif __name__",
    "\nprint(",
    "\n\n\n",
    # Chat-style models often wrap code in ``` fences; cut at the closing one.
    "\n```",
)


def _truncate_at_stop(completion: str) -> str:
    """Return ``completion`` up to (but not including) the first stop string."""
    min_idx = len(completion)
    for s in _HUMANEVAL_STOP_SEQUENCES:
        idx = completion.find(s)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return completion[:min_idx]


# Pattern for stripping a chat-style model's leading code-fence opener
# (e.g. "```python\n" / "```\n"). The harness concatenates prompt +
# completion verbatim, so a leading "```python\n" desyncs the indent
# and crashes the test runner.
_LEADING_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n", re.IGNORECASE)


def _postprocess_completion(completion: str) -> str:
    """Clean a raw model completion for `prompt + completion` exec.

    - Strip a leading ```python / ``` code-fence opener (chat-style models).
    - Truncate at the first stop sequence (see ``_HUMANEVAL_STOP_SEQUENCES``).
    - Drop trailing whitespace ONLY — NEVER lstrip / strip, because that
      would eat the 4-space indent that the HumanEval prompt expects
      immediately after the docstring. With indent eaten, the
      `prompt + completion` join puts the function body at column 0
      and the harness raises ``IndentationError`` on every problem.
    """
    completion = _LEADING_FENCE_RE.sub("", completion)
    completion = _truncate_at_stop(completion)
    return completion.rstrip()


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
        n_samples: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 42,
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

        # Paper §D: pass@10 with temperature=0.8, top_p=0.95, n=20 completions
        # per problem (n=20 gives a low-variance pass@10 estimate; n=10 is the
        # minimum needed but noisy). When n_samples == 1 we silently fall back
        # to greedy (do_sample=False) for cheap dry-runs.
        use_sampling = n_samples > 1
        # Seed once before the generation loop so the sampling sequence is
        # deterministic — without this every rerun yields a different pass@10.
        if use_sampling:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        completions: Dict[str, list] = {}
        for i, problem in enumerate(problems):
            prefix = humaneval_generation_prefix(
                problem["prompt"], prompt_style=prompt_style,
            )
            inputs = tokenizer(
                prefix, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)
            # pad_token_id is required to silence transformers' warning on
            # every generate() call (the loader already aliases pad→eos when
            # the tokenizer ships without an explicit pad token, but passing
            # the id makes the contract explicit and survives pickle round-
            # trips that occasionally reset tokenizer.pad_token to None).
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            if use_sampling:
                gen_kwargs.update(
                    do_sample=True, temperature=temperature, top_p=top_p,
                    num_return_sequences=n_samples,
                )
            else:
                gen_kwargs.update(do_sample=False, temperature=0.0)
            out = model.generate(**inputs, **gen_kwargs)
            # Token-id slicing — see tydiqa.py comment for why a
            # `completion[len(prefix):]` char-offset slice can't survive
            # the tokenizer's BOS auto-prepend + decode strip round-trip.
            prefix_tok_len = inputs["input_ids"].shape[1]
            for j in range(out.shape[0]):
                # Do NOT .strip() here. HumanEval prompts end at the
                # function-body indent column (typically 4 spaces after a
                # docstring), and `evaluate_functional_correctness` glues
                # `prompt + completion` verbatim before exec()'ing. A
                # leading lstrip would eat those 4 spaces and turn every
                # body into an IndentationError → pass@k = 0. The previous
                # version's .strip() was the proximate cause of the
                # bench-wide score collapse alongside the missing stop-seq
                # truncation. _postprocess_completion handles fence strip,
                # stop-sequence cut, and rstrip only.
                raw = tokenizer.decode(
                    out[j, prefix_tok_len:], skip_special_tokens=True,
                )
                completion = _postprocess_completion(raw)
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

        # The reference scorer lives in the optional `human_eval` package
        # (`pip install human-eval`). We surface a clear error if it's
        # missing rather than a confusing NameError on the call below;
        # the temp completions file is removed either way.
        try:
            try:
                from human_eval.evaluation import (
                    evaluate_functional_correctness,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "HumanEval requires the `human_eval` package. "
                    "Install it with `pip install human-eval` (note the dash) "
                    "and rerun. n_samples completions have been written to "
                    f"{temp_path}; you can score them later with the same "
                    "package.",
                ) from exc
            pass_at_k = evaluate_functional_correctness(
                sample_file=temp_path, k=[1, 10], timeout=10, n_workers=4,
            )
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        summary = {
            "pass@1": pass_at_k.get("pass@1", 0.0),
            "pass@10": pass_at_k.get("pass@10", 0.0),
            "num_problems": len(problems),
            "benchmark": "humaneval",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "HumanEval | pass@10: %.4f | pass@1: %.4f | n_samples=%d | problems=%d",
            summary["pass@10"], summary["pass@1"], n_samples, len(problems),
        )
        return summary
