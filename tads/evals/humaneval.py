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


def _extract_body_after_signature(completion: str, entry_point: str) -> Optional[str]:
    """If the completion re-emits the function signature, peel it off and
    return just the body lines (so `prompt + body` is syntactically valid).

    Chat-style instruction-tuned models routinely answer the HumanEval
    prompt with the full function rewritten — sometimes inside a
    ```python fence, sometimes prefixed with prose. Concatenating
    `prompt + "def has_close_elements(...):"` to the original prompt
    redefines the function inside its own body and crashes the test
    runner. When we can find a signature line for ``entry_point`` we
    return everything AFTER it (i.e. the body), so the harness ends up
    exec'ing one well-formed definition.

    Returns None if no signature is found (the caller falls back to the
    raw continuation path).
    """
    sig_pat = re.compile(
        rf"^[ \t]*def\s+{re.escape(entry_point)}\s*\(",
        re.MULTILINE,
    )
    m = sig_pat.search(completion)
    if not m:
        return None
    # Skip past the signature line (find the next newline AFTER the `def`).
    nl = completion.find("\n", m.end())
    if nl == -1:
        return None
    body = completion[nl + 1:]
    return body


def _postprocess_completion(completion: str, entry_point: str = "") -> str:
    """Clean a raw model completion for `prompt + completion` exec.

    - Strip a leading ```python / ``` code-fence opener (chat-style models).
    - If the model rewrote the whole function (signature echoed), peel
      the signature off so we exec exactly one definition.
    - Truncate at the first stop sequence (see ``_HUMANEVAL_STOP_SEQUENCES``).
    - Drop trailing whitespace ONLY — NEVER lstrip / strip, because that
      would eat the 4-space indent that the HumanEval prompt expects
      immediately after the docstring. With indent eaten, the
      `prompt + completion` join puts the function body at column 0
      and the harness raises ``IndentationError`` on every problem.
    """
    completion = _LEADING_FENCE_RE.sub("", completion)
    if entry_point:
        body = _extract_body_after_signature(completion, entry_point)
        if body is not None:
            completion = body
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
        max_new_tokens: int = 512,
        n_samples: int = 20,
        n_samples_per_batch: int = 4,
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
        # Canonical HumanEval schema (openai/human-eval HumanEval.jsonl.gz):
        # task_id, prompt, entry_point, canonical_solution, test. The eval
        # uses prompt + entry_point; the scoring harness uses test. Validate
        # all 4 up-front so the run aborts cleanly if someone points
        # HUMANEVAL_DATA_DIR at a different jsonl (e.g. HumanEval-X or
        # MultiPL-E variants that rename fields). Without the check the
        # generation loop would KeyError on the first problem after model
        # load.
        _need_keys = {"task_id", "prompt", "entry_point", "test"}
        if not problems:
            raise ValueError(
                f"HumanEval file at {data_path} is empty. Expected 164 "
                f"problems from openai/human-eval HumanEval.jsonl.gz."
            )
        _missing = _need_keys - set(problems[0].keys())
        if _missing:
            raise ValueError(
                f"HumanEval file at {data_path} has wrong schema. "
                f"Expected keys {sorted(_need_keys)} per problem (openai/"
                f"human-eval canonical jsonl), got {sorted(problems[0].keys())}. "
                f"Missing: {sorted(_missing)}. This is NOT the HumanEval "
                f"dataset the eval expects — fetch the canonical file via "
                f"`bash scripts/download_humaneval.sh ${{HUMANEVAL_DATA_DIR}}` "
                f"and re-run.",
            )
        # NAIT paper Table 2 reports HumanEval on the canonical 164 problems.
        # Counts ≪ 164 mean a partial / truncated download (e.g. the user
        # only had a debug subset on disk).
        if not limit and len(problems) < 164:
            logger.warning(
                "HumanEval: only %d problems found — canonical set is 164. "
                "Score will not be comparable to NAIT paper Table 2.",
                len(problems),
            )
        if limit is not None:
            problems = problems[:limit]
        logger.info("HumanEval: %d problems | limit=%s", len(problems), limit)

        # Truncation safety: HumanEval prompt ends with the function signature
        # + docstring that the model continues. Default `truncation_side=
        # 'right'` would cut the docstring (loses examples / type hints).
        # Left-truncation cuts imports/preamble — model still has signature
        # + docstring to work with. Most prompts fit in 2048 tokens; this
        # only matters for the small number of long-docstring problems.
        tokenizer.truncation_side = "left"

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

        # Memory: generating all 20 sequences in one `model.generate` call
        # blows up the KV cache and intermediate softmax tensors by 20×
        # (transformers' generate batches num_return_sequences exactly like
        # a real batch, no PagedAttention). For a 7B bf16 model that pushes
        # peak VRAM well past 80 GB even before activations. Chunk the
        # n_samples into mini-batches of size ``n_samples_per_batch`` and
        # accumulate completions across the chunks. The total compute is
        # identical; peak memory drops by n_samples / n_samples_per_batch.
        per_call = max(1, int(n_samples_per_batch))
        if use_sampling:
            n_calls = (n_samples + per_call - 1) // per_call
        else:
            # Greedy: a single deterministic completion is the whole signal.
            n_calls = 1
            per_call = 1

        # Telemetry for max-tokens truncation. If completions consistently hit
        # max_new_tokens with no natural stop (no EOS, no stop_sequence match),
        # bump max_new_tokens higher and re-run. Logged in summary so a
        # systematically-underscoring run is diagnosable from the JSON alone.
        n_truncated_samples = 0
        n_total_samples = 0

        # Diagnostics — preserved across problems so we can dump a sample at
        # the end of the run. The first problem's first completion is shown
        # in the summary JSON so a 0-score run is immediately diagnosable
        # ("model produced empty / non-code / fragmented output" vs "scoring
        # harness threw" vs "stop sequences cut too aggressive").
        first_raw_sample: Optional[str] = None
        first_postprocessed: Optional[str] = None

        completions: Dict[str, list] = {}
        for i, problem in enumerate(problems):
            prefix = humaneval_generation_prefix(
                problem["prompt"], prompt_style=prompt_style,
            )
            inputs = tokenizer(
                prefix, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)
            prefix_tok_len = inputs["input_ids"].shape[1]
            entry_point = problem.get("entry_point", "")

            remaining = n_samples if use_sampling else 1
            for call_idx in range(n_calls):
                this_call = min(per_call, remaining)
                # pad_token_id is required to silence transformers' warning
                # (the loader already aliases pad→eos when the tokenizer
                # ships without one, but passing the id makes the contract
                # explicit and survives pickle round-trips that occasionally
                # reset tokenizer.pad_token to None).
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                if use_sampling:
                    gen_kwargs.update(
                        do_sample=True, temperature=temperature, top_p=top_p,
                        num_return_sequences=this_call,
                    )
                else:
                    gen_kwargs.update(do_sample=False, temperature=0.0)
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
                # Token-id slicing — see tydiqa.py comment for why a
                # `completion[len(prefix):]` char-offset slice can't survive
                # the tokenizer's BOS auto-prepend + decode strip round-trip.
                gen_tok_len = out.shape[1] - prefix_tok_len
                for j in range(out.shape[0]):
                    # Do NOT .strip() here. HumanEval prompts end at the
                    # function-body indent column (typically 4 spaces after
                    # a docstring); a leading lstrip eats those 4 spaces and
                    # turns every body into an IndentationError → pass@k=0.
                    raw = tokenizer.decode(
                        out[j, prefix_tok_len:], skip_special_tokens=True,
                    )
                    # Truncation telemetry: a generation that consumed all
                    # max_new_tokens AND has no stop-sequence match in the
                    # raw text never reached a natural end. With sampling
                    # this happens on long-body problems (loops / recursion)
                    # and the resulting completion is usually a partial
                    # function body that fails exec → pass@1=0. Count.
                    n_total_samples += 1
                    truncated = (
                        gen_tok_len >= max_new_tokens
                        and not any(s in raw for s in _HUMANEVAL_STOP_SEQUENCES)
                    )
                    if truncated:
                        n_truncated_samples += 1
                    completion = _postprocess_completion(raw, entry_point=entry_point)
                    completions.setdefault(problem["task_id"], []).append(completion)
                    if first_raw_sample is None:
                        first_raw_sample = raw[:1000]
                        first_postprocessed = completion[:1000]
                        logger.info(
                            "HumanEval sample[0] | raw=%r ... | "
                            "postprocessed=%r ...",
                            (raw[:200] + "...") if len(raw) > 200 else raw,
                            (completion[:200] + "...") if len(completion) > 200 else completion,
                        )
                remaining -= this_call
                # Drop the call-local tensor before allocating the next
                # chunk's KV cache. Without this the prior chunk's `out`
                # lingers until the next `out = ...` overwrites it, and
                # for max_new_tokens=256 × n_samples_per_batch that's a
                # multi-GB orphan held alive past its useful life.
                del out

            # Per-problem allocator cleanup. transformers' generate leaves
            # ~hundreds of MB of cached tensors in PyTorch's allocator
            # arena; without an explicit empty_cache between problems the
            # high-water mark for 164 problems × 20 samples stays at the
            # union of everything ever allocated and pushes a 7B run past
            # 80 GB even though the steady state would fit in ~30 GB.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 20 == 0:
                logger.info(
                    "  Progress: %d/%d (n_samples=%d, chunks=%d×%d)",
                    i + 1, len(problems), n_samples, n_calls, per_call,
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
            logger.info("HumanEval evaluate_functional_correctness → %r", pass_at_k)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        trunc_pct = (
            100.0 * n_truncated_samples / n_total_samples
            if n_total_samples else 0.0
        )
        if trunc_pct >= 5.0:
            logger.warning(
                "HumanEval: %d/%d samples (%.1f%%) hit max_new_tokens=%d "
                "without a natural stop — those completions are partial "
                "function bodies that fail exec. Increase max_new_tokens "
                "(currently %d) to recover the lost pass@k probability mass.",
                n_truncated_samples, n_total_samples, trunc_pct,
                max_new_tokens, max_new_tokens,
            )

        # NAIT paper Table 2 reports HumanEval as pass@10 (n=20 sampled
        # completions at T=0.8, p=0.95 — see Appendix D). pass@10 is therefore
        # the canonical primary metric for paper-comparable score-board
        # parsing. `accuracy` field aliases pass@10 so the §5-5 score reader
        # picks it without bench-specific branching ("accuracy" 후보 키 사용).
        # pass@1 is still recorded for diagnostic / debugging only.
        # Surface the raw pass_at_k dict in case the human_eval package
        # returned keys we don't recognize (some forks use pass_at_1 with
        # underscore, or numeric k as the key). If pass@10 is 0.0 the user
        # can inspect this field to see whether the harness produced
        # anything at all.
        raw_pass_at_k = {str(k): v for k, v in pass_at_k.items()}
        summary = {
            "accuracy": pass_at_k.get("pass@10", 0.0),    # primary = pass@10 (NAIT)
            "pass@10": pass_at_k.get("pass@10", 0.0),
            "pass@1":  pass_at_k.get("pass@1", 0.0),      # diagnostic, not primary
            "primary_metric": "pass@10",
            "num_problems": len(problems),
            "benchmark": "humaneval",
            # Diagnostics — surface in JSON so a low score's cause is visible
            # without re-reading the eval log.
            "raw_pass_at_k": raw_pass_at_k,
            "n_samples": n_samples,
            "n_total_samples": n_total_samples,
            "n_truncated_samples": n_truncated_samples,
            "truncated_pct": round(trunc_pct, 2),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "prompt_style": prompt_style,
            # First completion sample — instantly tells you whether the model
            # is producing valid code, garbage, or empty. Without this a 0
            # score is impossible to diagnose without re-running.
            "first_sample_raw": first_raw_sample,
            "first_sample_postprocessed": first_postprocessed,
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "HumanEval | pass@10: %.4f | pass@1: %.4f | n_samples=%d | "
            "problems=%d | truncated=%d/%d (%.1f%%)",
            summary["pass@10"], summary["pass@1"], n_samples, len(problems),
            n_truncated_samples, n_total_samples, trunc_pct,
        )
        return summary
