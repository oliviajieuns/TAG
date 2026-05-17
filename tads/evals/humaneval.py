"""HumanEval evaluator (pass@1 / pass@10 via the human-eval harness)."""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
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
    # Alpaca-template SFT models often hallucinate a NEXT instruction/response
    # turn after answering — `\n### Instruction:` / `\n### Response:`. Without
    # this stop, the model fills max_new_tokens with garbage which then
    # breaks the body-extraction (extra ### markers confuse text slicing).
    "\n### Instruction",
    "\n### Response",
    # ChatML / Qwen variants of the same hallucinated-turn pattern.
    "\n<|im_start|>",
    "\n<|im_end|>",
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

# Common Alpaca / ChatML preamble openings that the SFT model emits before
# the actual function body. We don't try to enumerate every variant — we
# just detect "any leading line that doesn't look like code" and skip past
# it down to the first line that looks like code. See _strip_prose_preamble.
_PROSE_OPENERS = (
    "sure", "here", "of course", "certainly", "i'll", "i will",
    "the function", "the answer", "below is", "this function",
    "to solve", "to complete", "we can", "let me", "let's",
)


def _looks_like_code_line(line: str) -> bool:
    """Heuristic: True if a line looks like Python source (vs prose).

    Used to skip prose preambles a chat-style model emits before the
    actual function body. Tuned to be lenient on the code side and
    strict on the prose side so we don't accidentally drop the first
    valid line of code.
    """
    s = line.lstrip()
    if not s:
        return False
    # Code-looking starting tokens (incomplete but high-precision).
    code_starts = (
        "def ", "class ", "import ", "from ", "return ", "if ", "elif ",
        "else", "for ", "while ", "try", "except", "finally",
        "with ", "raise ", "yield ", "lambda ", "@",
        "pass", "break", "continue", "global ", "nonlocal ",
        "assert ", "del ",
    )
    if s.startswith(code_starts):
        return True
    # Assignment / call / numeric / comment / dunder — also code.
    # We accept anything that has a `=`, `(`, `[`, `.` or starts with a
    # quote/digit/underscore as code-ish. Prose almost always starts with
    # a capital letter and a space (English sentence start).
    if s[0] in "_0123456789\"'#":
        return True
    if "(" in s or "=" in s or "." in s or "[" in s:
        return True
    return False


def _strip_prose_preamble(completion: str) -> str:
    """Drop a leading prose preamble. Returns the suffix starting at the
    first code-looking line.

    A common Alpaca-SFT failure mode: model emits "Sure! Here's the
    function:\n\ndef foo(...)..." or "Let me solve this step by step.\n
    ...code...". Without stripping the prose, the harness gets
    `prompt + "Sure! Here's..."` which is a SyntaxError on every problem.
    """
    if not completion:
        return completion
    lines = completion.split("\n")
    # If FIRST non-blank line already looks like code, no preamble.
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _looks_like_code_line(line):
            return completion if i == 0 else "\n".join(lines[i:])
        # Cheap early-out: if the first non-blank line starts with a known
        # prose opener, we're in preamble mode for sure.
        first_word = line.strip().lower().split()[:1]
        first_word = first_word[0] if first_word else ""
        if not (first_word.startswith(_PROSE_OPENERS) if first_word else False):
            # Not a known opener and not code-looking — still likely prose,
            # but be conservative: only skip up to the first code-looking
            # line, found via the second loop below.
            pass
        break
    # Find the first line that DOES look like code, and start there.
    for i, line in enumerate(lines):
        if _looks_like_code_line(line):
            return "\n".join(lines[i:])
    # No code-looking line found — return the original so the harness sees
    # the empty/prose completion and counts it as a failure (vs. silently
    # returning ""). The diagnostic JSON will show the prose.
    return completion


def _strip_echoed_prompt(completion: str, problem_prompt: str) -> str:
    """If the model echoed the original prompt at the start of its
    completion (common with chat-style wrappers), peel it off.

    We test the model's continuation against the LAST 60 chars of the
    prompt's docstring closing — that's the section most likely to be
    echoed verbatim, and matching the full prompt would be foiled by any
    whitespace normalisation the tokenizer applied during decode.
    """
    if not problem_prompt or not completion:
        return completion
    # Find the tail of the prompt that the model might have echoed.
    tail = problem_prompt.rstrip()[-60:]
    if not tail:
        return completion
    idx = completion.find(tail)
    if idx < 0:
        return completion
    return completion[idx + len(tail):]


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
    # If the body line of the re-emitted function has its own docstring,
    # also skip past that — otherwise we exec the prompt's docstring AND
    # the model's docstring back-to-back. Detect by checking if the first
    # non-blank body line starts with a triple-quote.
    stripped = body.lstrip("\n")
    leading_ws = body[:len(body) - len(stripped)]
    if stripped.lstrip().startswith(('"""', "'''")):
        # Find the closing triple-quote and skip past it.
        quote = '"""' if stripped.lstrip().startswith('"""') else "'''"
        end = stripped.find(quote, stripped.find(quote) + 3)
        if end != -1:
            nl2 = stripped.find("\n", end + 3)
            if nl2 != -1:
                body = leading_ws + stripped[nl2 + 1:]
    return body


def _postprocess_completion(
    completion: str,
    entry_point: str = "",
    problem_prompt: str = "",
) -> str:
    """Clean a raw model completion for `prompt + completion` exec.

    Pipeline (order matters):
      1. Strip a leading ```python / ``` code-fence opener.
      2. Strip an echo of the original prompt (chat-style models repeat
         the question before answering).
      3. Strip a prose preamble ("Sure! Here's the function:"). Tuned to
         skip down to the first code-looking line.
      4. If the model rewrote the whole function (signature echoed), peel
         the signature (+ its own docstring) off so the harness exec's
         exactly one definition.
      5. Truncate at the first stop sequence.
      6. Drop trailing whitespace ONLY — NEVER lstrip / strip, because
         the HumanEval prompt ends right before the body indent column
         (typically 4 spaces). If we lstrip here, `prompt + completion`
         puts the function body at column 0 and the harness raises
         ``IndentationError`` on every problem.
    """
    completion = _LEADING_FENCE_RE.sub("", completion)
    completion = _strip_echoed_prompt(completion, problem_prompt)
    if entry_point:
        body = _extract_body_after_signature(completion, entry_point)
        if body is not None:
            completion = body
        else:
            # Prose-only preamble before a body-style continuation (no
            # signature echo) — strip it. Skip THIS branch when the
            # signature path already kicked in, to avoid double-stripping
            # the body's first code line.
            completion = _strip_prose_preamble(completion)
    else:
        completion = _strip_prose_preamble(completion)
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

        # Diagnostics — preserved across problems so we can dump samples at
        # the end of the run. The first 3 problems' first completions are
        # shown in the summary JSON so a 0-score run is immediately
        # diagnosable ("model produced empty / non-code / fragmented output"
        # vs "scoring harness threw" vs "stop sequences cut too aggressive").
        # 3 samples (was 1) so a per-problem pattern is visible. Each entry
        # is {task_id, raw, postprocessed}.
        N_DIAG_SAMPLES = 3
        first_raw_samples: list = []

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
                    completion = _postprocess_completion(
                        raw,
                        entry_point=entry_point,
                        problem_prompt=problem["prompt"],
                    )
                    completions.setdefault(problem["task_id"], []).append(completion)
                    # Capture the FIRST sample of each of the first N_DIAG
                    # problems. j==0 to ensure we only record one per problem
                    # (not all 20 sampled completions of problem 0).
                    if (
                        j == 0
                        and len(first_raw_samples) < N_DIAG_SAMPLES
                    ):
                        first_raw_samples.append({
                            "task_id": problem["task_id"],
                            "raw": raw[:1500],
                            "postprocessed": completion[:1500],
                        })
                        logger.info(
                            "HumanEval sample %s | raw=%r ... | "
                            "postprocessed=%r ...",
                            problem["task_id"],
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
            # Drop `inputs` BEFORE empty_cache so the allocator can actually
            # reclaim the prefix tensor (would otherwise survive until the
            # next problem's `inputs = ...` rebinding).
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 20 == 0:
                logger.info(
                    "  Progress: %d/%d (n_samples=%d, chunks=%d×%d)",
                    i + 1, len(problems), n_samples, n_calls, per_call,
                )

        # Hand off to the official harness.
        # Persist the completions file alongside `output_file` so a low
        # score is debuggable offline (re-run the scorer with different
        # timeout, manually exec a single completion, etc.). The old
        # tempfile.NamedTemporaryFile path got auto-deleted in the
        # `finally` block below, leaving zero trace of what the model
        # actually produced — making "why is pass@10 = 5%?" un-diagnosable
        # without a full re-run.
        completions_path = os.path.splitext(output_file)[0] + "_completions.jsonl"
        with open(completions_path, "w") as cf:
            for task_id, comps in completions.items():
                for c in comps:
                    cf.write(
                        json.dumps({"task_id": task_id, "completion": c}) + "\n"
                    )
        temp_path = completions_path

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
            # timeout 10s → 30s: matches bigcode-evaluation-harness default.
            # A real fraction of HumanEval problems have recursion-heavy or
            # combinatoric reference solutions that 10s isn't enough for on
            # model-generated (unoptimised) code. We were losing ~1–3pt of
            # pass@10 to spurious timeouts.
            pass_at_k = evaluate_functional_correctness(
                sample_file=temp_path, k=[1, 10], timeout=30, n_workers=4,
            )
            logger.info("HumanEval evaluate_functional_correctness → %r", pass_at_k)
        finally:
            # Keep completions_path on disk for offline debugging. The
            # human-eval package also writes a `*_results.jsonl` next to it
            # with per-sample pass/fail + error info — leave that alone too.
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
            # First N=3 problems' raw + postprocessed samples — instantly
            # tells you whether the model is producing valid code, garbage,
            # or empty. Without this a 0 score is impossible to diagnose
            # without re-running.
            "first_samples": first_raw_samples,
            # Persistent completions file (jsonl, one line per (task, sample))
            # — for offline re-scoring with a different timeout, or for
            # diff'ing two runs' completions.
            "completions_file": completions_path,
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
