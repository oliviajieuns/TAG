"""Big-Bench Hard (BBH) evaluator — 5-shot Chain-of-Thought.

Paper context: NAIT (ICLR 2026, Appendix D) reports 5-shot CoT on BBH's
27 sub-tasks. Final score = mean of per-task accuracy.

Expected layout under ``data_dir`` (matches the official suzgun/BIG-Bench-Hard
distribution):

    bbh/
    ├── bbh/                      # 27 task files
    │   ├── boolean_expressions.json
    │   ├── causal_judgement.json
    │   ├── …
    └── cot-prompts/              # official 3-shot CoT prefixes (optional)
        ├── boolean_expressions.txt
        ├── …

If a task's cot-prompts file is present we use it as the few-shot prefix
(NAIT-paper faithful). Otherwise we fall back to constructing few-shot
demonstrations from the task's own first ``n_fewshot`` examples, which
yields a non-CoT direct-answer prompt — weaker but still functional.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


# Final-answer extraction:
#   1) "So the answer is X." / "The answer is: X"     (CoT convention)
#   2) trailing "Answer: X"                            (alt form)
#   3) last non-empty line                             (last-resort)
_ANSWER_PATTERNS = [
    # Non-greedy capture so "The answer is True. However, ..." → "True"
    # rather than "True. However, " — the greedy version was pulling
    # trailing prose into the canonical-label fallback and occasionally
    # missing the label entirely. The terminator class still rejects
    # period and newline so we stop at sentence boundary.
    re.compile(r"(?:so\s+)?the\s+answer\s+is[:\s]+([^\.\n]+?)(?=[\.\n]|$)",
               re.IGNORECASE),
    re.compile(r"^\s*answer\s*[:=]\s*([^\.\n]+?)(?=[\.\n]|$)",
               re.IGNORECASE | re.MULTILINE),
]

# After fallback extraction we additionally try to recover the canonical BBH
# answer shapes — "(A)", "True", "False", "Yes", "No" — that appear in
# several sub-tasks. If the response trails off with prose, look for the
# *last* occurrence of one of these patterns and prefer it over the raw line.
_BBH_LABEL_PATTERNS = [
    re.compile(r"\(([A-Z])\)"),                 # "(A)" multiple-choice
    re.compile(r"\b(True|False)\b", re.IGNORECASE),
    re.compile(r"\b(Yes|No)\b", re.IGNORECASE),
]


def _extract_answer(text: str) -> str:
    # Phase 1: explicit "the answer is X" / "Answer: X" patterns.
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip().rstrip(".").strip()
            # If the captured group itself is prose ending with "(A)" or
            # "True", prefer the trailing label — extracted prose is noisy.
            for lpat in _BBH_LABEL_PATTERNS:
                m2 = list(lpat.finditer(candidate))
                if m2:
                    return m2[-1].group(0)
            return candidate
    # Phase 2: no explicit "answer is" — search ONLY the last non-empty line
    # for canonical BBH labels. Searching the whole response (previous behaviour)
    # picked up labels inside reasoning prose like "if it were True, then ..."
    # and flipped the extracted answer at random. (Audit A3.)
    _lines_p2 = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if _lines_p2:
        _last_line_p2 = _lines_p2[-1]
        for lpat in _BBH_LABEL_PATTERNS:
            m = lpat.search(_last_line_p2)
            if m:
                return m.group(0)
    # Last-resort: last non-empty line. For BBH numeric / word-form tasks that
    # don't match any (A)/(B)/Yes/No/True/False label, a full trailing sentence
    # like "The answer is 42 approximately." used to be returned verbatim and
    # fail the gold-side comparison (which holds just "42"). If the last line
    # contains a number, prefer the LAST numeric span — this matches the gold
    # format for tasks like Dyck Language, Multistep Arithmetic, Object
    # Counting, etc. Word-form tasks (yes/no, fruit names) keep the original
    # last-line behaviour because no number will be found.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    last_line = lines[-1].rstrip(".").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", last_line)
    if nums:
        return nums[-1]
    return last_line


def _normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\(\)]", " ", s)  # keep parens for (A)/(B) style targets
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _list_task_files(data_dir: Path) -> List[Path]:
    """Locate the per-task JSON files. Handles both bbh/bbh/*.json and bbh/*.json."""
    nested = data_dir / "bbh"
    if nested.is_dir():
        files = sorted(nested.glob("*.json"))
        if files:
            return files
    return sorted(data_dir.glob("*.json"))


def _load_cot_prefix(data_dir: Path, task: str) -> Optional[str]:
    """Return the official cot-prompts/<task>.txt content if present."""
    for cand in (
        data_dir / "cot-prompts" / f"{task}.txt",
        data_dir.parent / "cot-prompts" / f"{task}.txt",
    ):
        if cand.exists():
            return cand.read_text().rstrip()
    return None


def _build_prompt(
    task_examples: List[Dict[str, str]],
    test_input: str,
    n_fewshot: int,
    cot_prefix: Optional[str],
) -> Tuple[str, bool]:
    """Return (prompt, used_cot). If cot_prefix is available, use it
    (paper-faithful CoT); otherwise build a direct-answer few-shot prompt
    from the task's own examples."""
    if cot_prefix is not None:
        # cot-prompts files conventionally end with the demos but no test Q yet.
        return f"{cot_prefix}\n\nQ: {test_input}\nA: Let's think step by step.", True

    shots = task_examples[:n_fewshot]
    chunks = [f"Q: {s['input']}\nA: {s['target']}" for s in shots]
    chunks.append(f"Q: {test_input}\nA:")
    return "\n\n".join(chunks), False


@register("bbh")
class BBHEvaluator(BenchmarkEvaluator):
    """Big-Bench Hard with 5-shot CoT (NAIT Appendix D)."""

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",  # unused (BBH has its own format)
        data_dir: Optional[str] = None,
        # max_new_tokens 256 = NAIT CoT answer (typical "Let's think step by
        # step. ... Therefore the answer is X." 100~200 tokens) 안에 충분히
        # 끝남. 512는 worst-case 여백이지만 평균 generation 시간 1.5~2x
        # 증가시켜 BBH 셀당 ~27h → ~15h 감축 효과. 답이 256 token 안에
        # 안 들어오면 truncate되지만 그건 모델이 발산한 케이스라
        # 점수에도 큰 의미 없음.
        max_new_tokens: int = 256,
        n_fewshot: int = 5,
        max_input_tokens: int = 3072,
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "BBH: `data_dir` is required (point to the bbh/ root "
                "containing per-task .json files and optional cot-prompts/)."
            )
        data_root = Path(data_dir)
        task_files = _list_task_files(data_root)
        if not task_files:
            raise FileNotFoundError(
                f"No BBH task .json files under {data_dir} (or {data_dir}/bbh/). "
                "Expected the suzgun/BIG-Bench-Hard layout."
            )

        cot_root = data_root if (data_root / "cot-prompts").is_dir() else task_files[0].parent
        n_cot_files = sum(
            1 for t in task_files
            if (cot_root / "cot-prompts" / f"{t.stem}.txt").exists()
            or (cot_root.parent / "cot-prompts" / f"{t.stem}.txt").exists()
        )
        if n_cot_files == 0:
            # Paper-faithfulness gate: NAIT reports BBH with the official
            # 3-shot CoT prompts. Without them we fall back to a non-CoT
            # direct-answer few-shot which scores ~10-15pt lower. Make this
            # impossible to miss in the logs.
            logger.error(
                "BBH: NO cot-prompts/*.txt files found under %s. The evaluator "
                "will fall back to a non-CoT direct-answer few-shot baseline, "
                "which is NOT paper-faithful. To match NAIT Table 2, clone "
                "github.com/suzgunmirac/BIG-Bench-Hard and point BBH_DATA_DIR "
                "at it (so cot-prompts/<task>.txt exists alongside bbh/<task>.json).",
                cot_root,
            )
        elif n_cot_files < len(task_files):
            logger.warning(
                "BBH: only %d/%d tasks have cot-prompts/*.txt — the remaining "
                "tasks will use direct-answer fallback (not paper-faithful).",
                n_cot_files, len(task_files),
            )
        logger.info(
            "BBH: %d task files | cot_prompts dir: %s | cot_prompt_files=%d | "
            "limit=%s | n_fewshot=%d",
            len(task_files), cot_root, n_cot_files, limit, n_fewshot,
        )

        # Truncation safety: BBH prompt ends with "Q: <test>\nA: Let's think
        # step by step." — the model must continue from that A: prefix. With
        # the HF tokenizer's default `truncation_side='right'`, an over-long
        # cot-prefix + test Q prompt would have the TEST QUESTION truncated
        # away. Left-truncation drops part of the cot-prefix demos from the
        # FRONT instead, which preserves the test query.
        tokenizer.truncation_side = "left"

        per_task: List[Dict[str, Any]] = []
        total_correct = 0
        total_count = 0
        used_cot_count = 0

        # Schema audit BEFORE running any model forward: every task file must
        # be a JSON dict with an `examples` key, each example must have
        # `input` and `target`. NAIT/BBH canonical (suzgun/BIG-Bench-Hard)
        # follows this shape strictly. Mismatches almost always mean the
        # user pointed BBH_DATA_DIR at the wrong repo (e.g. legacy
        # google/BIG-bench dump with `examples[*].inputs` plural, or a
        # tasksource conversion). Abort with a precise error rather than
        # silently producing accuracy=0 across all tasks.
        _bad_tasks: List[str] = []
        for _t in task_files:
            try:
                with open(_t) as _f:
                    _d = json.load(_f)
            except json.JSONDecodeError as _e:
                _bad_tasks.append(f"{_t.name}: JSON decode error ({_e})")
                continue
            if not isinstance(_d, dict) or "examples" not in _d:
                _bad_tasks.append(
                    f"{_t.name}: not a dict with `examples` key "
                    f"(top-level keys: "
                    f"{list(_d.keys())[:5] if isinstance(_d, dict) else type(_d).__name__})"
                )
                continue
            _ex = _d.get("examples")
            if not isinstance(_ex, list) or not _ex:
                _bad_tasks.append(f"{_t.name}: `examples` is not a non-empty list")
                continue
            _first = _ex[0]
            if not (isinstance(_first, dict)
                    and "input" in _first and "target" in _first):
                _bad_tasks.append(
                    f"{_t.name}: example records missing `input` or `target` "
                    f"(got keys: {sorted(_first.keys()) if isinstance(_first, dict) else type(_first).__name__})",
                )
        if _bad_tasks:
            raise ValueError(
                "BBH dataset schema mismatch — refusing to run. "
                f"Expected the suzgun/BIG-Bench-Hard layout (each per-task "
                f"file is a JSON dict `{{'examples': [{{'input': str, "
                f"'target': str}}, ...]}}`). Bad task files (first 5):\n  "
                + "\n  ".join(_bad_tasks[:5])
                + (f"\n  ... +{len(_bad_tasks) - 5} more" if len(_bad_tasks) > 5 else "")
                + f"\nFix: clone https://github.com/suzgunmirac/BIG-Bench-Hard "
                + f"and point BBH_DATA_DIR at the resulting `bbh/` directory.",
            )

        for task_path in task_files:
            task_name = task_path.stem
            with open(task_path) as f:
                data = json.load(f)
            examples = data.get("examples", []) or []
            if not examples:
                continue

            cot_prefix = _load_cot_prefix(cot_root, task_name)
            # When using cot-prompts, all JSON examples are test cases.
            # Without cot-prompts, the first n_fewshot become the demos.
            if cot_prefix is not None:
                test_examples = examples
            else:
                if len(examples) <= n_fewshot:
                    logger.warning(
                        "  %s: too few examples (%d) for n_fewshot=%d, skipping",
                        task_name, len(examples), n_fewshot,
                    )
                    continue
                test_examples = examples[n_fewshot:]

            if limit is not None:
                test_examples = test_examples[: limit]

            correct = 0
            for ex in test_examples:
                prompt, used_cot = _build_prompt(
                    examples, ex["input"], n_fewshot, cot_prefix,
                )
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_input_tokens,
                ).to(device)
                # stop_strings: BBH cot-prompts 포맷에서 "Q:" 는 다음
                # demonstration의 시작. 모델이 현재 답변을 마치고 "Q:"를
                # 다시 emit하면 그 시점에 generation 중단 — 평균 ~30-40%
                # token 절약 + 정답 추출에는 영향 없음(이미 답 부분은
                # 직전에 생성 완료). "Question:" 는 chat-style 변형 대비.
                # transformers >= 4.34 필수 (requirements.txt: >=4.40 OK).
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    stop_strings=["\nQ:", "\n\nQ:", "Question:", "\n\nQuestion:"],
                    tokenizer=tokenizer,
                )
                # Token-id slicing — see tydiqa.py comment for the
                # decode-round-trip BOS / whitespace mismatch the old
                # `response[len(prompt):]` couldn't survive.
                prompt_tok_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    out[0, prompt_tok_len:], skip_special_tokens=True,
                ).strip()

                pred = _extract_answer(response)
                if _normalize(pred) == _normalize(ex["target"]):
                    correct += 1

                # Drop the per-example CUDA tensors and reclaim allocator
                # blocks before the next 3072-token prompt is built. BBH
                # accumulates ~6500 generate() calls per benchmark with
                # prompt lengths near max_input_tokens; without periodic
                # empty_cache the allocator fragments enough that a long
                # task late in the run (notably tracking_shuffled_objects_*)
                # can hang or OOM when it can't find a contiguous KV cache
                # block. Mirrors the same hygiene humaneval.py:320 uses.
                del inputs, out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            n = len(test_examples)
            acc = correct / n if n else 0.0
            per_task.append({
                "task": task_name,
                "accuracy": acc,
                "correct": correct,
                "total": n,
                "used_cot_prompt": bool(cot_prefix),
            })
            total_correct += correct
            total_count += n
            if cot_prefix is not None:
                used_cot_count += 1
            logger.info(
                "  %s: %.4f (%d/%d)%s",
                task_name, acc, correct, n,
                "" if cot_prefix is not None else "  [no cot-prompts file → direct fewshot]",
            )

        # Two summary metrics (NAIT uses macro-average across tasks):
        macro_avg = (
            sum(t["accuracy"] for t in per_task) / len(per_task) if per_task else 0.0
        )
        micro_avg = total_correct / total_count if total_count else 0.0

        summary = {
            # `accuracy` is the cross-bench primary-metric alias the score-board
            # reader picks up without bench-specific branching. NAIT/BBH uses
            # macro-avg across tasks as the headline (paper Table 2), so that's
            # what `accuracy` mirrors. `macro_avg_accuracy` stays as the
            # bench-natural key.
            "accuracy": macro_avg,
            "macro_avg_accuracy": macro_avg,
            "micro_avg_accuracy": micro_avg,
            "total_correct": total_correct,
            "total_questions": total_count,
            "num_tasks": len(per_task),
            "tasks_with_official_cot_prompt": used_cot_count,
            "per_task": per_task,
            "benchmark": "bbh",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "BBH macro-avg: %.4f | micro-avg: %.4f | tasks=%d | cot_prompt_files=%d",
            macro_avg, micro_avg, len(per_task), used_cot_count,
        )
        return summary
