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

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


# Final-answer extraction:
#   1) "So the answer is X." / "The answer is: X"     (CoT convention)
#   2) trailing "Answer: X"                            (alt form)
#   3) last non-empty line                             (last-resort)
_ANSWER_PATTERNS = [
    re.compile(r"(?:so\s+)?the\s+answer\s+is[:\s]+([^\.\n]+)", re.IGNORECASE),
    re.compile(r"^\s*answer\s*[:=]\s*([^\.\n]+)", re.IGNORECASE | re.MULTILINE),
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
    # Phase 2: no explicit "answer is" — look for canonical BBH labels
    # anywhere in the response and return the LAST occurrence.
    for lpat in _BBH_LABEL_PATTERNS:
        ms = list(lpat.finditer(text))
        if ms:
            return ms[-1].group(0)
    # Last-resort: last non-empty line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1].rstrip(".").strip() if lines else text.strip()


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
        max_new_tokens: int = 512,
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

        per_task: List[Dict[str, Any]] = []
        total_correct = 0
        total_count = 0
        used_cot_count = 0

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
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                response = tokenizer.decode(out[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                pred = _extract_answer(response)
                if _normalize(pred) == _normalize(ex["target"]):
                    correct += 1

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
