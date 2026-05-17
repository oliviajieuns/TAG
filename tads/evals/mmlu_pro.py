"""MMLU-Pro evaluator — 5-shot CoT generation + answer extraction (NAIT Appendix D).

MMLU-Pro differs from vanilla MMLU in three ways that drive the eval shape:

  1. **10 options A..J** (not 4) — logit-based choice scoring breaks because
     several tokenizers split letters past 'E' into multi-token sequences.
  2. **Long-form CoT answers** — the canonical eval is generate-and-extract,
     not next-token argmax. NAIT Table 2 reports MMLU-Pro using 5-shot CoT.
  3. **Validation split ships reference CoT** (``cot_content`` column) — these
     are intended as the 5-shot demonstrations. We group by ``category`` so
     each test question is prompted with same-category demonstrations.

Answer extraction prefers explicit patterns ("The answer is (X)" / "answer
is X") then falls back to the LAST parenthesised single letter in the
response, then to the last bare A-J letter. Matches the suzgun/MMLU-Pro
reference extractor and the lm-eval-harness convention.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


# 10 answer letters (A..J). MMLU-Pro caps at 10 options; some questions have
# fewer (the dataset pads `options` so length is always 10, with trailing
# entries left blank or set to "N/A"). The extractor still only accepts the
# first ``len(non_empty_options)`` letters at scoring time.
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


# Extraction patterns, in priority order. Each captures the letter only.
# Non-greedy + sentence-boundary terminator so verbose prose after the
# letter doesn't bleed into the capture.
_ANSWER_PATTERNS = [
    # "The answer is (X)" / "The answer is X" (canonical)
    re.compile(r"answer\s+is\s+\(?([A-J])\)?", re.IGNORECASE),
    # "Answer: X" / "Answer: (X)" trailing form
    re.compile(r"^\s*answer\s*[:=]\s*\(?([A-J])\)?", re.IGNORECASE | re.MULTILINE),
    # "Therefore, (X)" / "So (X)" closing-claim form
    re.compile(r"\b(?:therefore|hence|so|thus)[,\s]+\(?([A-J])\)?\b", re.IGNORECASE),
]
_PAREN_LETTER = re.compile(r"\(([A-J])\)")
_BARE_LETTER = re.compile(r"\b([A-J])\b")


def _extract_answer(text: str, n_options: int) -> Optional[str]:
    """Return the predicted letter (uppercase) or None if nothing extractable.

    ``n_options`` clamps the valid letter range — a 4-option question that
    extracts "F" is a clear hallucination and should count as wrong, not
    silently accepted as letter index 5.
    """
    valid = set(LETTERS[:n_options])
    # Phase 1: explicit "answer is" patterns. Use the LAST match in case the
    # model self-corrects ("The answer is A. Wait, let me redo... The answer
    # is C.").
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            cand = matches[-1].upper()
            if cand in valid:
                return cand
    # Phase 2: last parenthesised letter (common in CoT closing "Hence (B)").
    pm = list(_PAREN_LETTER.finditer(text))
    if pm:
        cand = pm[-1].group(1).upper()
        if cand in valid:
            return cand
    # Phase 3: last bare letter anywhere — weak signal, but better than None.
    bm = list(_BARE_LETTER.finditer(text))
    if bm:
        cand = bm[-1].group(1).upper()
        if cand in valid:
            return cand
    return None


def _format_options(options: List[str]) -> Tuple[str, int]:
    """Render the option list as "A) opt1\nB) opt2\n..." and return
    ``(rendered, n_options_kept)``. Drops trailing blank entries the
    dataset uses to pad to 10."""
    cleaned = [o for o in options if isinstance(o, str) and o.strip() and o.strip().lower() != "n/a"]
    rendered = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(cleaned))
    return rendered, len(cleaned)


def _build_prompt(
    demo_items: List[Dict[str, Any]],
    test_q: str,
    test_opts: List[str],
    category: str,
) -> Tuple[str, int]:
    """5-shot CoT prompt:
        <header>
        Question: <demo1.q>
        Options:
        A) ...
        ...
        Answer: <demo1.cot>  ← already ends with "The answer is (X)"

        Question: <test.q>
        Options:
        A) ...
        ...
        Answer: Let's think step by step.
    """
    head = (
        f"The following are multiple choice questions (with answers) "
        f"about {category}. Think step by step and then finish your answer "
        f"with 'the answer is (X)' where X is the correct letter choice.\n\n"
    )
    chunks: List[str] = [head]
    for d in demo_items:
        d_opts, _ = _format_options(d.get("options") or [])
        d_cot = (d.get("cot_content") or "").strip()
        chunks.append(
            f"Question: {d['question']}\n"
            f"Options:\n{d_opts}\n"
            f"Answer: {d_cot}\n\n"
        )
    test_rendered, n_test_opts = _format_options(test_opts)
    chunks.append(
        f"Question: {test_q}\n"
        f"Options:\n{test_rendered}\n"
        f"Answer: Let's think step by step."
    )
    return "".join(chunks), n_test_opts


@register("mmlu_pro")
class MMLUProEvaluator(BenchmarkEvaluator):
    """MMLU-Pro 5-shot CoT generate + letter extraction (NAIT Appendix D)."""

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",  # unused (MMLU-Pro has its own format)
        data_dir: Optional[str] = None,
        # max_new_tokens=512: CoT reasoning is long. NAIT/MMLU-Pro reference
        # implementations use 512–768; 512 is the sweet spot where >95% of
        # responses terminate naturally inside the budget on 7B models, and
        # the rest are CoT divergence we'd zero anyway.
        max_new_tokens: int = 512,
        num_fewshot: int = 5,
        max_input_tokens: int = 3072,
        **kwargs,
    ) -> Dict[str, Any]:
        import pandas as pd

        if data_dir is None:
            raise ValueError(
                "MMLU-Pro: `data_dir` is required (point to a directory "
                "containing test-*.parquet + validation-*.parquet from "
                "TIGER-Lab/MMLU-Pro)."
            )

        test_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("test-") and f.endswith(".parquet")
        )
        val_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("validation-") and f.endswith(".parquet")
        )
        if not test_files:
            raise FileNotFoundError(
                f"No `test-*.parquet` in {data_dir}. Run "
                f"`bash scripts/download_mmlu_pro.sh {data_dir}` first.",
            )
        if not val_files:
            raise FileNotFoundError(
                f"No `validation-*.parquet` in {data_dir}. The validation "
                f"split is required for 5-shot CoT demonstrations.",
            )

        test_df = pd.concat(
            [pd.read_parquet(os.path.join(data_dir, f)) for f in test_files],
            ignore_index=True,
        )
        val_df = pd.concat(
            [pd.read_parquet(os.path.join(data_dir, f)) for f in val_files],
            ignore_index=True,
        )

        # Schema audit — fail fast on a non-MMLU-Pro mirror before running
        # any model forward. The canonical TIGER-Lab/MMLU-Pro fields are:
        #   question, options, answer, answer_index, category, cot_content
        need = {"question", "options", "answer", "category"}
        miss_test = need - set(test_df.columns)
        miss_val = need - set(val_df.columns)
        if miss_test or miss_val:
            raise ValueError(
                f"MMLU-Pro schema mismatch at {data_dir}.\n"
                f"  test missing: {sorted(miss_test) or 'OK'}\n"
                f"  validation missing: {sorted(miss_val) or 'OK'}\n"
                f"  expected: {sorted(need)} (TIGER-Lab/MMLU-Pro). Got "
                f"test={sorted(test_df.columns)}",
            )
        if "cot_content" not in val_df.columns:
            logger.warning(
                "MMLU-Pro: validation split has no `cot_content` column — "
                "few-shot prompts will lack reference CoT, which degrades "
                "score by ~5–10pt vs paper-faithful.",
            )

        # Group validation demos by category. NAIT/MMLU-Pro Appendix D uses
        # category-matched demos so the 5-shot prefix exposes the model to
        # the same reasoning style as the test question.
        demos_by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for r in val_df.to_dict("records"):
            demos_by_cat.setdefault(r["category"], []).append(r)
        # Truncate to num_fewshot per category for prompt-size determinism.
        for cat in list(demos_by_cat):
            demos_by_cat[cat] = demos_by_cat[cat][:num_fewshot]

        test_records = test_df.to_dict("records")
        if limit is not None:
            test_records = test_records[:limit]
        logger.info(
            "MMLU-Pro: %d test | %d categories | %d-shot | limit=%s",
            len(test_records), len(demos_by_cat), num_fewshot, limit,
        )

        # Left-truncation (see mmlu.py / bbh.py rationale): preserve test
        # question + "Answer: Let's think step by step." suffix; drop demos
        # from the FRONT if the 5-shot prefix overflows max_input_tokens.
        tokenizer.truncation_side = "left"

        per_cat_correct: Dict[str, int] = {}
        per_cat_total: Dict[str, int] = {}
        per_question: List[Dict[str, Any]] = []
        n_extract_fail = 0

        for i, ex in enumerate(test_records):
            cat = ex["category"]
            demos = demos_by_cat.get(cat, [])
            prompt, n_opts = _build_prompt(demos, ex["question"], list(ex["options"]), cat)

            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=max_input_tokens,
            ).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Stop on a new "Question:" — model often hallucinates the
                # next demonstration after answering.
                stop_strings=["\nQuestion:", "\n\nQuestion:"],
                tokenizer=tokenizer,
            )
            prompt_tok_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(
                out[0, prompt_tok_len:], skip_special_tokens=True,
            ).strip()
            # Trim hallucinated next-demo (safety net besides stop_strings).
            for stop in ("\n\nQuestion:", "\nQuestion:"):
                idx = response.find(stop)
                if idx != -1:
                    response = response[:idx]
                    break

            pred = _extract_answer(response, n_opts)
            gold = str(ex["answer"]).strip().upper()
            ok = (pred == gold) if pred is not None else False
            if pred is None:
                n_extract_fail += 1

            per_cat_correct[cat] = per_cat_correct.get(cat, 0) + int(ok)
            per_cat_total[cat] = per_cat_total.get(cat, 0) + 1
            per_question.append({
                "category": cat,
                "predicted": pred,
                "gold": gold,
                "correct": ok,
            })

            # Release CUDA tensors before the next 3K-token prompt. Same
            # pattern as bbh.py / gsm8k.py / tydiqa.py.
            del inputs, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 200 == 0:
                running = sum(per_cat_correct.values()) / max(1, sum(per_cat_total.values()))
                logger.info(
                    "  Progress: %d/%d | acc=%.4f", i + 1, len(test_records), running,
                )

        # NAIT reports macro avg over categories (each category weighted equally,
        # 14 categories). Micro is also surfaced for diagnostic.
        per_cat_acc = {
            c: per_cat_correct[c] / per_cat_total[c] if per_cat_total[c] else 0.0
            for c in per_cat_total
        }
        macro_avg = (
            sum(per_cat_acc.values()) / len(per_cat_acc) if per_cat_acc else 0.0
        )
        total_correct = sum(per_cat_correct.values())
        total_count = sum(per_cat_total.values())
        micro_avg = total_correct / total_count if total_count else 0.0

        summary = {
            # `accuracy` aliases the primary (macro) for score-board uniformity.
            "accuracy": macro_avg,
            "macro_avg_accuracy": macro_avg,
            "micro_avg_accuracy": micro_avg,
            "total_correct": total_correct,
            "total_questions": total_count,
            "num_categories": len(per_cat_total),
            "n_extract_fail": n_extract_fail,
            "per_category": per_cat_acc,
            "per_question": per_question,
            "benchmark": "mmlu_pro",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "MMLU-Pro macro: %.4f | micro: %.4f | n=%d | extract_fail=%d (%.1f%%)",
            macro_avg, micro_avg, total_count, n_extract_fail,
            100.0 * n_extract_fail / max(1, total_count),
        )
        return summary
