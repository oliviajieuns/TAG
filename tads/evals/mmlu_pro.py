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


def _format_options(options) -> Tuple[str, int]:
    """Render the option list as "A) opt1\nB) opt2\n..." and return
    ``(rendered, n_options_kept)``. Drops trailing blank entries the
    dataset uses to pad to 10.

    Tolerates: list / tuple / numpy.ndarray of strings, plus None / NaN
    (treated as empty). Parquet round-trips list columns as np.ndarray,
    and some mirrors store the answer in dict-of-strings form which we
    also accept (sorted by key).
    """
    if options is None:
        return "", 0
    # Handle dict form {"A": "...", "B": "..."} — rare but seen on some mirrors.
    if isinstance(options, dict):
        items = sorted(options.items())
        opts = [v for _, v in items]
    else:
        try:
            opts = list(options)
        except TypeError:
            return "", 0
    cleaned: List[str] = []
    for o in opts:
        if o is None:
            continue
        s = str(o)
        if not s.strip() or s.strip().lower() == "n/a":
            continue
        cleaned.append(s)
    rendered = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(cleaned))
    return rendered, len(cleaned)


def _safe_str(v, default: str = "") -> str:
    """Return v as a string, treating None/NaN/empty-array as the default.
    Avoids the numpy/pandas `arr or x` ambiguous-truth-value trap."""
    if v is None:
        return default
    # float NaN check (parquet-loaded missing values surface as NaN, not None)
    if isinstance(v, float) and v != v:
        return default
    s = str(v)
    return s if s.strip() else default


def _build_prompt(
    demo_items: List[Dict[str, Any]],
    test_q: str,
    test_opts,
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
        # CRITICAL: `d.get("options") or []` triggers the numpy
        # "truth value of an array with more than one element is ambiguous"
        # error when parquet loads list columns as np.ndarray. _format_options
        # already handles None / empty, so pass the value through directly.
        d_opts, _ = _format_options(d.get("options"))
        d_cot = _safe_str(d.get("cot_content")).strip()
        chunks.append(
            f"Question: {d.get('question', '')}\n"
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

        # ---- Schema normalisation -------------------------------------------------
        # TIGER-Lab/MMLU-Pro canonical (HF):
        #     question / options / answer / answer_index / category / cot_content / src
        # But mirrors vary: HF datasets→to_parquet exports preserve all, but some
        # older or community mirrors use `choices` / `option` / `option_a..option_j`
        # for options; `subject` / `src` for category; and omit `answer` when
        # only the index is provided. Auto-map common variants here BEFORE
        # running the schema audit so we don't reject otherwise-valid data.
        _ALIAS = {
            "options": ["options", "choices", "option", "mc_options", "answer_choices"],
            "category": ["category", "subject", "src"],
            "answer": ["answer", "answer_text"],
            "answer_index": ["answer_index", "answer_idx", "label"],
            "question": ["question", "prompt"],
            "cot_content": ["cot_content", "cot", "explanation"],
        }

        def _collect_option_a_to_j(df):
            """If options live in separate `option_a` .. `option_j` columns
            (some preprocessed mirrors), merge into a single `options` list
            column. Returns the (possibly modified) DataFrame."""
            per_letter = {}
            for L in LETTERS:
                for cand in (f"option_{L.lower()}", f"option_{L}", f"opt_{L.lower()}", f"opt_{L}"):
                    if cand in df.columns:
                        per_letter[L] = cand
                        break
            if not per_letter:
                return df
            df = df.copy()

            def _row_to_list(row):
                out = []
                for L in LETTERS:
                    col = per_letter.get(L)
                    if col is None:
                        break
                    v = row.get(col)
                    if v is None:
                        break
                    out.append(str(v))
                return out
            df["options"] = df.apply(_row_to_list, axis=1)
            logger.info(
                "MMLU-Pro: collected options from %d per-letter columns (%s).",
                len(per_letter), ",".join(sorted(per_letter.values())),
            )
            return df

        def _parse_json_options(df, name: str):
            """If `options` is stored as a JSON-encoded string ('["...", "..."]'),
            decode each row to a list. Idempotent on already-list values."""
            if "options" not in df.columns or df.empty:
                return df
            sample = df["options"].iloc[0]
            if isinstance(sample, str) and sample.strip().startswith("["):
                df = df.copy()
                def _maybe_parse(v):
                    if isinstance(v, str):
                        try:
                            return json.loads(v)
                        except Exception:
                            return [v]
                    return v
                df["options"] = df["options"].map(_maybe_parse)
                logger.info(
                    "MMLU-Pro (%s): parsed JSON-encoded `options` column to list.",
                    name,
                )
            return df

        def _normalize_schema(df, name: str):
            cols = set(df.columns)
            # First pass: collect option_a..option_j into a single `options` column.
            df = _collect_option_a_to_j(df)
            cols = set(df.columns)
            # Rename single-name aliases.
            for canon, aliases in _ALIAS.items():
                if canon in cols:
                    continue
                for a in aliases[1:]:
                    if a in cols:
                        df = df.rename(columns={a: canon})
                        logger.info(
                            "MMLU-Pro (%s): renamed column %r → %r", name, a, canon,
                        )
                        cols = set(df.columns)
                        break
            # Decode JSON-string options if needed.
            df = _parse_json_options(df, name)
            # If we have answer_index but no answer, derive the letter from index.
            if "answer" not in df.columns and "answer_index" in df.columns:
                df = df.copy()
                df["answer"] = df["answer_index"].map(
                    lambda i: LETTERS[int(i)] if 0 <= int(i) < len(LETTERS) else ""
                )
                logger.info(
                    "MMLU-Pro (%s): derived `answer` letter from `answer_index`.",
                    name,
                )
            return df

        test_df = _normalize_schema(test_df, "test")
        val_df = _normalize_schema(val_df, "validation")

        # Schema audit — fail fast on a non-MMLU-Pro mirror before running any
        # model forward. After normalisation we require the canonical names.
        need = {"question", "options", "answer", "category"}
        miss_test = need - set(test_df.columns)
        miss_val = need - set(val_df.columns)
        if miss_test or miss_val:
            # Dump observed schema + a sample row so the user can fix the data
            # source instead of guessing what we expected.
            def _row_preview(df):
                try:
                    row = df.iloc[0].to_dict()
                except Exception:
                    return "<no row 0>"
                out = {}
                for k, v in row.items():
                    if v is None:
                        out[k] = None
                    elif isinstance(v, (list, tuple)):
                        out[k] = f"<{type(v).__name__} len={len(v)}: {list(v)[:2]}...>"
                    elif hasattr(v, "__len__") and not isinstance(v, str):
                        # numpy arrays etc.
                        try:
                            preview = list(v)[:2]
                            out[k] = f"<{type(v).__name__} len={len(v)}: {preview}...>"
                        except Exception:
                            out[k] = f"<{type(v).__name__}>"
                    elif isinstance(v, str) and len(v) > 80:
                        out[k] = v[:80] + "..."
                    else:
                        out[k] = v
                return out
            raise ValueError(
                f"MMLU-Pro schema mismatch at {data_dir}.\n"
                f"  test missing: {sorted(miss_test) or 'OK'}\n"
                f"  validation missing: {sorted(miss_val) or 'OK'}\n"
                f"  expected (after alias normalisation): {sorted(need)}\n"
                f"  observed test columns:       {sorted(test_df.columns)}\n"
                f"  observed validation columns: {sorted(val_df.columns)}\n"
                f"  test row 0 sample: {_row_preview(test_df)}\n"
                f"  validation row 0 sample: {_row_preview(val_df)}\n"
                f"  Canonical source: TIGER-Lab/MMLU-Pro (HF). If you used a "
                f"different mirror, re-download with "
                f"`bash scripts/download_mmlu_pro.sh {data_dir}` or with the HF "
                f"datasets API:\n"
                f"    from datasets import load_dataset\n"
                f"    ds = load_dataset('TIGER-Lab/MMLU-Pro')\n"
                f"    ds['test'].to_parquet('{data_dir}/test-00000-of-00001.parquet')\n"
                f"    ds['validation'].to_parquet('{data_dir}/validation-00000-of-00001.parquet')",
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
            cat = _safe_str(r.get("category"), "unknown")
            demos_by_cat.setdefault(cat, []).append(r)
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
            cat = _safe_str(ex.get("category"), "unknown")
            demos = demos_by_cat.get(cat, [])
            # Pass options through as-is (None / list / ndarray all handled by
            # _format_options); avoid `list(None)` if options is missing.
            raw_opts = ex.get("options")
            if raw_opts is None:
                raw_opts = []
            elif not isinstance(raw_opts, (list, tuple)):
                try:
                    raw_opts = list(raw_opts)
                except TypeError:
                    raw_opts = [raw_opts]
            prompt, n_opts = _build_prompt(
                demos, _safe_str(ex.get("question")), raw_opts, cat,
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
            gold = _safe_str(ex.get("answer")).strip().upper()
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
