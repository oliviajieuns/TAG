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
    """Resolve a single token id for each of "A","B","C","D".

    Tries the space-prefixed form first (matches the trailing space in the
    "Answer:" template for sentencepiece / BPE), then the bare letter, then
    finally falls back to the *first* token of either encoding. Always
    returns exactly ``len(CHOICES)`` ids so the caller can ``logits[ids]``
    safely — even on tokenisers where every encoding ends up multi-token
    (rare, but seen in heavily-merged BPE vocabularies).
    """
    ids = []
    for c in CHOICES:
        chosen = None
        for candidate in (f" {c}", c):
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if len(enc) == 1:
                chosen = enc[0]
                break
        if chosen is None:
            # Fallback: use the first token id of the bare letter encoding.
            # This is imperfect but keeps the array length consistent and
            # avoids an IndexError downstream when comparing argmax.
            enc = tokenizer.encode(c, add_special_tokens=False)
            if enc:
                chosen = enc[0]
                logger.warning(
                    "MMLU: tokenizer split %r into %d tokens; using first id (%d). "
                    "Choice-letter logits will be approximate for this model.",
                    c, len(enc), chosen,
                )
            else:
                raise RuntimeError(
                    f"MMLU: tokenizer encoded {c!r} to empty list; cannot proceed.",
                )
        ids.append(chosen)
    assert len(ids) == len(CHOICES), "choice_ids must be aligned with CHOICES"
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
        # Canonical MMLU `all` schema (HF parquet from cais/mmlu): question,
        # choices, answer, subject. Validate up-front so the eval aborts
        # cleanly if someone points data_dir at a different MMLU mirror
        # (e.g. tasksource which uses `option_a/option_b/...` instead of a
        # `choices` array). Without this check the loop would KeyError on
        # the first example, after spending minutes loading the model.
        _need_cols = {"question", "choices", "answer", "subject"}
        _missing = _need_cols - set(test_df.columns)
        if _missing:
            raise ValueError(
                f"MMLU parquet at {data_dir} has wrong schema. "
                f"Expected columns {sorted(_need_cols)} (HF canonical: "
                f"cais/mmlu `all` split), got {sorted(test_df.columns)}. "
                f"Missing: {sorted(_missing)}. This is NOT the MMLU dataset "
                f"the eval expects — point MMLU_DATA_DIR at a directory "
                f"containing the canonical parquet shards "
                f"(`test-XXXXX-of-XXXXX.parquet` + `dev-XXXXX-of-XXXXX.parquet`) "
                f"from huggingface.co/datasets/cais/mmlu (config `all`).",
            )
        # Sanity: `choices` must be a list/array of 4, `answer` an int 0..3.
        _bad_row = None
        for _i, _row in test_df.head(5).iterrows():
            _c = _row.get("choices")
            if not (hasattr(_c, "__len__") and len(_c) == 4):
                _bad_row = ("choices is not a 4-element list/array", _i, _c)
                break
            try:
                _a = int(_row.get("answer", -1))
            except (TypeError, ValueError):
                _bad_row = ("answer is not int-coercible", _i, _row.get("answer"))
                break
            if _a < 0 or _a > 3:
                _bad_row = (f"answer={_a} out of range [0..3]", _i, _a)
                break
        if _bad_row is not None:
            reason, _i, val = _bad_row
            raise ValueError(
                f"MMLU parquet at {data_dir} has wrong values: {reason} "
                f"(row {_i}, sample={val!r}). Expected `choices` as a 4-element "
                f"list of strings and `answer` as int 0..3. The schema doesn't "
                f"match the canonical cais/mmlu `all` split — refusing to run.",
            )
        # The dev split is conventionally a single shard
        # (``dev-00000-of-00001.parquet``) but cluster mirrors and re-shardings
        # can change that suffix. Glob defensively so the eval doesn't crash
        # on a different shard count.
        dev_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("dev-") and f.endswith(".parquet")
        )
        if not dev_files:
            raise FileNotFoundError(
                f"No `dev-*.parquet` files found in {data_dir}. "
                "Expected the MMLU `all` directory.",
            )
        dev_df = pd.concat(
            [pd.read_parquet(os.path.join(data_dir, f)) for f in dev_files],
            ignore_index=True,
        )

        subjects = test_df["subject"].unique()
        logger.info("MMLU: %d subjects | limit=%s", len(subjects), limit)

        results = []
        # Track truncation once per run (don't spam): a 2048-token cap can
        # decapitate the prompt prefix when a Qwen ChatML 5-shot template is
        # verbose, and a chopped few-shot is silently downgraded to (1..4)-shot.
        _trunc_warned = False
        _trunc_count = 0
        for subject in subjects:
            dev_examples = dev_df[dev_df["subject"] == subject].to_dict("records")[:num_fewshot]
            test_examples = test_df[test_df["subject"] == subject].to_dict("records")
            if limit is not None:
                test_examples = test_examples[:limit]

            correct, total = 0, 0
            for ex in test_examples:
                prompt = _build_few_shot_prompt(dev_examples, ex, subject)
                # Tokenise once without truncation to measure the real length;
                # only the truncated version is fed to the model.
                full_ids = tokenizer(prompt, return_tensors="pt").input_ids
                if full_ids.shape[1] > 2048:
                    _trunc_count += 1
                    if not _trunc_warned:
                        logger.warning(
                            "MMLU: prompt length %d > max_length 2048 — "
                            "the 5-shot prefix is being clipped on the LEFT, "
                            "which silently degrades few-shot quality. "
                            "Subject=%s. (Further occurrences counted but not logged.)",
                            full_ids.shape[1], subject,
                        )
                        _trunc_warned = True
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
        if _trunc_count > 0:
            logger.warning(
                "MMLU: %d / %d test examples had their 5-shot prefix "
                "truncated. Score may be lower than the paper's number.",
                _trunc_count, total_total,
            )
        summary = {
            "overall_accuracy": overall,
            "total_correct": total_correct,
            "total_questions": total_total,
            "num_subjects": len(results),
            "truncated_prompts": _trunc_count,
            "per_subject": results,
            "benchmark": "mmlu",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("MMLU overall: %.4f", overall)
        return summary
