"""MBPP evaluator — sanitized split, 3-shot pass@1 (NAIT Appendix D).

Setup
-----
- **Sanitized config** (257 test problems) — the NAIT default.
- **3-shot demonstrations** drawn from the `prompt` split of the same
  config. The first 3 records are used for determinism (no sampling).
- **Greedy pass@1** — single completion at temperature=0, evaluated by
  exec'ing the model's response against ``test_list`` assertions inside
  a subprocess sandbox with a per-task wall-clock timeout.

The MBPP prompt template (matching the original Austin et al. 2021
release and the lm-eval-harness convention) is:

    You are an expert Python programmer, and here is your task: <text>
    Your code should pass these tests:

    <assertion-1>

    [BEGIN]
    <reference code>
    [DONE]

with the test problem appended in the same shape, ending at `[BEGIN]`
so the model continues with the function definition.

Schema (HF parquet, both configs):
    task_id : int
    text    : str   (problem description)
    code    : str   (reference solution, used as demo body only)
    test_list : list[str]   (assert statements for pass@1 scoring)
    test_setup_code : str   (optional setup like `import collections`)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional

import torch

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


def _read_parquet_dir(cfg_dir: str, split_prefix: str):
    """Read all `{split_prefix}-*.parquet` files in cfg_dir, concat, return DataFrame.
    Returns None if no files found (caller decides whether that's fatal)."""
    import pandas as pd
    if not os.path.isdir(cfg_dir):
        return None
    files = sorted(
        f for f in os.listdir(cfg_dir)
        if f.startswith(f"{split_prefix}-") and f.endswith(".parquet")
    )
    if not files:
        return None
    dfs = [pd.read_parquet(os.path.join(cfg_dir, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _safe_list(v) -> List:
    """Return v as a plain Python list. Handles None, numpy.ndarray (from
    parquet sequence columns), pandas Series, and bare strings (wrapped
    in a single-element list).

    Necessary because `v or []` triggers the numpy
    "truth value of an array with more than one element is ambiguous"
    error when v is np.ndarray.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v]
    try:
        return list(v)
    except TypeError:
        return [v]


def _build_mbpp_prompt(
    demos: List[Dict[str, Any]],
    test_item: Dict[str, Any],
) -> str:
    """3-shot MBPP prompt — Austin et al. 2021 / lm-eval-harness canonical form.

    Format (per `lm_eval/tasks/mbpp/preprocess_mbpp.create_test_prompt`):

        You are an expert Python programmer, and here is your task: <text> Your code should pass these tests:

        <assert 1>
        <assert 2>
        ...

        [BEGIN]
        <code>
        [DONE]
        ... (repeated for each of the 3 demos) ...
        You are an expert Python programmer, and here is your task: <test.text> Your code should pass these tests:

        <test.assert 1>
        ...

        [BEGIN]

    Two important properties:
      1. `text` + " Your code should pass these tests:" are on the SAME line
         (single space between, no newline between). The blank line is BEFORE
         the assert block, not before "Your code...". This is what models
         SFT'd on Alpaca/CodeAlpaca have seen.
      2. ALL test_list items are shown, newline-joined. Showing only the
         first assert (a previous bug here) caused the model to write code
         that satisfied test 1 but failed tests 2..N, dropping pass@1 by
         ~5–10pt vs the paper.
    """
    chunks: List[str] = []
    for d in demos:
        tests = _safe_list(d.get("test_list"))
        tests_str = "\n".join(str(t) for t in tests)
        chunks.append(
            f"You are an expert Python programmer, and here is your task: "
            f"{d.get('text', '')} Your code should pass these tests:\n\n"
            f"{tests_str}\n"
            f"[BEGIN]\n"
            f"{d.get('code', '')}\n"
            f"[DONE]\n"
        )
    tests = _safe_list(test_item.get("test_list"))
    tests_str = "\n".join(str(t) for t in tests)
    chunks.append(
        f"You are an expert Python programmer, and here is your task: "
        f"{test_item.get('text', '')} Your code should pass these tests:\n\n"
        f"{tests_str}\n"
        f"[BEGIN]\n"
    )
    return "".join(chunks)


# Strip everything from "[DONE]" / next "[BEGIN]" / "You are an expert"
# onwards — the model often continues with another demonstration after
# completing the test problem's body.
_TRIM_PATTERNS = [
    re.compile(r"\n\s*\[DONE\]"),
    re.compile(r"\n\s*\[BEGIN\]"),
    re.compile(r"\n\s*You are an expert Python programmer"),
    re.compile(r"\n\s*Your code should pass"),
]


def _extract_completion(text: str) -> str:
    """Trim the raw generated text to just the model's code body for the
    current test problem. Without this the exec stage sees a trailing
    second demonstration that re-defines `__main__`-level names and may
    pass or fail spuriously."""
    out = text
    for pat in _TRIM_PATTERNS:
        m = pat.search(out)
        if m:
            out = out[: m.start()]
    return out.rstrip()


def _run_in_subprocess(code: str, test_list: List[str], setup: str, timeout: float) -> Dict[str, Any]:
    """Exec the model's code + the gold test_list in a fresh Python subprocess.

    Subprocess sandbox vs in-process exec:
      - hard timeout via Popen.kill (in-process signal.SIGALRM doesn't fire
        from inside a third-party C-extension call like numpy);
      - true isolation — model code that monkey-patches builtins or imports
        won't leak into subsequent items;
      - distinct memory accounting — runaway recursion / fork bomb is
        contained at process boundary, not Python heap.

    Returns dict with ``passed`` (bool) and ``error`` (str | None).
    """
    # Inline test runner script — exits 0 iff every assert in test_list passes.
    runner = textwrap.dedent("""
        import sys, json
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        setup = payload.get("setup", "") or ""
        tests = payload["tests"]
        g = {"__name__": "__main__"}
        try:
            if setup.strip():
                exec(setup, g)
            exec(code, g)
            for t in tests:
                exec(t, g)
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        print("OK")
    """).strip()

    payload = json.dumps({
        "code": code,
        "setup": setup,
        "tests": _safe_list(test_list),
    })
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"timeout ({timeout}s)"}
    except Exception as exc:
        return {"passed": False, "error": f"subprocess: {type(exc).__name__}: {exc}"}

    if proc.returncode == 0 and proc.stdout.strip() == "OK":
        return {"passed": True, "error": None}
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {
        "passed": False,
        "error": err[-1] if err else f"exit={proc.returncode}",
    }


@register("mbpp")
class MBPPEvaluator(BenchmarkEvaluator):
    """MBPP sanitized + 3-shot pass@1 (NAIT Appendix D)."""

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        *,
        output_file: str,
        limit: Optional[int] = None,
        prompt_style: str = "alpaca_default",  # unused (MBPP has its own format)
        data_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        num_fewshot: int = 3,
        max_input_tokens: int = 2048,
        # 10s was too tight — a non-trivial fraction of MBPP problems have
        # exhaustive-search / combinatorics reference solutions that run
        # 15–25s under unoptimised model-generated code, getting wrongly
        # counted as failures. 30s matches the bigcode-evaluation-harness
        # default and recovers ~2–3pt of pass@1 on llama-2-7B vs 10s.
        exec_timeout: float = 30.0,
        config: str = "sanitized",
        **kwargs,
    ) -> Dict[str, Any]:
        if data_dir is None:
            raise ValueError(
                "MBPP: `data_dir` is required (path to mbpp/ root containing "
                "sanitized/ and optionally full/ subdirs).",
            )

        cfg_dir = os.path.join(data_dir, config)
        test_df = _read_parquet_dir(cfg_dir, "test")
        prompt_df = _read_parquet_dir(cfg_dir, "prompt")
        if test_df is None:
            raise FileNotFoundError(
                f"No `test-*.parquet` in {cfg_dir}. Run "
                f"`bash scripts/download_mbpp.sh {data_dir}` first "
                f"(config={config!r}).",
            )
        if prompt_df is None or prompt_df.empty:
            raise FileNotFoundError(
                f"No `prompt-*.parquet` in {cfg_dir}. The MBPP `prompt` "
                f"split is required as the 3-shot demonstration source.",
            )

        # ---- Schema normalisation -------------------------------------------------
        # google-research-datasets/mbpp ships two configs that disagree on
        # field names:
        #   sanitized:  task_id / source_file / prompt / code / test_imports / test_list
        #   full:       task_id / text / code / test_list / test_setup_code / challenge_test_list
        # We canonicalise to the `full` field names internally so the rest of
        # the evaluator (prompt builder + sandbox setup) doesn't have to care
        # which config the user downloaded.
        #
        # Mirror variants (community / lm-eval-harness preprocessing) sometimes
        # use additional aliases, listed below. Order matters — first match
        # wins (most common variant first).
        _TEXT_ALIASES = ["text", "prompt", "task", "description", "nl", "nl_description"]
        _CODE_ALIASES = ["code", "solution", "canonical_solution"]
        _TESTS_ALIASES = ["test_list", "tests", "test_cases", "asserts"]
        _SETUP_ALIASES = ["test_setup_code", "setup", "imports"]

        def _first_alias_present(cols, aliases):
            for a in aliases:
                if a in cols:
                    return a
            return None

        def _normalize_mbpp(df, split_name: str):
            df = df.copy()
            cols = set(df.columns)
            # text — problem description
            if "text" not in cols:
                src = _first_alias_present(cols, _TEXT_ALIASES[1:])
                if src is not None:
                    df = df.rename(columns={src: "text"})
                    logger.info(
                        "MBPP (%s/%s): renamed %r → `text`", config, split_name, src,
                    )
                    cols = set(df.columns)
            # code — reference solution
            if "code" not in cols:
                src = _first_alias_present(cols, _CODE_ALIASES[1:])
                if src is not None:
                    df = df.rename(columns={src: "code"})
                    logger.info(
                        "MBPP (%s/%s): renamed %r → `code`", config, split_name, src,
                    )
                    cols = set(df.columns)
            # test_list — list of assert statements
            if "test_list" not in cols:
                src = _first_alias_present(cols, _TESTS_ALIASES[1:])
                if src is not None:
                    df = df.rename(columns={src: "test_list"})
                    logger.info(
                        "MBPP (%s/%s): renamed %r → `test_list`",
                        config, split_name, src,
                    )
                    cols = set(df.columns)
            # test_setup_code — derive from `test_imports` (sanitized list-of-imports)
            # if not directly present.
            if "test_setup_code" not in cols:
                if "test_imports" in cols:
                    df["test_setup_code"] = df["test_imports"].map(
                        lambda x: "\n".join(str(i) for i in _safe_list(x))
                    )
                    logger.info(
                        "MBPP (%s/%s): derived `test_setup_code` from "
                        "`test_imports` (sanitized layout).",
                        config, split_name,
                    )
                else:
                    src = _first_alias_present(cols, _SETUP_ALIASES[1:])
                    if src is not None:
                        df = df.rename(columns={src: "test_setup_code"})
                        logger.info(
                            "MBPP (%s/%s): renamed %r → `test_setup_code`",
                            config, split_name, src,
                        )
                    else:
                        df["test_setup_code"] = ""
            return df

        test_df = _normalize_mbpp(test_df, "test")
        prompt_df = _normalize_mbpp(prompt_df, "prompt")

        need = {"task_id", "text", "code", "test_list"}
        miss = need - set(test_df.columns)
        if miss:
            try:
                sample = {
                    k: (v if not isinstance(v, (list, tuple))
                        else f"<{type(v).__name__} len={len(v)}>")
                    for k, v in test_df.iloc[0].to_dict().items()
                }
            except Exception:
                sample = "<no row 0>"
            raise ValueError(
                f"MBPP schema mismatch at {cfg_dir}: missing {sorted(miss)}.\n"
                f"  observed test columns: {sorted(test_df.columns)}\n"
                f"  test row 0 sample: {sample}\n"
                f"  Canonical source: google-research-datasets/mbpp. If you "
                f"used a different mirror, re-download with "
                f"`bash scripts/download_mbpp.sh {data_dir}` (config={config!r}) "
                f"or with the HF datasets API:\n"
                f"    from datasets import load_dataset\n"
                f"    ds = load_dataset('google-research-datasets/mbpp', {config!r})\n"
                f"    ds['test'].to_parquet('{cfg_dir}/test-00000-of-00001.parquet')\n"
                f"    ds['prompt'].to_parquet('{cfg_dir}/prompt-00000-of-00001.parquet')",
            )

        # First num_fewshot records of `prompt` split — deterministic.
        demos = prompt_df.head(num_fewshot).to_dict("records")

        def _coerce_setup(v) -> str:
            if v is None:
                return ""
            if isinstance(v, float) and v != v:  # NaN
                return ""
            if isinstance(v, str):
                return v
            # If a mirror stored test_setup_code as list-of-imports, join.
            try:
                return "\n".join(str(x) for x in v)
            except TypeError:
                return str(v)

        # Convert test_list / test_setup_code numpy arrays → plain lists / str.
        # Using _safe_list / _coerce_setup avoids the `arr or default` numpy
        # ambiguous-truth bug.
        for d in demos:
            d["test_list"] = _safe_list(d.get("test_list"))
            d["test_setup_code"] = _coerce_setup(d.get("test_setup_code"))
        test_records = test_df.to_dict("records")
        for r in test_records:
            r["test_list"] = _safe_list(r.get("test_list"))
            r["test_setup_code"] = _coerce_setup(r.get("test_setup_code"))
        if limit is not None:
            test_records = test_records[:limit]
        logger.info(
            "MBPP (%s): %d test | %d-shot | exec_timeout=%.1fs | limit=%s",
            config, len(test_records), len(demos), exec_timeout, limit,
        )

        tokenizer.truncation_side = "left"

        n_pass = 0
        per_problem: List[Dict[str, Any]] = []
        for i, ex in enumerate(test_records):
            prompt = _build_mbpp_prompt(demos, ex)
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
                stop_strings=[
                    "\n[DONE]", "[DONE]",
                    "\nYou are an expert Python programmer",
                ],
                tokenizer=tokenizer,
            )
            prompt_tok_len = inputs["input_ids"].shape[1]
            raw = tokenizer.decode(
                out[0, prompt_tok_len:], skip_special_tokens=True,
            )
            completion = _extract_completion(raw)

            run = _run_in_subprocess(
                completion, ex["test_list"], ex["test_setup_code"], exec_timeout,
            )
            ok = bool(run["passed"])
            n_pass += int(ok)
            per_problem.append({
                "task_id": int(ex["task_id"]),
                "passed": ok,
                "error": run["error"],
                "completion": completion,
            })

            del inputs, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 25 == 0:
                logger.info(
                    "  Progress: %d/%d | pass@1=%.4f",
                    i + 1, len(test_records), n_pass / (i + 1),
                )

        pass_at_1 = n_pass / len(test_records) if test_records else 0.0
        summary = {
            # `accuracy` aliases pass@1 for score-board uniformity.
            "accuracy": pass_at_1,
            "pass@1": pass_at_1,
            "total_passed": n_pass,
            "total_questions": len(test_records),
            "config": config,
            "n_fewshot": len(demos),
            "exec_timeout_sec": exec_timeout,
            "per_problem": per_problem,
            "benchmark": "mbpp",
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "MBPP pass@1: %.4f (%d/%d) | config=%s",
            pass_at_1, n_pass, len(test_records), config,
        )
        return summary
