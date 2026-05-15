"""Evaluation entrypoint.

Usage:
    python -m tads.eval \\
        --config configs/experiments/light_tads_05b.yaml \\
        --ckpt /path/to/epoch_3 \\
        --benchmarks mmlu,gsm8k,humaneval,tydiqa \\
        --out_dir results/light_tads_05b/

The benchmark list is split by commas; each name is looked up in the
:mod:`tads.evals` registry. Data paths for each benchmark come from the
config (``<benchmark>_data_dir`` keys) or from ``--<benchmark>_data_dir``
on the CLI.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tads.core.utils import (
    clear_runtime_caches,
    disable_coredumps,
    load_config,
    quiet_repeated_warnings,
    setup_logger,
)
from tads.evals import get_evaluator, list_evaluators
from tads.modeling.loader import load_for_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config (model_path, prompt_style).")
    p.add_argument("--ckpt", required=True, help="Checkpoint directory (LoRA adapter or full).")
    p.add_argument(
        "--benchmarks", default="mmlu",
        help=f"Comma-separated. Registered: {list_evaluators()}",
    )
    p.add_argument("--out_dir", required=True, help="Output directory for JSON summaries.")
    p.add_argument("--limit", type=int, default=None, help="Per-benchmark sample cap.")
    p.add_argument(
        "--training_mode", default=None, choices=[None, "full", "lora"],
        help="Override checkpoint type detection.",
    )
    p.add_argument(
        "--cuda_device", type=int, default=0,
        help="CUDA device index for evaluation.",
    )
    # Per-benchmark data dir overrides.
    p.add_argument("--mmlu_data_dir", default=None)
    p.add_argument("--gsm8k_data_dir", default=None)
    p.add_argument("--humaneval_data_dir", default=None)
    p.add_argument("--tydiqa_data_dir", default=None)
    p.add_argument("--bbh_data_dir", default=None,
                   help="BBH root containing per-task .json + cot-prompts/.")
    # lm_harness extras.
    p.add_argument("--harness_task", default="mmlu", help="lm_harness `task` kwarg.")
    p.add_argument("--lm_eval_path", default=None, help="PYTHONPATH addition for lm-eval-harness fork.")
    return p.parse_args()


def _data_dir_for(
    name: str,
    cli: Dict[str, Optional[str]],
    cfg: Dict[str, Any],
) -> Optional[str]:
    return cli.get(f"{name}_data_dir") or cfg.get(f"{name}_data_dir")


def main() -> None:
    # Cap RLIMIT_CORE on this process and forks — see tads.train.main for
    # rationale. Eval rarely segfaults but if it does (CUDA OOM, model
    # load mismatch) the dump is still ~250 GB worth of bf16 weights.
    disable_coredumps()

    # Clean GC / CUDA allocator / IPC-handle state before loading the
    # model. Eval routinely re-runs against the same checkpoint set
    # (auto_eval_7b_fullft.sh polls in a loop), and stale handles from
    # a crashed prior iteration can otherwise pin VRAM that the new
    # load_for_eval can't allocate.
    clear_runtime_caches()

    # OFFLINE BY DEFAULT — see tads.train.main for rationale.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    quiet_repeated_warnings()

    # Eval is designed to run on a single GPU. If invoked under torchrun
    # (which sets RANK / LOCAL_RANK / WORLD_SIZE), only the rank-0 process
    # should run the evaluation — otherwise every worker re-runs the full
    # benchmark and they collide on output files. Workers exit cleanly.
    #
    # The previous predicate `int(os.environ.get("RANK","0")) != 0` was
    # too aggressive: a stale RANK left in the shell (from a prior torchrun
    # session, or a SLURM env that aliases SLURM_PROCID → RANK) would
    # silently return without ever parsing args, giving the appearance
    # of "eval terminates immediately" with no log line.
    #
    # New gate: require BOTH a real torchrun signature (WORLD_SIZE > 1
    # AND LOCAL_RANK present) AND RANK != 0 before exiting. A noisy
    # log line on entry makes "silent immediate exit" impossible going
    # forward — if you see no log at all, the process is being killed
    # by something external (oom-killer, SIGTERM, etc.).
    _rank_env = os.environ.get("RANK")
    _local_rank_env = os.environ.get("LOCAL_RANK")
    _world_size_env = os.environ.get("WORLD_SIZE")
    print(
        f"[eval] entry | pid={os.getpid()} | RANK={_rank_env!r} | "
        f"LOCAL_RANK={_local_rank_env!r} | WORLD_SIZE={_world_size_env!r}",
        flush=True,
    )
    _is_torchrun_child = (
        _local_rank_env is not None
        and int(_world_size_env or "1") > 1
    )
    if _is_torchrun_child and int(_rank_env or "0") != 0:
        print(
            f"[eval] non-rank-0 torchrun worker (rank={_rank_env}) — exiting cleanly",
            flush=True,
        )
        return

    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    logger = setup_logger(str(log_dir), name="eval")

    # Output files used to be plain `mmlu.json` / `eval_summary.json` —
    # opaque once you copied a few of them into a shared results folder
    # and couldn't tell `eval_summary.json` from `tads_10`'s vs
    # `data_agent_10`'s. Prefix every artifact with an experiment label
    # so a flat directory listing tells you which (model, method) the
    # numbers belong to. Source of truth, in order:
    #   1. cfg["experiment_name"]  — explicit override in YAML
    #   2. <config-parent-dir>_<config-stem>  — e.g.
    #      configs/experiments/main_7b/llama2/tads_10.yaml
    #      → "llama2_tads_10"
    #   3. <config-stem>  — last-resort for ad-hoc configs.
    _cfg_path = Path(args.config)
    _parent = _cfg_path.parent.name
    if cfg.get("experiment_name"):
        experiment_label = str(cfg["experiment_name"])
    elif _parent and _parent not in ("configs", ".", ""):
        experiment_label = f"{_parent}_{_cfg_path.stem}"
    else:
        experiment_label = _cfg_path.stem

    logger.info(
        "Eval start | exp=%s | ckpt=%s | base=%s",
        experiment_label, args.ckpt, cfg.get("model_path"),
    )

    benchmarks: List[str] = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmarks:
        raise SystemExit(
            f"--benchmarks must list at least one benchmark. "
            f"Available: {list_evaluators()}"
        )
    unknown = [b for b in benchmarks if b not in list_evaluators()]
    if unknown:
        raise SystemExit(f"Unknown benchmarks: {unknown}. Available: {list_evaluators()}")

    model, tokenizer, device = load_for_eval(
        base_model=cfg["model_path"],
        ckpt_dir=args.ckpt,
        training_mode=args.training_mode,
        device=f"cuda:{args.cuda_device}",
    )

    prompt_style = cfg.get("prompt_style", "alpaca_default")
    cli_paths = {
        "mmlu_data_dir": args.mmlu_data_dir,
        "gsm8k_data_dir": args.gsm8k_data_dir,
        "humaneval_data_dir": args.humaneval_data_dir,
        "bbh_data_dir": args.bbh_data_dir,
        "tydiqa_data_dir": args.tydiqa_data_dir,
    }

    summaries = []
    failures: List[Dict[str, str]] = []
    for bench in benchmarks:
        evaluator = get_evaluator(bench)
        output_file = out_dir / f"{experiment_label}-{bench}.json"
        kw: Dict[str, Any] = {}
        if bench == "lm_harness":
            kw.update(
                task=args.harness_task,
                base_model=cfg["model_path"],
                ckpt_dir=args.ckpt,
                training_mode=args.training_mode,
                lm_eval_path=args.lm_eval_path,
            )
        # Per-benchmark try/except: a single benchmark failure (missing data
        # dir, OOM during generation, corrupted parquet) used to abort the
        # entire eval and lose results for the other 3-4 benchmarks that
        # already finished. Capture the error and keep going so partial
        # metrics still land in eval_summary.json.
        try:
            summary = evaluator.evaluate(
                model, tokenizer, device,
                output_file=str(output_file),
                limit=args.limit,
                prompt_style=prompt_style,
                data_dir=_data_dir_for(bench, cli_paths, cfg),
                **kw,
            )
            summaries.append(summary)
        except Exception as e:
            logger.exception("Benchmark %s failed; continuing with remaining benchmarks", bench)
            failures.append({"benchmark": bench, "error": f"{type(e).__name__}: {e}"})
        finally:
            # Free transformer KV caches / generate buffers between
            # benchmarks. Without this the high-water mark accumulates
            # across the 5-bench sequence (HumanEval's n_samples=20 +
            # BBH's 3072-token prompts especially) and a 7B run that
            # would steady-state at ~30 GB peaks past 80 GB by the time
            # we hit the last benchmark.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "experiment": experiment_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt,
        "base_model": cfg.get("model_path"),
        "limit": args.limit,
        "prompt_style": prompt_style,
        "summaries": summaries,
        "failures": failures,
    }
    summary_path = out_dir / f"{experiment_label}-eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    if failures:
        logger.warning(
            "Eval finished with %d/%d benchmark failure(s): %s",
            len(failures), len(benchmarks), [f["benchmark"] for f in failures],
        )
    logger.info("Eval done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
