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
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tads.core.utils import load_config, quiet_repeated_warnings, setup_logger
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    quiet_repeated_warnings()

    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    logger = setup_logger(str(log_dir), name="eval")
    logger.info(
        "Eval start | ckpt=%s | base=%s",
        args.ckpt, cfg.get("model_path"),
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
    for bench in benchmarks:
        evaluator = get_evaluator(bench)
        output_file = out_dir / f"{bench}.json"
        kw: Dict[str, Any] = {}
        if bench == "lm_harness":
            kw.update(
                task=args.harness_task,
                base_model=cfg["model_path"],
                ckpt_dir=args.ckpt,
                training_mode=args.training_mode,
                lm_eval_path=args.lm_eval_path,
            )
        summary = evaluator.evaluate(
            model, tokenizer, device,
            output_file=str(output_file),
            limit=args.limit,
            prompt_style=prompt_style,
            data_dir=_data_dir_for(bench, cli_paths, cfg),
            **kw,
        )
        summaries.append(summary)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": args.ckpt,
        "base_model": cfg.get("model_path"),
        "limit": args.limit,
        "prompt_style": prompt_style,
        "summaries": summaries,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Eval done. Summary: %s", out_dir / "eval_summary.json")


if __name__ == "__main__":
    main()
