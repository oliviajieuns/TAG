"""lm-evaluation-harness wrapper.

Spawns a child ``python -m lm_eval`` process for one task at a time. Useful
for large-scale evaluation (MMLU full, BBH, TruthfulQA, etc.) where the
official harness is preferred over our minimal implementations.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BenchmarkEvaluator, register

logger = logging.getLogger(__name__)


@register("lm_harness")
class LMHarnessEvaluator(BenchmarkEvaluator):
    """Run a single lm-evaluation-harness task.

    Pass the harness task name via the ``task`` kwarg (or default to
    ``mmlu``). Requires the ``lm-eval`` package installed.
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
        task: str = "mmlu",
        num_fewshot: Optional[int] = None,
        batch_size: str = "auto",
        base_model: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        training_mode: Optional[str] = None,
        lm_eval_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if ckpt_dir is None or base_model is None:
            raise ValueError(
                "lm_harness evaluator needs `ckpt_dir` and `base_model` "
                "(passed via tads.eval CLI)."
            )

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = out_path.with_suffix(".log")

        if training_mode is None:
            training_mode = (
                "lora" if (Path(ckpt_dir) / "adapter_config.json").exists() else "full"
            )

        model_args = (
            f"pretrained={ckpt_dir},dtype=bfloat16,trust_remote_code=True"
            if training_mode == "full"
            else f"pretrained={base_model},peft={ckpt_dir},dtype=bfloat16,trust_remote_code=True"
        )

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task,
            "--batch_size", str(batch_size),
            "--output_path", str(out_path),
        ]
        if limit is not None:
            cmd += ["--limit", str(int(limit))]
        if num_fewshot is not None:
            cmd += ["--num_fewshot", str(int(num_fewshot))]

        env = os.environ.copy()
        if lm_eval_path:
            env["PYTHONPATH"] = (
                f"{lm_eval_path}:{env.get('PYTHONPATH', '')}".rstrip(":")
            )
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        logger.info("lm_harness | task=%s | cmd=%s", task, " ".join(cmd))
        t0 = time.time()
        with open(log_path, "w") as f_log:
            proc = subprocess.run(
                cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT, check=False,
            )
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"failed(rc={proc.returncode})"

        summary = {
            "task": task,
            "status": status,
            "elapsed_sec": elapsed,
            "output_path": str(out_path),
            "log_path": str(log_path),
            "benchmark": f"lm_harness:{task}",
        }
        if proc.returncode == 0 and out_path.exists():
            try:
                with open(out_path) as f:
                    data = json.load(f)
                summary["results"] = data.get("results", {})
            except Exception as e:
                logger.warning("Could not parse harness output: %s", e)
        return summary
