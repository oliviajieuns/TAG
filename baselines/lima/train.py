"""LIMA (Zhou et al., 2023) baseline training — pure data-replacement SFT.

No selection algorithm: LIMA is the 1030-sample dataset itself. We load it
and run the shared SFT loop unchanged.

Usage:
    source scripts/setup_env.sh
    # First time only (gated HF dataset):
    #   huggingface-cli login
    #   visit https://huggingface.co/datasets/GAIR/lima and accept terms
    # Or download to disk and set LIMA_DATA_FILES.
    CUDA_VISIBLE_DEVICES=0 python -m baselines.lima.train \\
        --config configs/experiments/main_7b/llama2/lima.yaml \\
        --tag LIMA

Eval (separate root):
    python -m tads.eval --config <same cfg> \\
        --ckpt ${OUTPUT_ROOT}/main_7b/llama2/lima/lima_lima/epoch_3 \\
        --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \\
        --out_dir ${LIMA_EVAL_RESULTS_ROOT}/llama2/lima/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Cap BLAS / OMP thread pools BEFORE `import torch` — libgomp + MKL read
# these at library init, so setting them later has zero effect. Symptom:
#   libgomp: Thread creation failed: Resource temporarily unavailable
# scripts/setup_env.sh also exports these for the source-then-launch flow.
import os as _os_for_threads
for _k, _v in (
    ("OMP_NUM_THREADS", "16"), ("MKL_NUM_THREADS", "16"),
    ("OPENBLAS_NUM_THREADS", "16"), ("NUMEXPR_NUM_THREADS", "16"),
    ("VECLIB_MAXIMUM_THREADS", "16"),
):
    _os_for_threads.environ.setdefault(_k, _v)

# transformers 5.0 video-registry workaround — see tads.train.main.
try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    pass

import torch

from tads.core.schedulers import get_cosine_schedule_with_warmup
from tads.core.timing import PhaseTimer
from tads.core.utils import (
    clear_runtime_caches,
    disable_coredumps,
    load_config,
    set_seed,
    setup_logger,
)
from tads.modeling.loader import load_model, load_tokenizer
from tads.pipelines.sft import make_dataloader, sft_one_epoch

from .data import build_lima_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="LIMA", help="Variant tag (e.g. LIMA).")
    p.add_argument(
        "--data_files",
        default=None,
        help=(
            "Path / glob to LIMA train.jsonl (or .json / .parquet). Overrides "
            "$LIMA_DATA_FILES and cfg.lima.data_files. Use this when the env "
            "var doesn't survive into the python process (nohup / new tmux "
            "session / cron / etc.)."
        ),
    )
    return p.parse_args()


def main() -> None:
    disable_coredumps()
    clear_runtime_caches()
    for v in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    tag_slug = args.tag.lower().replace("-", "_")
    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"] / f"lima_{tag_slug}"
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(log_dir), name=f"lima_{tag_slug}_{ts}")
    logger.info("LIMA baseline | tag=%s | output_dir=%s", args.tag, output_dir)

    timer = PhaseTimer(log=logger, method="lima")

    with timer.phase("tokenizer_load", "setup"):
        tokenizer = load_tokenizer(cfg["model_path"])
    with timer.phase("model_load", "setup"):
        model = load_model(
            cfg["model_path"],
            training_mode=str(cfg.get("training_mode", "full")),
            lora_cfg=cfg.get("lora"),
            use_ddp=False,
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    device = next(model.parameters()).device

    # Resolution: --data_files CLI > $LIMA_DATA_FILES env > cfg.lima.data_files
    # > HF hub fallback. Empty string is treated as None so an `export
    # LIMA_DATA_FILES=` (or accidentally unquoted shell var) doesn't silently
    # route us into the offline-fail branch.
    _cli_val = args.data_files
    _env_val = os.environ.get("LIMA_DATA_FILES") or None
    _cfg_val = (cfg.get("lima") or {}).get("data_files") or None
    data_files = _cli_val or _env_val or _cfg_val
    logger.info(
        "LIMA data_files resolution | cli=%r | env=%r | cfg=%r | resolved=%r",
        _cli_val, _env_val, _cfg_val, data_files,
    )
    with timer.phase("dataset_build", "data"):
        dataset = build_lima_dataset(
            tokenizer=tokenizer,
            cache_dir=cfg["data_cache"],
            max_seq_len=int(cfg["max_seq_len"]),
            data_files=data_files,
            prompt_style=str(cfg.get("prompt_style") or "alpaca_default"),
        )
    logger.info("LIMA dataset size: %d", len(dataset))

    train_epochs = int(cfg["train_epochs"])
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum"])
    approx_steps = max(1, len(dataset) // (batch_size * grad_accum))
    total_steps = approx_steps * train_epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * float(cfg["warmup_ratio"]))),
        num_training_steps=total_steps,
    )

    metrics_log = []
    for epoch in range(1, train_epochs + 1):
        logger.info("=== LIMA epoch %d/%d ===", epoch, train_epochs)
        t0 = time.time()
        with timer.phase(f"sft_epoch{epoch}", "sft"):
            loader = make_dataloader(
                dataset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch,
            )
            avg_loss = sft_one_epoch(
                model=model, loader=loader,
                optimizer=optimizer, scheduler=scheduler,
                grad_accum=grad_accum, grad_clip=float(cfg["gradient_clip"]),
                device=device, epoch=epoch, logger=logger,
            )
        metrics_log.append({
            "epoch": epoch, "train_loss": avg_loss, "n": len(dataset),
            "wall_sec": round(time.time() - t0, 2),
        })
        with timer.phase(f"checkpoint_epoch{epoch}", "checkpoint"):
            ckpt = output_dir / f"epoch_{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, "module") else model
            m.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    timer.save_report(output_dir / "timing_breakdown.json")
    timer.log_table()
    logger.info("LIMA training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
