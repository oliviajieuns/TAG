"""AlpaGasus (Chen et al., 2024) baseline training.

Uses the official pre-filtered `chatgpt_9k.json` from
github.com/gpt4life/alpagasus (rating threshold 4.5 by GPT-4 judge). No
OpenAI API key required — we just match the filtered instruction strings
against our local Alpaca-GPT4 records to recover the selected indices.

Pipeline:
    1. Load full Alpaca-GPT4 (HF Dataset, tokenised) — same as SelectIT.
    2. Load AlpaGasus filtered JSON (filename is whatever the user
       downloaded — `chatgpt_9k.json`, `claude_t45.json`, etc.).
    3. Build an instruction → index map over the full Alpaca-GPT4 set,
       then look up each filtered instruction.
    4. SFT on the matched subset.

Usage:
    # One-time setup — download official filtered JSON to disk:
    #   wget https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered/chatgpt_9k.json
    # then:
    source scripts/setup_env.sh
    export ALPAGASUS_FILTERED_FILE=/path/to/chatgpt_9k.json
    CUDA_VISIBLE_DEVICES=0 python -m baselines.alpagasus.train \\
        --config configs/experiments/main_7b/llama2/alpagasus.yaml \\
        --tag AlpaGasus-ChatGPT-9k
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    pass

import torch
from torch.utils.data import Subset

from tads.core.data_io import read_records, read_records_glob
from tads.core.schedulers import get_cosine_schedule_with_warmup
from tads.core.timing import PhaseTimer
from tads.core.utils import (
    clear_runtime_caches,
    disable_coredumps,
    load_config,
    set_seed,
    setup_logger,
)
from tads.data.alpaca import build_alpaca_dataset
from tads.modeling.loader import load_model, load_tokenizer
from tads.pipelines.sft import make_dataloader, sft_one_epoch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--filtered_file",
        default=None,
        help=(
            "Path to AlpaGasus pre-filtered JSON (e.g. chatgpt_9k.json). "
            "Default: $ALPAGASUS_FILTERED_FILE or cfg.alpagasus.filtered_file."
        ),
    )
    p.add_argument("--tag", default="AlpaGasus", help="Variant tag.")
    return p.parse_args()


def _load_alpaca_raw(data_files_spec: str) -> list:
    """Load Alpaca-GPT4 raw records via the shared JSON/JSONL/Parquet helper."""
    return read_records_glob(data_files_spec)


def _extract_instruction(rec) -> str:
    """Pull the canonical instruction string from a record (Alpaca or AlpaGasus shape)."""
    if "instruction" in rec:
        ins = rec["instruction"]
        inp = rec.get("input") or ""
        return f"{ins}\n{inp}".strip() if inp else ins.strip()
    if "conversations" in rec and isinstance(rec["conversations"], list):
        c = rec["conversations"]
        v = c[0] if isinstance(c[0], str) else c[0].get("value", "")
        return v.strip()
    raise KeyError(f"No instruction field in record; keys={list(rec.keys())}")


def _match_indices(filtered_records: list, full_records: list) -> list:
    """Return indices in `full_records` matching any record in `filtered_records`,
    keyed on instruction text. Logs unmatched count."""
    full_idx = {}
    for i, rec in enumerate(full_records):
        try:
            key = _extract_instruction(rec)
        except KeyError:
            continue
        # Last-write wins on duplicate instructions; Alpaca-GPT4 has very few.
        full_idx.setdefault(key, i)

    matched: list = []
    miss = 0
    for f in filtered_records:
        try:
            key = _extract_instruction(f)
        except KeyError:
            miss += 1
            continue
        if key in full_idx:
            matched.append(full_idx[key])
        else:
            miss += 1
    if miss:
        logger.warning(
            "AlpaGasus: %d/%d filtered records didn't match an Alpaca-GPT4 row "
            "(instruction wording diff between Alpaca and Alpaca-GPT4 mirrors).",
            miss, len(filtered_records),
        )
    # Deduplicate while preserving order.
    seen: set = set()
    deduped: list = []
    for i in matched:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


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
    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"] / f"alpagasus_{tag_slug}"
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(log_dir), name=f"alpagasus_{tag_slug}_{ts}")
    logger.info("AlpaGasus baseline | tag=%s | output_dir=%s", args.tag, output_dir)

    timer = PhaseTimer(log=logger, method="alpagasus")

    # Resolve filtered file path: CLI > env > cfg.
    _cli_val = args.filtered_file
    _env_val = os.environ.get("ALPAGASUS_FILTERED_FILE")
    _cfg_val = (cfg.get("alpagasus") or {}).get("filtered_file")
    filtered_file = _cli_val or _env_val or _cfg_val
    if not filtered_file or not Path(filtered_file).exists():
        # Show exactly which of the three sources had what — usually pinpoints
        # the missed step (no source, wrong path, dataset on different mount).
        raise FileNotFoundError(
            "AlpaGasus: filtered JSON not found.\n"
            f"  args.filtered_file (CLI)            = {_cli_val!r}\n"
            f"  ALPAGASUS_FILTERED_FILE (env var)   = {_env_val!r}\n"
            f"  cfg.alpagasus.filtered_file (yaml)  = {_cfg_val!r}\n"
            f"  resolved (first non-empty)          = {filtered_file!r}\n"
            f"  exists on disk                      = {bool(filtered_file) and Path(filtered_file).exists()}\n"
            "Fix one of:\n"
            "  1) `bash scripts/download_alpagasus.sh` (puts chatgpt_9k.json at\n"
            "     the path setup_env.sh's ALPAGASUS_FILTERED_FILE points to)\n"
            "  2) `--filtered_file /path/to/chatgpt_9k.json` directly on the\n"
            "     training command\n"
            "  3) `export ALPAGASUS_FILTERED_FILE=...` BEFORE launching python\n"
            "     in the same shell session"
        )

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

    with timer.phase("dataset_build", "data"):
        dataset = build_alpaca_dataset(
            tokenizer=tokenizer,
            cache_dir=cfg["data_cache"],
            max_seq_len=int(cfg["max_seq_len"]),
            dataset_name=cfg.get("dataset_name"),
            data_files=cfg.get("data_files"),
            prompt_style=str(cfg.get("prompt_style") or "alpaca_default"),
        )
    with timer.phase("raw_records_load", "data"):
        raw_records = _load_alpaca_raw(str(cfg["data_files"]))
    if len(raw_records) != len(dataset):
        raise RuntimeError(
            f"AlpaGasus: raw ({len(raw_records)}) vs tokenised dataset "
            f"({len(dataset)}) length mismatch — index alignment broken."
        )
    logger.info("Alpaca-GPT4 size: %d", len(dataset))

    # AlpaGasus filtered JSON ships as JSON-list in the official repo, but
    # some forks redistribute as JSONL — go through the shared helper so we
    # never re-introduce the "Extra data: line 2 column 1" bug.
    with timer.phase("alpagasus.load_filtered", "selection"):
        filtered = read_records(filtered_file)
    logger.info("AlpaGasus filtered records: %d (from %s)", len(filtered), filtered_file)

    with timer.phase("alpagasus.match_indices", "selection"):
        selected_indices = _match_indices(filtered, raw_records)
    if not selected_indices:
        raise RuntimeError(
            "AlpaGasus: zero indices matched. Filtered JSON's instruction text "
            "appears to come from a different Alpaca mirror than our local "
            "Alpaca-GPT4. Confirm instruction strings overlap."
        )
    logger.info(
        "AlpaGasus selected %d / %d Alpaca-GPT4 rows (proportion=%.3f)",
        len(selected_indices), len(dataset),
        len(selected_indices) / max(1, len(dataset)),
    )
    with open(output_dir / "selected_indices.json", "w") as f:
        json.dump(selected_indices, f)

    subset = Subset(dataset, selected_indices)
    train_epochs = int(cfg["train_epochs"])
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum"])
    approx_steps = max(1, len(subset) // (batch_size * grad_accum))
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
        logger.info("=== AlpaGasus epoch %d/%d ===", epoch, train_epochs)
        t0 = time.time()
        with timer.phase(f"sft_epoch{epoch}", "sft"):
            loader = make_dataloader(
                subset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch,
            )
            avg_loss = sft_one_epoch(
                model=model, loader=loader,
                optimizer=optimizer, scheduler=scheduler,
                grad_accum=grad_accum, grad_clip=float(cfg["gradient_clip"]),
                device=device, epoch=epoch, logger=logger,
            )
        metrics_log.append({
            "epoch": epoch, "train_loss": avg_loss,
            "selected": len(selected_indices),
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
    logger.info("AlpaGasus training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
