"""NAIT (ICLR 2026) baseline training.

Usage:
    python -m tads.nait.train \\
        --config configs/methods/nait.yaml \\
        --seed_path seeds/mix.json --tag NAIT-Mix
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from tads.core.schedulers import get_cosine_schedule_with_warmup
from tads.core.utils import load_config, set_seed, setup_logger
from tads.data.alpaca import build_alpaca_dataset
from tads.modeling.loader import load_model, load_tokenizer
from tads.nait.direction import (
    extract_delta_from_seed,
    fit_directions,
    score_candidates,
)
from tads.pipelines.sft import make_dataloader, sft_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed_path", required=True, help="Path to NAIT seed JSON.")
    p.add_argument("--tag", required=True, help="Variant tag, e.g. NAIT-Mix.")
    p.add_argument("--n_seeds", type=int, default=None,
                   help="Sub-sample size for seeds (default: cfg.nait.n_seeds).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"] / f"nait_{args.tag.lower().replace('-', '_')}"
    cfg["output_dir"] = str(output_dir)  # see tads.train comment re: parallel-job isolation
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(str(log_dir), name=f"nait_{args.tag}_{ts}")
    logger.info("NAIT baseline | tag=%s | seed_path=%s", args.tag, args.seed_path)

    # ---------- seeds ----------
    with open(args.seed_path) as f:
        seed_items = json.load(f)
    n_seeds = int(args.n_seeds or cfg.get("nait", {}).get("n_seeds", 100))
    if len(seed_items) > n_seeds:
        random.seed(seed)
        seed_items = random.sample(seed_items, n_seeds)
    logger.info("Loaded %d seed items", len(seed_items))

    # ---------- model / dataset ----------
    tokenizer = load_tokenizer(cfg["model_path"])
    model = load_model(
        cfg["model_path"],
        training_mode=str(cfg.get("training_mode", "full")),
        lora_cfg=cfg.get("lora"),
        use_ddp=False,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    device = next(model.parameters()).device

    dataset = build_alpaca_dataset(
        tokenizer=tokenizer,
        cache_dir=cfg["data_cache"],
        max_seq_len=int(cfg["max_seq_len"]),
        dataset_name=cfg.get("dataset_name"),
        data_files=cfg.get("data_files"),
        prompt_style=str(cfg.get("prompt_style", "alpaca_default")),
    )
    logger.info("Dataset size: %d", len(dataset))

    nait_cfg = cfg.get("nait", {}) or {}
    layers = list(nait_cfg.get("layers", [-1]))
    nait_batch_size = int(nait_cfg.get("batch_size", 2))

    # ---------- direction extraction ----------
    logger.info("Step 1: extracting delta from seeds ...")
    t0 = time.time()
    delta_per_layer = extract_delta_from_seed(
        model, tokenizer, seed_items, device, layers,
        max_seq_len=int(cfg["max_seq_len"]),
        logger=logger,
    )
    logger.info("  Δ extracted in %.1fs", time.time() - t0)

    logger.info("Step 2: fitting top-1 PCA directions ...")
    directions = fit_directions(delta_per_layer)
    for l, v in directions.items():
        logger.info("  layer=%d | norm=%.4f | dim=%d", l, v.norm().item(), v.size(0))

    # ---------- scoring + selection (resumable) ----------
    selected_path = output_dir / "selected_indices.json"
    if selected_path.exists():
        logger.info("RESUME: loading cached selection from %s", selected_path)
        with open(selected_path) as f:
            selected_indices = json.load(f)
    else:
        logger.info("Step 3: scoring candidates ...")
        t0 = time.time()
        scores = score_candidates(
            model, dataset, directions, device, nait_batch_size, logger,
        )
        logger.info("  Scoring done in %.1fs", time.time() - t0)
        k = max(1, int(len(dataset) * float(cfg["selection_ratio"])))
        selected_indices = scores.topk(k).indices.cpu().tolist()
        with open(selected_path, "w") as f:
            json.dump(selected_indices, f)
        with open(output_dir / "scores.json", "w") as f:
            json.dump(scores.tolist(), f)
        logger.info("Selected %d/%d", k, len(dataset))

    # ---------- SFT ----------
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
        logger.info("=== NAIT epoch %d/%d ===", epoch, train_epochs)
        loader = make_dataloader(subset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch)
        avg_loss = sft_one_epoch(
            model=model, loader=loader,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum=grad_accum, grad_clip=float(cfg["gradient_clip"]),
            device=device, epoch=epoch, logger=logger,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "selected": len(selected_indices),
        }
        metrics_log.append(metrics)
        logger.info("Epoch %d done | %s", epoch, metrics)

        ckpt = output_dir / f"epoch_{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        m = model.module if hasattr(model, "module") else model
        m.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

    logger.info("NAIT training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
