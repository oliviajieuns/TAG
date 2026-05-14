"""Unified training entrypoint.

Usage:
    python -m tads.train --config configs/experiments/light_tads_05b.yaml
    torchrun --nproc_per_node=4 -m tads.train \\
        --config configs/experiments/7b_fullft_tads_50.yaml

The method (random/full/data_agent/tads) is selected by the ``method`` key
inside the YAML config.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Subset
from tads.core.schedulers import get_cosine_schedule_with_warmup

from tads.core.agent import PPOAgent
from tads.core.trajectory_anchor import TrajectoryAnchor
from tads.core.utils import (
    cuda_mem_str,
    is_main_process,
    load_config,
    local_rank,
    quiet_repeated_warnings,
    set_seed,
    setup_logger,
    world_size,
)
from tads.data.alpaca import build_alpaca_dataset
from tads.modeling.loader import get_hidden_size, load_model, load_tokenizer
from tads.pipelines.selection import save_selection, select_indices
from tads.pipelines.sft import make_dataloader, sft_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Top-level overrides, e.g. selection_ratio=0.3 train_epochs=1",
    )
    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides) -> None:
    """Apply ``key=value`` overrides (top-level keys only). Values are
    parsed as float/int/bool when possible, else kept as string."""
    for kv in overrides:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        # naive type coercion
        if v.lower() in {"true", "false"}:
            cfg[k] = v.lower() == "true"
        else:
            try:
                cfg[k] = int(v)
            except ValueError:
                try:
                    cfg[k] = float(v)
                except ValueError:
                    cfg[k] = v


def _setup_ddp() -> bool:
    """Initialise torch.distributed if launched under torchrun."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank())
        return True
    return dist.is_initialized()


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Silence HF tokenizer's per-call "Token indices sequence length..."
    # advisory; it fires on every batch when any sample is longer than
    # max_seq_len, even though we truncate intentionally.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    quiet_repeated_warnings()

    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.override)

    use_ddp = _setup_ddp()
    method = str(cfg["method"])
    seed = int(cfg["seed"])
    set_seed(seed)

    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"]
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    logger = setup_logger(str(log_dir), name=f"train_{method}")
    if is_main_process():
        logger.info("=" * 60)
        logger.info(
            "TADS unified trainer | method=%s | ddp=%s | world_size=%d",
            method, use_ddp, world_size(),
        )
        logger.info("Config:\n%s", json.dumps(cfg, indent=2, default=str))
        logger.info("=" * 60)

    # ---------- tokenizer / model ----------
    tokenizer = load_tokenizer(cfg["model_path"])

    training_mode = str(cfg.get("training_mode", "full"))
    model = load_model(
        cfg["model_path"],
        training_mode=training_mode,
        lora_cfg=cfg.get("lora"),
        use_ddp=use_ddp,
        local_rank=local_rank(),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        attn_implementation=cfg.get("attn_implementation"),
    )
    device = (
        torch.device(f"cuda:{local_rank()}") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    hidden_size = get_hidden_size(model)
    if is_main_process():
        logger.info("hidden_size=%d | device=%s", hidden_size, device)

    # ---------- dataset ----------
    # Isolate the HF datasets cache per (model, prompt_style) so multiple
    # concurrent training jobs (e.g. qwen + mistral + deepseek launched in
    # parallel) don't race on the same fingerprint / lock files. The base
    # path comes from cfg["data_cache"]; we append the model_key and
    # prompt_style so tokenisation caches stay distinct even if two configs
    # share the same data_cache root.
    model_key = str(cfg.get("model_key", "default"))
    style_key = str(cfg.get("prompt_style", "alpaca_default"))
    effective_cache = os.path.join(
        str(cfg["data_cache"]), model_key, style_key,
    )
    if is_main_process():
        logger.info("HF datasets cache: %s", effective_cache)
    dataset = build_alpaca_dataset(
        tokenizer=tokenizer,
        cache_dir=effective_cache,
        max_seq_len=int(cfg["max_seq_len"]),
        dataset_name=cfg.get("dataset_name"),
        data_files=cfg.get("data_files"),
        prompt_style=style_key,
    )
    n_total_full = len(dataset)

    sub_n = cfg.get("dataset_subset_size")
    if sub_n is not None and int(sub_n) < n_total_full:
        g = torch.Generator(); g.manual_seed(seed)
        keep = torch.randperm(n_total_full, generator=g).tolist()[: int(sub_n)]
        dataset = dataset.select(keep)
        if is_main_process():
            logger.info("Sub-sampled dataset to %d (seed=%d)", len(dataset), seed)
    n_total = len(dataset)

    # ---------- method-specific setup ----------
    selection_ratio = float(cfg["selection_ratio"])
    train_epochs = int(cfg["train_epochs"])
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum"])
    lr = float(cfg["learning_rate"])
    warmup_ratio = float(cfg["warmup_ratio"])
    grad_clip = float(cfg["gradient_clip"])

    agent: Optional[PPOAgent] = None
    anchor: Optional[TrajectoryAnchor] = None

    if method in ("tads", "data_agent") and is_main_process():
        agent_cfg = cfg.get("agent", {}) or {}
        agent = PPOAgent(
            state_dim=hidden_size,
            lr=float(agent_cfg.get("lr", 3e-4)),
            clip_eps=float(agent_cfg.get("clip_eps", 0.2)),
            gamma=float(agent_cfg.get("gamma", 0.99)),
            gae_lam=float(agent_cfg.get("gae_lam", 0.95)),
            ppo_epochs=int(agent_cfg.get("ppo_epochs", 4)),
            entropy_coef=float(agent_cfg.get("entropy_coef", 0.01)),
            value_coef=float(agent_cfg.get("value_coef", 0.5)),
            mb_size=int(agent_cfg.get("mb_size", 1024)),
            advantage_mode=str(agent_cfg.get("advantage_mode", "group_relative")),
            value_clip=bool(agent_cfg.get("value_clip", True)),
            device=str(device),
        )

    if method == "tads" and is_main_process():
        anchor_cfg = cfg.get("anchor", {}) or {}
        anchor = TrajectoryAnchor(
            layer_idx=int(anchor_cfg.get("layer_idx", -1)),
            layer_indices=anchor_cfg.get("layer_indices"),
            max_samples_for_pca=int(anchor_cfg.get("max_samples_for_pca", 2000)),
            pca_batch_size=int(anchor_cfg.get("pca_batch_size", 4)),
            device=str(device),
        )

    # ---------- optimizer / scheduler ----------
    approx_steps_per_epoch = max(
        1, int(n_total * selection_ratio / batch_size / grad_accum / max(1, world_size())),
    )
    total_steps = approx_steps_per_epoch * train_epochs
    # 8-bit AdamW cuts optimizer state from ~56GB to ~14GB per GPU on 7B
    # full fine-tuning. Matches the NAIT paper's recipe (bnb.optim.AdamW8bit).
    wd = float(cfg.get("weight_decay", 0.1))
    if bool(cfg.get("use_8bit_optimizer", False)):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=lr, weight_decay=wd,
        )
        if is_main_process():
            logger.info("Optimizer: bitsandbytes.AdamW8bit | wd=%s", wd)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd,
        )
        if is_main_process():
            logger.info("Optimizer: torch.AdamW (fp32) | wd=%s", wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * warmup_ratio)),
        num_training_steps=total_steps,
    )

    # ---------- training loop ----------
    metrics_log = []
    for epoch in range(1, train_epochs + 1):
        if is_main_process():
            logger.info("=" * 60)
            logger.info("Epoch %d / %d | method=%s", epoch, train_epochs, method)
            logger.info("=" * 60)
        t0 = time.time()

        selected, extras = select_indices(
            method,
            model=model,
            agent=agent,
            anchor=anchor,
            dataset=dataset,
            cfg=cfg,
            epoch=epoch,
            seed=seed,
            device=device,
        )
        save_selection(output_dir, epoch, selected)

        subset = Subset(dataset, selected)
        loader = make_dataloader(
            subset, batch_size=batch_size, shuffle=True, seed=seed,
        )
        avg_loss = sft_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum=grad_accum,
            grad_clip=grad_clip,
            device=device,
            epoch=epoch,
            logger=logger,
        )
        elapsed = time.time() - t0
        metrics = {
            "epoch": epoch,
            "method": method,
            "selected_n": len(selected),
            "n_total": n_total,
            "train_loss": avg_loss,
            "elapsed_sec": elapsed,
            **extras,
        }
        if is_main_process():
            metrics_log.append(metrics)
            logger.info("Epoch %d done | %s", epoch, metrics)

            ckpt_path = output_dir / f"epoch_{epoch}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, "module") else model
            m.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            if agent is not None:
                agent.save(str(ckpt_path / "agent.pt"))
            if anchor is not None:
                torch.save(anchor.state_dict(), str(ckpt_path / "trajectory_anchor.pt"))
                with open(ckpt_path / "anchor_history.json", "w") as f:
                    json.dump(anchor.get_history_summary(), f, indent=2)
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)
            logger.info("Checkpoint saved: %s", ckpt_path)

        if use_ddp:
            dist.barrier()

    if is_main_process():
        logger.info("Training complete (%d epochs).", train_epochs)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
