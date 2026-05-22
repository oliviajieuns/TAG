"""Data Agent baseline training entrypoint.

Per-epoch loop:
    1. Forward LLM over the WHOLE candidate pool (eval mode); collect
       state s_i = sequence-mean of last hidden layer, plus L_i and H_i.
    2. Actor.get_action(s_i) → a_i, log_prob_i (Beta sample).
    3. Compute paper-faithful R_i (min-max-normalised L/H + variance ratio).
    4. PPO update on (s, a, log_prob, R).
    5. Select top-K indices by a_i.
    6. SFT one epoch on Subset(dataset, indices).
    7. Save epoch_N/ checkpoint.

Usage:
    source scripts/setup_env.sh
    CUDA_VISIBLE_DEVICES=0 python -m baselines.data_agent.train \\
        --config configs/experiments/main_7b/llama2/data_agent_10.yaml \\
        --tag DataAgent-PPO
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Cap BLAS / OMP thread pools BEFORE `import torch` — libgomp + MKL read
# these at library init, so setting them later has zero effect. Symptom:
#   libgomp: Thread creation failed: Resource temporarily unavailable
# scripts/setup_env.sh also exports these for the source-then-launch flow.
import os as _os_for_threads
for _k, _v in (
    ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
    ("OPENBLAS_NUM_THREADS", "4"), ("NUMEXPR_NUM_THREADS", "4"),
    ("VECLIB_MAXIMUM_THREADS", "4"),
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
from torch.utils.data import Subset

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

from .agent import PPOAgent
from .select import collect_episode_and_select

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", required=True, help="Variant tag (e.g. DataAgent-PPO).")
    return p.parse_args()


def _resolve_state_dim(model) -> int:
    """Return the hidden-size of the underlying HF model."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model"):
        m = m.base_model
        if hasattr(m, "model"):
            m = m.model
    if hasattr(m, "config") and hasattr(m.config, "hidden_size"):
        return int(m.config.hidden_size)
    raise RuntimeError(
        "Could not resolve model hidden_size for the PPO actor state_dim. "
        "Pass --state_dim explicitly or fix the model wrapper.",
    )


def main() -> None:
    disable_coredumps()
    clear_runtime_caches()
    for v in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(v, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    tag_slug = args.tag.lower().replace("-", "_")
    output_dir = (
        Path(cfg["output_root"])
        / cfg["output_subdir"]
        / f"data_agent_{tag_slug}"
    )
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logger(str(log_dir), name=f"data_agent_{args.tag}_{ts}")
    log.info("Data Agent baseline | tag=%s | output_dir=%s", args.tag, output_dir)

    timer = PhaseTimer(log=log, method="data_agent")

    # ---------- model / tokenizer / dataset ----------
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
    log.info("Dataset size: %d", len(dataset))

    # ---------- PPO actor ----------
    da_cfg = cfg.get("data_agent", {}) or {}
    state_dim = _resolve_state_dim(model)
    with timer.phase("agent_init", "selection"):
            agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=int(da_cfg.get("hidden_dim", 128)),
            lr=float(da_cfg.get("actor_lr", 3e-4)),
            clip_eps=float(da_cfg.get("ppo_eps_clip", 0.2)),
            gamma=float(da_cfg.get("ppo_gamma", 0.99)),
            gae_lam=float(da_cfg.get("ppo_gae_lambda", 0.95)),
            ppo_epochs=int(da_cfg.get("ppo_k_epochs", 4)),
            entropy_coef=float(da_cfg.get("entropy_coef", 0.0)),
            value_coef=float(da_cfg.get("value_coef", 0.5)),
            mb_size=int(da_cfg.get("ppo_mb_size", 1024)),
            advantage_mode=str(da_cfg.get("advantage_mode", "group_relative")),
            value_clip=bool(da_cfg.get("value_clip", False)),
            device=str(device),
        )
    log.info(
        "PPOAgent ready | state_dim=%d | hidden=%d | k_epochs=%d | "
        "eps_clip=%.2f | adv_mode=%s",
        state_dim,
        int(da_cfg.get("hidden_dim", 128)),
        int(da_cfg.get("ppo_k_epochs", 4)),
        float(da_cfg.get("ppo_eps_clip", 0.2)),
        str(da_cfg.get("advantage_mode", "group_relative")),
    )

    # ---------- training loop ----------
    train_epochs = int(cfg["train_epochs"])
    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum"])
    episode_batch_size = int(cfg.get("episode_batch_size", 1))
    selection_ratio = float(cfg["selection_ratio"])

    metrics_log = []

    for epoch in range(1, train_epochs + 1):
        log.info("=== DataAgent epoch %d/%d ===", epoch, train_epochs)

        # 1-5. Episode → top-K (sub-phases bracketed inside select.py)
        with timer.phase(f"episode_epoch{epoch}", "selection"):
            episode = collect_episode_and_select(
                model=model,
                dataset=dataset,
                agent=agent,
                selection_ratio=selection_ratio,
                batch_size=episode_batch_size,
                device=str(device),
                epoch=epoch,
                seed=seed,
                exp_tag=str(cfg.get("model_key", "?")) + "/alpaca/data_agent",
                timer=timer,
            )
        selected_indices = episode["selected_indices"]
        with open(output_dir / f"selected_indices_epoch{epoch}.json", "w") as f:
            json.dump(selected_indices, f)

        # 6. SFT one epoch on the selected subset.
        # NOTE: optimizer/scheduler are re-built from scratch each epoch with
        # `num_training_steps = approx_steps` (i.e. one epoch). Carrying the
        # scheduler across epochs would need a total_steps estimate that
        # depends on every future selection — for a PPO baseline this is fine
        # because the per-epoch cosine restart is consistent with how other
        # baselines in this codebase handle dynamic subsets.
        with timer.phase("sft_setup", "sft"):
            subset = Subset(dataset, selected_indices)
            approx_steps = max(1, len(subset) // (batch_size * grad_accum))
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["learning_rate"]),
                weight_decay=float(cfg.get("weight_decay", 0.1)),
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(approx_steps * float(cfg["warmup_ratio"]))),
                num_training_steps=approx_steps,
            )
            loader = make_dataloader(
                subset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch,
            )
        with timer.phase(f"sft_epoch{epoch}", "sft"):
            avg_loss = sft_one_epoch(
                model=model, loader=loader,
                optimizer=optimizer, scheduler=scheduler,
                grad_accum=grad_accum, grad_clip=float(cfg["gradient_clip"]),
                device=device, epoch=epoch, logger=log,
            )

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "selected": len(selected_indices),
            "r_loss_mean": episode["r_loss_mean"],
            "r_entropy_mean": episode["r_entropy_mean"],
            "r_weight": episode["r_weight"],
            "a_mean": episode["a_mean"],
            "a_std": episode["a_std"],
            "actor_loss": episode["actor_loss"],
            "critic_loss": episode["critic_loss"],
        }
        metrics_log.append(metrics)
        log.info("Epoch %d done | %s", epoch, metrics)

        # 7. Save epoch_N/ checkpoint (baseline layout — eval reads epoch_N/).
        with timer.phase(f"checkpoint_epoch{epoch}", "checkpoint"):
            ckpt = output_dir / f"epoch_{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            m = model.module if hasattr(model, "module") else model
            m.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            agent.save(str(ckpt / "ppo_agent.pt"))
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    timer.save_report(output_dir / "timing_breakdown.json")
    timer.log_table()
    log.info("Data Agent training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
