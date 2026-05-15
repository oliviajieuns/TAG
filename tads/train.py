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
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# transformers 5.0 eager-imports `from torchvision.io import VideoReader`
# via its video model registry, which fails on torchvision builds without
# ffmpeg support — even though our LLM-only training never touches video.
# Stub the missing attribute BEFORE any transformers import so the import
# resolves to a harmless placeholder. Eval entrypoint is intentionally
# left untouched (user request: training-only mitigation).
try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    # torchvision absent entirely is fine — our code never uses it.
    pass

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


def _atomic_json_dump(obj, path: Path) -> None:
    """Atomically write JSON via tmp+fsync+rename so a crash mid-write can't
    leave a half-written file that the next run silently misparses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    """Apply ``key=value`` overrides; supports dotted nested keys.

    Examples:
        selection_ratio=0.3
        anchor.layer_idx=-1
        agent.lr=5.0e-5
        tads.lam=2.0

    Values are parsed as bool/int/float when possible, else kept as string.
    """
    def _coerce(v: str):
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    for kv in overrides:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        coerced = _coerce(v)
        if "." not in k:
            cfg[k] = coerced
            continue
        # Nested: walk down (creating intermediate dicts on the fly), then
        # set the leaf. Refuse to overwrite a non-dict intermediate to avoid
        # silently shadowing a scalar with a dict.
        parts = k.split(".")
        node = cfg
        for part in parts[:-1]:
            existing = node.get(part)
            if existing is None:
                node[part] = {}
            elif not isinstance(existing, dict):
                raise ValueError(
                    f"--override {k}={v}: cannot descend into non-dict key "
                    f"{part!r} (current value: {existing!r})",
                )
            node = node[part]
        node[parts[-1]] = coerced


def _setup_ddp() -> bool:
    """Initialise torch.distributed if launched under torchrun.

    NCCL timeout is bumped from PyTorch's 10-min default to 120 min: rank 0
    runs collect_episode (full forward over the ~52K candidate pool) solo
    while the other ranks idle at the next barrier. For Llama-2-7B at
    episode_batch_size=16 that pass takes 30–90 min; with the default
    timeout the idle ranks would trip a Watchdog collective-timeout error
    mid-selection and crash the job.
    """
    if "RANK" in os.environ and not dist.is_initialized():
        from datetime import timedelta
        # Backend selectable via TADS_DDP_BACKEND env. NCCL is fast but
        # has the idle-then-hang failure mode under rank-0-solo workloads
        # like collect_episode. gloo is ~5x slower but TCP-based and far
        # more robust; use it as an escape hatch when NCCL won't behave.
        #     export TADS_DDP_BACKEND=gloo
        backend = os.environ.get("TADS_DDP_BACKEND", "nccl").lower()
        if backend not in ("nccl", "gloo"):
            print(f"[ddp] unknown TADS_DDP_BACKEND={backend!r}, falling back to nccl",
                  flush=True)
            backend = "nccl"
        print(f"[ddp] initialising process group | backend={backend}", flush=True)
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=120),
        )
        torch.cuda.set_device(local_rank())
        return True
    return dist.is_initialized()


def _reinit_ddp_after_long_idle(model, use_ddp: bool):
    """Destroy and recreate the NCCL process group + re-wrap model in DDP.

    Empirically, after rank 0 spends 30+ minutes in collect_episode while
    other ranks sit in file-polling (no NCCL traffic), the NCCL communicator
    enters a state where the next collective hangs — even with our
    120-minute init timeout. The diagnostic is "SFT step entry step=0
    appears on every rank but no rank reaches step backward done", meaning
    forward worked but the first all_reduce inside backward stalls.

    Recovery: tear the process group down and bring it back up before SFT
    starts. The DDP wrapper holds a reference to the old (dead) group, so
    we also have to unwrap the model and re-wrap it after the reinit.
    """
    if not dist.is_initialized() or not use_ddp:
        return model
    from datetime import timedelta
    inner = model.module if hasattr(model, "module") else model
    backend = dist.get_backend()
    print(f"[ddp-reinit] rank={dist.get_rank()} destroying old group", flush=True)
    dist.destroy_process_group()
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=120))
    print(f"[ddp-reinit] rank={dist.get_rank()} fresh group up", flush=True)
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    # Match the original wrap. find_unused_parameters mirrors load_model.
    find_unused = any(
        not p.requires_grad for p in inner.parameters()
    )  # heuristic: LoRA has frozen base
    return torch.nn.parallel.DistributedDataParallel(
        inner,
        device_ids=[lr],
        output_device=lr,
        find_unused_parameters=find_unused,
    )


def main() -> None:
    # OFFLINE BY DEFAULT — every model / tokenizer / dataset must be on local
    # disk. The HF datasets / hub / transformers libraries otherwise reach
    # over the network even when the data file is local (metadata refresh,
    # version pings, dataset-card lookup), and on cluster nodes without
    # outbound HTTPS that triggers a flaky "tries to download → cache lock
    # corruption" failure mode. Users who explicitly want the hub fallback
    # can override any of these to "0" before launching.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Silence HF tokenizer's per-call "Token indices sequence length..."
    # advisory; it fires on every batch when any sample is longer than
    # max_seq_len, even though we truncate intentionally.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # NCCL idle protection. The internal watchdog default (600 s) trips
    # 3x during a 30-minute rank-0 solo collect_episode and silently
    # marks the communicator dead — the next SFT all_reduce then hangs
    # forever. Raise the heartbeat ceiling so the communicator survives
    # the long single-rank phase. Also disable async error handling so
    # any real NCCL failure raises immediately instead of hanging.
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "99999")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "0")

    quiet_repeated_warnings()

    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.override)

    use_ddp = _setup_ddp()
    method = str(cfg["method"])
    seed = int(cfg["seed"])
    set_seed(seed)

    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"]
    # Expose the per-experiment output_dir to downstream modules — notably
    # pipelines.selection._broadcast_selection, which writes a temporary
    # selection-share file. Without this, multiple parallel jobs (qwen +
    # llama2 + mistral + deepseek launched concurrently via run_main_7b.sh)
    # would all fall back to ``cfg["output_root"]`` and clobber each
    # other's _selection_epoch{N}.json — silently mixing their selected
    # indices across experiments.
    cfg["output_dir"] = str(output_dir)
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

    # ---------- resume: find latest epoch checkpoint, if any ----------
    def _find_latest_epoch_ckpt(out_dir: Path):
        """Return (epoch_num, path) of latest epoch_N/ under out_dir, or (0, None).

        A checkpoint is considered "complete" only if the ``_complete`` sentinel
        file exists in it. The save sequence (model, tokenizer, optimizer,
        scheduler, agent, anchor, metrics) writes this sentinel as the very
        last step. Partial checkpoints (process killed mid-save) lack the
        sentinel and are skipped — preventing the next run from auto-resuming
        a corrupted optimizer.pt and tripping a 1EB OOM on garbage tensor
        metadata.
        """
        if not out_dir.exists():
            return 0, None
        epochs = []
        for p in out_dir.glob("epoch_*"):
            if not p.is_dir():
                continue
            try:
                n = int(p.name.replace("epoch_", ""))
            except ValueError:
                continue
            if not (p / "_complete").exists():
                continue
            # Backstop: the model file must also be present.
            if (p / "config.json").exists() or (p / "adapter_config.json").exists():
                epochs.append((n, p))
        if not epochs:
            return 0, None
        epochs.sort()
        return epochs[-1]

    resume_epoch, resume_ckpt = _find_latest_epoch_ckpt(output_dir)
    if resume_ckpt is not None and is_main_process():
        logger.info(
            "RESUMING from %s (epoch %d completed; continuing at epoch %d)",
            resume_ckpt, resume_epoch, resume_epoch + 1,
        )
    if use_ddp:
        dist.barrier()

    # ---------- tokenizer / model ----------
    # Load tokenizer from base path (does not change between epochs).
    tokenizer = load_tokenizer(cfg["model_path"])

    # If resuming, load model weights from the checkpoint dir; else from base.
    model_load_path = str(resume_ckpt) if resume_ckpt is not None else cfg["model_path"]
    if is_main_process():
        logger.info("Loading model from: %s", model_load_path)

    training_mode = str(cfg.get("training_mode", "full"))
    model = load_model(
        model_load_path,
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

    # ---------- resume: restore optimizer/scheduler/agent/anchor/metrics ----------
    metrics_log = []
    if resume_ckpt is not None:
        # bitsandbytes 8-bit optimizer state is bnb-version-coupled; mismatched
        # restore silently leaves momentum at 0. Surface a precise warning so
        # the user can pin the version instead of debugging "why is loss
        # plateauing right after resume".
        env_meta_path = resume_ckpt / "env_meta.json"
        if env_meta_path.exists() and is_main_process():
            try:
                with open(env_meta_path) as _f:
                    saved_meta = json.load(_f)
                if saved_meta.get("use_8bit_optimizer"):
                    saved_bnb = saved_meta.get("bitsandbytes")
                    try:
                        import bitsandbytes as _bnb_now  # noqa: WPS433
                        live_bnb = _bnb_now.__version__
                    except Exception:
                        live_bnb = None
                    if saved_bnb is not None and live_bnb != saved_bnb:
                        logger.warning(
                            "bitsandbytes version mismatch on resume: "
                            "saved=%s, live=%s. AdamW8bit state may fail to "
                            "deserialise (the catch below will fall back to "
                            "fresh momentum). Pin %s to keep continuity.",
                            saved_bnb, live_bnb, saved_bnb,
                        )
            except Exception as e:
                logger.warning("Could not read env_meta.json (%s)", e)

        opt_path = resume_ckpt / "optimizer.pt"
        if opt_path.exists():
            try:
                # map_location="cpu": optimizer state for 7B full-FT is ~14 GB;
                # the live optimizer that was just constructed already holds
                # GPU memory for its (empty) state. Loading directly to GPU
                # would peak at 2× (28 GB) before the old state is freed and
                # OOM the rank. Loading to CPU and letting load_state_dict
                # move tensors per-param keeps the peak at ~14 GB.
                # weights_only=False is required because optimizer state
                # (especially bnb.AdamW8bit) contains non-tensor pickled
                # quantisation metadata; it's also future-proofs for
                # PyTorch 2.6+ where weights_only defaults to True.
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location="cpu", weights_only=False),
                )
                if is_main_process():
                    logger.info("Restored optimizer state from %s", opt_path)
            except Exception as e:
                if is_main_process():
                    logger.warning("Could not restore optimizer (%s); using fresh state", e)
        sch_path = resume_ckpt / "scheduler.pt"
        if sch_path.exists():
            try:
                scheduler.load_state_dict(
                    torch.load(sch_path, map_location="cpu", weights_only=False),
                )
                if is_main_process():
                    logger.info("Restored scheduler state from %s", sch_path)
            except Exception as e:
                if is_main_process():
                    logger.warning("Could not restore scheduler (%s); using fresh state", e)
        if agent is not None:
            agent_path = resume_ckpt / "agent.pt"
            if agent_path.exists() and hasattr(agent, "load"):
                try:
                    agent.load(str(agent_path))
                    if is_main_process():
                        logger.info("Restored PPO agent state from %s", agent_path)
                except Exception as e:
                    if is_main_process():
                        logger.warning("Could not restore agent (%s)", e)
        if anchor is not None:
            anchor_path = resume_ckpt / "trajectory_anchor.pt"
            if anchor_path.exists():
                try:
                    anchor.load_state_dict(
                        torch.load(anchor_path, map_location="cpu", weights_only=False),
                    )
                    if is_main_process():
                        logger.info("Restored trajectory anchor from %s", anchor_path)
                except Exception as e:
                    if is_main_process():
                        logger.warning("Could not restore anchor (%s)", e)
        metrics_json = output_dir / "metrics.json"
        if metrics_json.exists():
            try:
                with open(metrics_json) as f:
                    metrics_log = json.load(f)
                if is_main_process():
                    logger.info("Restored %d epoch metric rows", len(metrics_log))
            except Exception as e:
                if is_main_process():
                    logger.warning("Could not restore metrics.json (%s)", e)

    start_epoch = resume_epoch + 1
    if start_epoch > train_epochs:
        if is_main_process():
            logger.info(
                "All %d epochs already completed (resume_epoch=%d). Nothing to do.",
                train_epochs, resume_epoch,
            )
        if use_ddp:
            dist.destroy_process_group()
        return

    # ---------- training loop ----------
    for epoch in range(start_epoch, train_epochs + 1):
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

        if len(selected) == 0:
            raise RuntimeError(
                f"Epoch {epoch}: selected indices is empty. SFT would produce "
                "0 batches and DDP all_reduce at end of empty loop is a known "
                "hang source. Check selection_ratio and dataset size.",
            )

        # NCCL reinit workaround: OPT-IN via TADS_NCCL_REINIT=1.
        # Empirically `destroy_process_group` itself can hang when the
        # NCCL communicator is in a deeply broken state — making this
        # "recovery" path itself a deadlock. Default is OFF; rely on
        # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=99999 + async-error env vars
        # to prevent the communicator from dying in the first place.
        if (
            method in ("tads", "data_agent")
            and use_ddp
            and os.environ.get("TADS_NCCL_REINIT", "0") == "1"
        ):
            model = _reinit_ddp_after_long_idle(model, use_ddp)

        subset = Subset(dataset, selected)
        loader = make_dataloader(
            subset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch,
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
        # ---------- per-epoch checkpoint save (rank 0 only) ----------
        # Bug history: training with tads/data_agent under DDP was crashing
        # immediately after epoch 1 with no checkpoint on disk. Two failure
        # modes are mitigated below:
        #   (a) rank 0 OOMs / errors during a single save step (e.g.
        #       torch.save(optimizer.state_dict()) for bnb 8-bit) and the
        #       whole process exits, leaving workers stuck on the post-save
        #       barrier until NCCL timeout — no partial state is recorded.
        #   (b) memory accumulated during collect_episode + SFT pushes rank 0
        #       to the edge; the additional CPU buffer that save_pretrained
        #       allocates tips it over.
        # Mitigations: free CPU+GPU memory before the save sequence, wrap
        # every save step in its own try/except so one failure doesn't lose
        # everything, surface tracebacks so the user can diagnose, and ALWAYS
        # reach the post-save barrier so workers don't hang past the
        # collective timeout.
        if is_main_process():
            metrics_log.append(metrics)
            logger.info("Epoch %d done | %s", epoch, metrics)

            # Drop transient tensors before allocating the save buffers.
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("Pre-save memory cleanup | %s", cuda_mem_str())
            except Exception as e:  # never block save on a cleanup hiccup
                logger.warning("Pre-save cleanup failed (continuing): %s", e)

            ckpt_path = output_dir / f"epoch_{epoch}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            save_errors: list = []

            def _safe(step_name: str, fn) -> bool:
                """Run a save step; on exception record it and keep going."""
                try:
                    t = time.time()
                    fn()
                    logger.info("Saved %s | %.1fs", step_name, time.time() - t)
                    return True
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error(
                        "Save step '%s' FAILED: %s\n%s", step_name, exc, tb,
                    )
                    save_errors.append((step_name, repr(exc)))
                    return False

            m = model.module if hasattr(model, "module") else model
            ok_model = _safe("model.safetensors",
                             lambda: m.save_pretrained(str(ckpt_path)))
            _safe("tokenizer",
                  lambda: tokenizer.save_pretrained(str(ckpt_path)))
            ok_opt = _safe("optimizer.pt",
                           lambda: torch.save(optimizer.state_dict(),
                                              str(ckpt_path / "optimizer.pt")))
            _safe("scheduler.pt",
                  lambda: torch.save(scheduler.state_dict(),
                                     str(ckpt_path / "scheduler.pt")))

            # env_meta — bnb version etc.
            env_meta: Dict[str, Any] = {
                "torch": torch.__version__,
                "use_8bit_optimizer": bool(cfg.get("use_8bit_optimizer", False)),
            }
            try:
                import bitsandbytes as _bnb  # noqa: WPS433 (lazy)
                env_meta["bitsandbytes"] = _bnb.__version__
            except Exception:
                env_meta["bitsandbytes"] = None
            _safe("env_meta.json",
                  lambda: _atomic_json_dump(env_meta, ckpt_path / "env_meta.json"))

            if agent is not None:
                _safe("agent.pt",
                      lambda: agent.save(str(ckpt_path / "agent.pt")))
            if anchor is not None:
                _safe("trajectory_anchor.pt",
                      lambda: torch.save(anchor.state_dict(),
                                         str(ckpt_path / "trajectory_anchor.pt")))
                _safe("anchor_history.json",
                      lambda: _atomic_json_dump(
                          anchor.get_history_summary(),
                          ckpt_path / "anchor_history.json"))
            _safe("metrics.json",
                  lambda: _atomic_json_dump(metrics_log,
                                            output_dir / "metrics.json"))

            # Sentinel: ONLY written when the two state files that auto-resume
            # depends on (model weights + optimizer) both succeeded. A failed
            # auxiliary save (anchor history etc.) is non-fatal — log it,
            # carry on. A failed core save → no sentinel → resume skips this
            # epoch and re-trains it next run.
            if ok_model and ok_opt:
                sentinel = ckpt_path / "_complete"
                sentinel_tmp = ckpt_path / "_complete.tmp"
                try:
                    with open(sentinel_tmp, "w") as f:
                        f.write(str(epoch))
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(sentinel_tmp, sentinel)
                    logger.info("Checkpoint saved + sealed: %s", ckpt_path)
                except Exception as exc:
                    logger.error("Sentinel write FAILED: %s", exc)
                    save_errors.append(("_complete", repr(exc)))
            else:
                logger.error(
                    "Core save failed (model_ok=%s, optim_ok=%s) — sentinel "
                    "not written; epoch %d will be redone on resume.",
                    ok_model, ok_opt, epoch,
                )

            if save_errors:
                # Drop a sidecar so the diagnostic survives the next epoch.
                err_path = ckpt_path / "_save_errors.json"
                try:
                    _atomic_json_dump(
                        {"epoch": epoch, "errors": save_errors},
                        err_path,
                    )
                except Exception:
                    pass

            # Optional: keep only last K checkpoints to avoid disk bloat.
            _keep = int(cfg.get("keep_last_n_checkpoints", 0))
            if _keep > 0:
                existing = sorted(
                    [p for p in output_dir.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.replace("epoch_", "")),
                )
                for old in existing[:-_keep]:
                    import shutil as _shutil
                    _shutil.rmtree(old, ignore_errors=True)
                    logger.info("Removed old checkpoint: %s", old)

        # Workers MUST hit the barrier even if rank 0 raised inside the save
        # block — otherwise NCCL stalls until its timeout and rank 0's exit
        # code is masked by a generic collective failure on every worker.
        if use_ddp:
            dist.barrier()

    if is_main_process():
        logger.info("Training complete (%d epochs).", train_epochs)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
