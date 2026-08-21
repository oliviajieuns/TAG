"""Unified training entrypoint.

Usage:
    python -m tag.train --config configs/experiments/light_legacy_05b.yaml
    torchrun --nproc_per_node=4 -m tag.train \\
        --config configs/experiments/7b_fullft_legacy_50.yaml

The method (random/full/selection) is selected by the ``method`` key inside
the YAML config. Comparison baselines (data_agent / nait / selectit /
lima / alpagasus / q2q) live under ``baselines.<method>.train`` and
have their own entrypoints — ``tag.train`` rejects them with an
actionable redirect error.

Run layout (history-preserving)
-------------------------------
Each invocation writes its checkpoints under

    <output_dir>/runs/<run_tag>/

so re-running with tweaked hyperparameters never overwrites a prior run.
The ``_latest`` symlink under ``<output_dir>/`` tracks the most recent
sealed epoch and is what ``tag.eval`` reads by default.

    # Fresh run with auto-timestamped tag
    torchrun -m tag.train --config <cfg>
        # → <output_dir>/runs/20260515_230514/

    # Tagged run (great for hyperparameter sweeps)
    torchrun -m tag.train --config <cfg> --run_suffix=lr2e5
        # → <output_dir>/runs/20260515_230514_lr2e5/
    torchrun -m tag.train --config <cfg> --run_suffix=lr5e5 \\
        --override learning_rate=5e-5
        # → <output_dir>/runs/20260515_230515_lr5e5/

    # Resume the most recent run (auto-resume picks the largest sealed epoch)
    torchrun -m tag.train --config <cfg> --run_tag=latest

    # See all prior runs
    python -m tag.train --config <cfg> --list_runs

Each run dir contains:
    cfg.yaml + cfg.json   — full resolved hyperparameter snapshot
    epoch_last/           — final-epoch weights only (tag.train writes
                            just the last epoch to save disk). Comparison
                            baselines under baselines.<method> still
                            emit epoch_N/ per epoch.
                            Contents: optimizer.pt, scheduler.pt,
                            trajectory_anchor.pt, env_meta.json,
                            anchor_history.json, _complete sentinel.
    metrics.json          — per-epoch loss + selection diagnostics
    selected_indices_epoch{N}.json  — exact data subset used per epoch
    logs/                 — train_<method>_<ts>_r<rank>.log
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Cap BLAS / OMP thread pools BEFORE `import torch` — libgomp + MKL read
# these at library init (which fires inside `import torch`), so setting
# them later has zero effect. Skip-symptom of the un-capped version on
# big-core hosts (>64 cores) is, inside TrajectoryAnchor.update's PCA loop:
#     libgomp: Thread creation failed: Resource temporarily unavailable
# triggered by 32 layers × torch.linalg.eigh each spawning OMP_NUM_THREADS
# default (= num_cores) workers in tight succession, blowing past the
# host's `ulimit -u` / cgroup pids.max. scripts/setup_env.sh also exports
# these for the source-then-launch flow; this block covers the case where
# the user invokes `python -m tag.train ...` without sourcing first.
for _k, _v in (
    ("OMP_NUM_THREADS", "16"), ("MKL_NUM_THREADS", "16"),
    ("OPENBLAS_NUM_THREADS", "16"), ("NUMEXPR_NUM_THREADS", "16"),
    ("VECLIB_MAXIMUM_THREADS", "16"),
):
    os.environ.setdefault(_k, _v)

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
from tag.core.schedulers import (
    get_cosine_schedule_with_warmup,
    optimizer_steps_per_epoch,
)
from tag.core.timing import PhaseTimer

from tag.core.run_layout import (
    find_latest_complete_epoch,
    list_runs as _list_runs,
    make_run_tag,
    resolve_latest,
    run_dir_for,
    save_cfg_snapshot,
    update_latest,
)
from tag.core.trajectory_anchor import TrajectoryAnchor
from tag.core.utils import (
    clear_runtime_caches,
    cuda_mem_str,
    disable_coredumps,
    is_main_process,
    load_config,
    local_rank,
    quiet_repeated_warnings,
    set_seed,
    setup_logger,
    world_size,
)
from tag.data.alpaca import build_alpaca_dataset
from tag.modeling.loader import get_hidden_size, load_model, load_tokenizer
from tag.pipelines.selection import save_selection, select_indices
from tag.pipelines.sft import make_dataloader, sft_one_epoch


# selection.tag keys consumed before the gate context is built, so their
# absence from tag_ctx["params"] is correct rather than a dropped key.
# Used by the guard in main() — see the comment there.
TAG_PARAMS_CONSUMED_ELSEWHERE = frozenset({
    "counterfactual_data_files",  # -> cf_datasets
    "dedup_clusters_file",        # -> cluster_ids
})



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
        help="Top-level or dotted nested overrides, e.g. selection_ratio=0.3 anchor.max_samples_for_pca=2048 selection.lam=0.5",
    )
    p.add_argument(
        "--run_tag",
        default=None,
        help=(
            "Folder name for this training run under <output_dir>/runs/. "
            "Defaults to a timestamp YYYYMMDD_HHMMSS so a re-run with tweaked "
            "hyperparameters never overwrites a previous run. Pass an existing "
            "run_tag to RESUME that run (auto-resume reads the largest "
            "_complete-sealed epoch_N inside it). Pass --run_tag=latest to "
            "resume whatever the _latest pointer currently selects."
        ),
    )
    p.add_argument(
        "--run_suffix",
        default="",
        help=(
            "Optional suffix appended to the auto timestamp tag, e.g. "
            "--run_suffix=lr2e5 produces runs/20260515_180000_lr2e5/. Ignored "
            "if --run_tag is also given."
        ),
    )
    p.add_argument(
        "--list_runs",
        action="store_true",
        help="Print the existing runs/ history under <output_dir> and exit.",
    )
    return p.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides) -> None:
    """Apply ``key=value`` overrides; supports dotted nested keys.

    Examples:
        selection_ratio=0.3
        anchor.max_samples_for_pca=2048
        anchor.layer_idx=-1
        selection.lam=0.5

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
        # Backend selectable via TAG_DDP_BACKEND env. NCCL is fast but
        # has the idle-then-hang failure mode under rank-0-solo workloads
        # like collect_episode. gloo is ~5x slower but TCP-based and far
        # more robust; use it as an escape hatch when NCCL won't behave.
        #     export TAG_DDP_BACKEND=gloo
        backend = os.environ.get("TAG_DDP_BACKEND", "nccl").lower()
        if backend not in ("nccl", "gloo"):
            print(f"[ddp] unknown TAG_DDP_BACKEND={backend!r}, falling back to nccl",
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
    # Cap RLIMIT_CORE on this process and all forks (torchrun spawns).
    # The shell-level `ulimit -c 0` in setup_env.sh only protects launches
    # that actually sourced it; cron / tmux-reopen / fresh-login flows
    # bypass it and a single segfaulting 7B-DDP rank then drops ~240 GB
    # of core onto the 50 GB user-volume, ENOSPC'ing everything else.
    # Enforce from Python so the shell isn't load-bearing.
    disable_coredumps()

    # Start from a clean process-local cache state: GC arena, CUDA
    # allocator cache, and stale CUDA-IPC handles from prior runs. See
    # clear_runtime_caches() docstring for the failure modes this guards.
    clear_runtime_caches()

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

    # ---------- run-layout: history-preserving runs/<tag>/ + _latest pointer ----------
    # Each invocation writes its checkpoints under <output_dir>/runs/<run_tag>/
    # so re-running with tweaked hyperparameters never overwrites a prior run.
    # The _latest symlink under <output_dir>/ tracks the most recent run, and
    # eval defaults to reading from there. See tag.core.run_layout for the
    # full contract.
    if args.list_runs:
        if is_main_process():
            existing = _list_runs(output_dir)
            if not existing:
                print(f"No runs/ directory under {output_dir}.")
            else:
                latest = resolve_latest(output_dir)
                latest_name = latest.name if latest is not None else "(unset)"
                print(f"Runs under {output_dir}:")
                for tag, _ in existing:
                    marker = "  <- _latest" if (latest and tag == latest.name) else ""
                    print(f"  {tag}{marker}")
                print(f"_latest -> {latest_name}")
        return

    if args.run_tag == "latest":
        latest = resolve_latest(output_dir)
        if latest is None:
            raise FileNotFoundError(
                f"--run_tag=latest requested but no _latest pointer under "
                f"{output_dir}. Run training without --run_tag first.",
            )
        run_tag = latest.name
    elif args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = make_run_tag(args.run_suffix)
    run_dir = run_dir_for(output_dir, run_tag)

    # Expose both dirs to downstream modules — notably
    # pipelines.selection._broadcast_selection, which writes a temporary
    # selection-share file. Without this, multiple parallel jobs (qwen +
    # llama2 + mistral + deepseek launched concurrently via run_main_7b.sh)
    # would all fall back to ``cfg["output_root"]`` and clobber each
    # other's _selection_epoch{N}.json — silently mixing their selected
    # indices across experiments. ``output_dir`` is still the
    # per-experiment dir; ``run_dir`` is the per-run dir inside it.
    cfg["output_dir"] = str(run_dir)
    cfg["experiment_dir"] = str(output_dir)
    cfg["run_tag"] = run_tag
    log_dir = Path(cfg.get("log_dir", run_dir / "logs"))
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    logger = setup_logger(str(log_dir), name=f"train_{method}")
    timer = PhaseTimer(log=logger if is_main_process() else None, method=method)
    if is_main_process():
        logger.info("=" * 60)
        logger.info(
            "TAG unified trainer | method=%s | ddp=%s | world_size=%d",
            method, use_ddp, world_size(),
        )
        logger.info("experiment_dir = %s", output_dir)
        logger.info("run_dir        = %s   (run_tag=%s)", run_dir, run_tag)
        logger.info("Config:\n%s", json.dumps(cfg, indent=2, default=str))
        logger.info("=" * 60)
        # Persist the resolved cfg snapshot so a future eval / audit knows
        # exactly which hyperparameters produced this run's checkpoints.
        # Atomic write inside the helper.
        save_cfg_snapshot(run_dir, cfg)

    # ---------- resume: find latest epoch checkpoint INSIDE current run ----------
    # Resume only crosses epochs WITHIN the same run_tag — never across runs,
    # which would silently mix hyperparameter regimes. To resume yesterday's
    # run, pass --run_tag=<that_tag> (or --run_tag=latest).
    resume_epoch, resume_ckpt = find_latest_complete_epoch(run_dir)
    if resume_ckpt is not None and is_main_process():
        logger.info(
            "RESUMING from %s (epoch %d completed; continuing at epoch %d)",
            resume_ckpt, resume_epoch, resume_epoch + 1,
        )
    if use_ddp:
        dist.barrier()

    # ---------- tokenizer / model ----------
    # Load tokenizer from base path (does not change between epochs).
    with timer.phase("tokenizer_load", "setup"):
        tokenizer = load_tokenizer(cfg["model_path"])

    # If resuming, load model weights from the checkpoint dir; else from base.
    model_load_path = str(resume_ckpt) if resume_ckpt is not None else cfg["model_path"]
    if is_main_process():
        logger.info("Loading model from: %s", model_load_path)

    training_mode = str(cfg.get("training_mode", "full"))
    # If we're resuming, the checkpoint dir's contents are the source of
    # truth for training_mode — a LoRA epoch dir has only
    # ``adapter_config.json`` (no full weights), and a full-FT dir has
    # ``config.json``. A config-vs-checkpoint mismatch used to silently
    # send full-FT through the LoRA path (or vice versa), producing
    # cryptic load errors a few lines later. Auto-correct + warn so the
    # resume actually picks up where the previous run left off.
    _adapter_path: Optional[str] = None
    if resume_ckpt is not None:
        _resume_path = Path(resume_ckpt)
        _has_adapter = (_resume_path / "adapter_config.json").exists()
        _has_full = (_resume_path / "config.json").exists() and not _has_adapter
        _detected = "lora" if _has_adapter else ("full" if _has_full else None)
        if _detected is not None and _detected != training_mode:
            if is_main_process():
                logger.warning(
                    "training_mode=%r in config disagrees with resume checkpoint "
                    "(%s contains %s). Overriding to %r to match the checkpoint.",
                    training_mode, resume_ckpt,
                    "adapter_config.json (LoRA)" if _has_adapter else "config.json (full)",
                    _detected,
                )
            training_mode = _detected
        # LoRA epoch dirs only contain the adapter — base weights must come
        # from cfg["model_path"]. Route the resume_ckpt as adapter_path and
        # reset model_load_path back to the base. Full-FT epoch dirs hold
        # the whole model so model_load_path stays pointed at them.
        if training_mode == "lora" and _has_adapter:
            _adapter_path = str(resume_ckpt)
            model_load_path = cfg["model_path"]
    with timer.phase("model_load", "setup"):
        model = load_model(
            model_load_path,
            training_mode=training_mode,
            lora_cfg=cfg.get("lora"),
            use_ddp=use_ddp,
            local_rank=local_rank(),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            attn_implementation=cfg.get("attn_implementation"),
            adapter_path=_adapter_path,
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
    # `or` (not `get(..., default)`) so an empty-string or null YAML value
    # falls back to alpaca_default instead of being passed down literally.
    style_key = str(cfg.get("prompt_style") or "alpaca_default")
    effective_cache = os.path.join(
        str(cfg["data_cache"]), model_key, style_key,
    )
    if is_main_process():
        logger.info("HF datasets cache: %s", effective_cache)
    with timer.phase("dataset_build", "data"):
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
    keep_indices = None
    if sub_n is not None and int(sub_n) < n_total_full:
        g = torch.Generator(); g.manual_seed(seed)
        keep_indices = torch.randperm(n_total_full, generator=g).tolist()[: int(sub_n)]
        dataset = dataset.select(keep_indices)
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

    anchor: Optional[TrajectoryAnchor] = None

    if method == "selection" and is_main_process():
        anchor_cfg = cfg.get("anchor", {}) or {}
        anchor = TrajectoryAnchor(
            layer_idx=int(anchor_cfg.get("layer_idx", -1)),
            layer_indices=anchor_cfg.get("layer_indices"),
            max_samples_for_pca=int(anchor_cfg.get("max_samples_for_pca", 2000)),
            pca_batch_size=int(anchor_cfg.get("pca_batch_size", 4)),
            device=str(device),
        )

    # ---------- MVF context (score_mode: mvf; low-quality-pool score) ----------
    # Builds the reliability/learnability side inputs on rank 0 only —
    # selection itself runs on rank 0, and the counterfactual tokenisation
    # would otherwise race on the HF datasets cache across ranks.
    mvf_ctx = None
    tag_ctx = None
    selection_cfg_top = cfg.get("selection", {}) or {}
    _score_mode = str(selection_cfg_top.get("score_mode", "legacy"))
    _VALID_SCORE_MODES = ("legacy", "mvf", "tag")
    if method == "selection" and _score_mode not in _VALID_SCORE_MODES:
        # Previously an unknown score_mode fell through to the LEGACY path
        # with no error and no warning, so a typo silently ran an ungated
        # baseline under the name of a gated arm.
        raise ValueError(
            f"selection.score_mode={_score_mode!r} is not recognised. Valid: "
            f"{_VALID_SCORE_MODES} ('legacy' = legacy Eq. 10, 'mvf' = "
            f"multi-view fusion, 'tag' = reliability-gated Eq. 1).",
        )
    if (
        method == "selection"
        and _score_mode == "mvf"
        and is_main_process()
    ):
        from tag.core.dedup import load_clusters
        from tag.core.reliability import completeness_from_dataset

        mvf_cfg = selection_cfg_top.get("mvf", {}) or {}
        cf_files = mvf_cfg.get("counterfactual_data_files") or None
        # K > 1 counterfactual pools (evidential-lite Q, plan §2.1 v3):
        # accept a single path, a YAML list, or a comma-separated string —
        # the last so the documented env plumbing can express K > 1:
        #   export TAG_CF_FILES=pool/counterfactual_1.json,pool/counterfactual_2.json
        if cf_files and not isinstance(cf_files, (list, tuple)):
            cf_files = [p.strip() for p in str(cf_files).split(",") if p.strip()]
        cf_datasets = []
        if cf_files:
            with timer.phase("counterfactual_build", "data"):
                for k, one in enumerate(cf_files, start=1):
                    cf_dataset = build_alpaca_dataset(
                        tokenizer=tokenizer,
                        cache_dir=os.path.join(
                            effective_cache,
                            "counterfactual" if k == 1 else f"counterfactual_{k}",
                        ),
                        max_seq_len=int(cfg["max_seq_len"]),
                        dataset_name=None,
                        data_files=str(one),
                        prompt_style=style_key,
                    )
                    if keep_indices is not None:
                        cf_dataset = cf_dataset.select(keep_indices)
                    if len(cf_dataset) != n_total:
                        raise ValueError(
                            f"Counterfactual pool #{k} size {len(cf_dataset)} != "
                            f"candidate pool size {n_total}. Counterfactual files "
                            f"must be index-aligned with data_files (regenerate "
                            f"both with scripts/make_corrupted_pool.py).",
                        )
                    cf_datasets.append(cf_dataset)
        cluster_ids = None
        cluster_file = mvf_cfg.get("dedup_clusters_file") or None
        if cluster_file:
            cluster_ids = load_clusters(str(cluster_file))
            if keep_indices is not None:
                cluster_ids = [cluster_ids[i] for i in keep_indices]
            if len(cluster_ids) != n_total:
                raise ValueError(
                    f"dedup_clusters_file length {len(cluster_ids)} != pool "
                    f"size {n_total} — cluster file built for a different pool?",
                )
        completeness = completeness_from_dataset(
            dataset,
            eos_token_id=tokenizer.eos_token_id,
            c_trunc=float(mvf_cfg.get("c_trunc", 0.2)),
        )
        mvf_ctx = {
            "completeness": completeness,
            "cf_datasets": cf_datasets or None,
            "cluster_ids": cluster_ids,
            "params": {
                "eta": float(mvf_cfg.get("eta", 0.5)),
                "gamma": float(mvf_cfg.get("gamma", 1.0)),
                "eps": float(mvf_cfg.get("eps", 0.01)),
                # ---- v3 score parameters (plan §2) ----
                "d_floor": float(mvf_cfg.get("d_floor", 0.5)),
                "progress_mode": str(mvf_cfg.get("progress_mode", "split")),
                "reliability_mode": str(mvf_cfg.get("reliability_mode", "sigmoid")),
                # Env-interpolated empty strings mean "unset" — normalise here
                # so downstream float() never sees '' (0.0 stays a valid value,
                # hence the explicit check rather than `or None`).
                "reliability_scale": (
                    None
                    if mvf_cfg.get("reliability_scale") is None
                    or str(mvf_cfg.get("reliability_scale")).strip() == ""
                    else float(mvf_cfg.get("reliability_scale"))
                ),
                "reliability_ref_file": (mvf_cfg.get("reliability_ref_file") or None),
                "reliability_rezero": bool(mvf_cfg.get("reliability_rezero", True)),
                "calibration_target_pct": float(mvf_cfg.get("calibration_target_pct", 0.10)),
                "calibration_target_q": float(mvf_cfg.get("calibration_target_q", 0.8)),
                "allow_late_reliability": bool(mvf_cfg.get("allow_late_reliability", False)),
                "static": bool(mvf_cfg.get("static", False)),
                "adaptive_lam": bool(mvf_cfg.get("adaptive_lam", False)),
            },
        }
        logger.info(
            "MVF context ready | counterfactuals=%d | dedup_clusters=%s | "
            "eta=%.2f gamma=%.2f eps=%.3f d_floor=%.2f c_trunc=%.2f | "
            "reliability=%s(rezero=%s, scale=%s) | progress=%s | static=%s | "
            "adaptive_lam=%s",
            len(cf_datasets),
            "yes" if cluster_ids is not None else "no",
            mvf_ctx["params"]["eta"], mvf_ctx["params"]["gamma"],
            mvf_ctx["params"]["eps"], mvf_ctx["params"]["d_floor"],
            float(mvf_cfg.get("c_trunc", 0.2)),
            mvf_ctx["params"]["reliability_mode"],
            mvf_ctx["params"]["reliability_rezero"],
            mvf_ctx["params"]["reliability_scale"],
            mvf_ctx["params"]["progress_mode"],
            mvf_ctx["params"]["static"],
            mvf_ctx["params"]["adaptive_lam"],
        )

    # ---------- TAG context (score_mode: tag; paper Eq. 1) ----------
    # Same rank-0-only contract as the MVF block above: the counterfactual
    # tokenisation would race on the HF datasets cache across ranks, and
    # selection itself only runs on rank 0.
    if (
        method == "selection"
        and _score_mode == "tag"
        and is_main_process()
    ):
        from tag.core.dedup import load_clusters
        from tag.core.reliability import completeness_from_dataset

        tag_cfg = selection_cfg_top.get("tag", {}) or {}
        cf_files = tag_cfg.get("counterfactual_data_files") or None
        if cf_files and not isinstance(cf_files, (list, tuple)):
            cf_files = [p.strip() for p in str(cf_files).split(",") if p.strip()]
        if not cf_files:
            raise ValueError(
                "selection.score_mode='tag' requires selection.tag.counterfactual_data_files "
                "(the x^- pool). Generate it with:\n"
                "    python scripts/make_corrupted_pool.py ... --emit-counterfactual\n"
                "then export TAG_CF_FILES=<pool>/counterfactual.json",
            )
        cf_datasets = []
        with timer.phase("counterfactual_build", "data"):
            for k, one in enumerate(cf_files, start=1):
                cf_dataset = build_alpaca_dataset(
                    tokenizer=tokenizer,
                    cache_dir=os.path.join(
                        effective_cache,
                        "counterfactual" if k == 1 else f"counterfactual_{k}",
                    ),
                    max_seq_len=int(cfg["max_seq_len"]),
                    dataset_name=None,
                    data_files=str(one),
                    prompt_style=style_key,
                )
                if keep_indices is not None:
                    cf_dataset = cf_dataset.select(keep_indices)
                if len(cf_dataset) != n_total:
                    raise ValueError(
                        f"Counterfactual pool #{k} size {len(cf_dataset)} != "
                        f"candidate pool size {n_total}. Counterfactual files "
                        f"must be index-aligned with data_files (regenerate "
                        f"both with scripts/make_corrupted_pool.py).",
                    )
                cf_datasets.append(cf_dataset)
        cluster_ids = None
        cluster_file = tag_cfg.get("dedup_clusters_file") or None
        if cluster_file:
            cluster_ids = load_clusters(str(cluster_file))
            if keep_indices is not None:
                cluster_ids = [cluster_ids[i] for i in keep_indices]
            if len(cluster_ids) != n_total:
                raise ValueError(
                    f"dedup_clusters_file length {len(cluster_ids)} != pool "
                    f"size {n_total} — cluster file built for a different pool?",
                )
        completeness = completeness_from_dataset(
            dataset,
            eos_token_id=tokenizer.eos_token_id,
            c_trunc=float(tag_cfg.get("c_trunc", 0.2)),
        )

        def _opt_float(key):
            """Env-interpolated '' means unset; 0.0 must stay a valid value."""
            v = tag_cfg.get(key)
            return None if v is None or str(v).strip() == "" else float(v)

        tag_ctx = {
            "completeness": completeness,
            "dataset": dataset,
            "cf_datasets": cf_datasets,
            "cluster_ids": cluster_ids,
            "eos_token_id": tokenizer.eos_token_id,
            "params": {
                # ---- span aggregation (paper Eqs. 4-5) ----
                "span_tokens": int(tag_cfg.get("span_tokens", 16)),
                "tau": float(tag_cfg.get("tau", 0.5)),
                "tau_mode": str(tag_cfg.get("tau_mode", "per_token")),
                "min_span_tokens": int(tag_cfg.get("min_span_tokens", 4)),
                "tail_mode": str(tag_cfg.get("tail_mode", "min")),
                "tail_quantile": float(tag_cfg.get("tail_quantile", 0.0)),
                "include_eos": bool(tag_cfg.get("include_eos", False)),
                "prefix_tokens": int(tag_cfg.get("prefix_tokens", 0)),
                # ---- gate (paper Eq. 6) ----
                "c_trunc": float(tag_cfg.get("c_trunc", 0.2)),
                "eps_den": float(tag_cfg.get("eps_den", 1e-3)),
                "min_common_tokens": int(tag_cfg.get("min_common_tokens", 8)),
                "undefined_policy": str(tag_cfg.get("undefined_policy", "neutral")),
                "undefined_gate_value": float(
                    tag_cfg.get("undefined_gate_value", 0.6)
                ),
                "gate_scale": _opt_float("gate_scale"),
                "gate_ref_file": (tag_cfg.get("gate_ref_file") or None),
                "calibration_target_pct": float(
                    tag_cfg.get("calibration_target_pct", 0.10)
                ),
                "calibration_target_q": float(tag_cfg.get("calibration_target_q", 0.8)),
                "dispersion_discount": bool(tag_cfg.get("dispersion_discount", True)),
                "null_correction": bool(tag_cfg.get("null_correction", True)),
                "target_zero_rate": float(tag_cfg.get("target_zero_rate", 0.05)),
                # Score-time weakening ablations.  These intentionally stay
                # outside GateConfig/cache identity: every arm consumes the
                # same calibrated raw G and changes only its fusion weight.
                "gate_power": float(tag_cfg.get("gate_power", 1.0)),
                "gate_strength": float(tag_cfg.get("gate_strength", 1.0)),
                # ---- lifecycle ----
                "allow_late_gate": bool(tag_cfg.get("allow_late_gate", False)),
                "store_token_losses": bool(tag_cfg.get("store_token_losses", False)),
                "static": bool(tag_cfg.get("static", False)),
                # Shared, precomputed gate cache (scripts/precompute_gate.py).
                # Empty = per-run cache in the run dir.
                "gate_cache_file": (tag_cfg.get("gate_cache_file") or None),
            },
        }
        # Every selection.tag.* key must reach the gate. This dict used to be
        # a hand-maintained whitelist, and `prefix_tokens` was never added to
        # it: main_7b/llama2/tag_10.yaml asked for 32, _build_gate_config read
        # its own default of 0, and the Table 2 TAG row trained for three
        # seeds on a gate that zeroed 49% of a CLEAN pool instead of the
        # configured 5%. Nothing failed; the number was just wrong.
        # `null_correction` and `target_zero_rate` were missing the same way
        # and only looked correct because the arm happened to want the
        # defaults — main2/tag_nonull_7b.yaml's `null_correction: false` would
        # have been dropped in silence, running the ablation arm as the
        # non-ablation.
        #
        # So: refuse to start when a key in the YAML is not forwarded, rather
        # than let it default. Keys consumed before this point are named here
        # because they are legitimately not gate parameters.
        _dropped = sorted(
            set(tag_cfg) - set(tag_ctx["params"]) - TAG_PARAMS_CONSUMED_ELSEWHERE
        )
        if _dropped:
            raise ValueError(
                f"selection.tag keys set in the config never reach the gate: "
                f"{_dropped}. They would silently fall back to the defaults in "
                f"tag.pipelines.selection._build_gate_config, producing a run "
                f"whose G does not match what the config says. Forward them in "
                f"tag/train.py's tag_ctx['params'], or add them to "
                f"TAG_PARAMS_CONSUMED_ELSEWHERE if they are not gate parameters."
            )

        logger.info(
            "TAG context ready | counterfactuals=%d | dedup_clusters=%s | "
            "W=%d tau=%.3f(%s) min_span=%d tail=%s | c_trunc=%.2f | "
            "gate_scale=%s ref=%s | power=%.3f strength=%.3f | "
            "undefined=%s | static=%s",
            len(cf_datasets),
            "yes" if cluster_ids is not None else "no",
            tag_ctx["params"]["span_tokens"], tag_ctx["params"]["tau"],
            tag_ctx["params"]["tau_mode"], tag_ctx["params"]["min_span_tokens"],
            tag_ctx["params"]["tail_mode"], tag_ctx["params"]["c_trunc"],
            tag_ctx["params"]["gate_scale"], tag_ctx["params"]["gate_ref_file"],
            tag_ctx["params"]["gate_power"], tag_ctx["params"]["gate_strength"],
            tag_ctx["params"]["undefined_policy"], tag_ctx["params"]["static"],
        )

    # ---------- optimizer / scheduler ----------
    # Plan the schedule from the data layout the trainer ACTUALLY executes.
    # DistributedSampler pads each rank to ceil(K / world_size), DataLoader
    # keeps its final partial batch, and the SFT loop keeps its final partial
    # accumulation window.  The historical one-line int(...) floored all
    # three divisions at once: Table-2 TAG planned 40 steps/epoch while it
    # executes 41, so the last three updates ran after cosine reached zero.
    planned_selected = (
        n_total if method == "full"
        else max(1, int(n_total * selection_ratio))
    )
    planned_steps_per_epoch = optimizer_steps_per_epoch(
        planned_selected,
        batch_size=batch_size,
        grad_accum=grad_accum,
        world_size=max(1, world_size()),
    )
    total_steps = planned_steps_per_epoch * train_epochs
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
    # 8-bit AdamW cuts optimizer state from ~56GB to ~14GB per GPU on 7B
    # full fine-tuning. Matches the NAIT paper's recipe (bnb.optim.AdamW8bit).
    #
    # bnb's import path is the most common CUDA-lib failure surface in this
    # codebase: the package compiles against a specific libcudart minor
    # version and routinely fails to import with OSError ("libcudart.so.X.Y:
    # cannot open shared object file"), RuntimeError ("CUDA Setup failed
    # despite GPU being available"), or AttributeError on nodes where the
    # CUDA driver is older than the wheel. When that happens we'd rather
    # train with fp32 AdamW (slower / more VRAM but functionally correct)
    # than crash the run — fall back with a loud warning so the user can
    # pin a working bnb build at their leisure.
    wd = float(cfg.get("weight_decay", 0.1))
    want_8bit = bool(cfg.get("use_8bit_optimizer", False))
    optimizer = None

    # Two-stage bnb safety:
    #   (A) PROBE — import bnb and run a real AdamW8bit.step() on a 1-element
    #       synthetic CUDA tensor. This exercises bnb's full CUDA kernel
    #       launch path (libcublasLt.so.X.Y, libcudart.so.X.Y) which is
    #       commonly lazily loaded only at first-step time. Without the
    #       probe, a version mismatch in those libs shows up 30+ minutes
    #       into collect_episode (data_agent/selection spend that entire window
    #       BEFORE the first optimizer.step()), wasting the run.
    #   (B) BUILD — construct the real optimizer over model.parameters().
    #       Only reached if (A) passes.
    # Any exception from (A) flips want_8bit off for the rest of the run.
    # Both stages catch the same exception classes (ImportError, OSError,
    # RuntimeError, AttributeError) — these cover "module missing", "lib
    # not found", "CUDA setup failed", and "missing kernel symbol" in turn.
    if want_8bit and torch.cuda.is_available():
        _probe_dev = torch.device(f"cuda:{local_rank()}")
        try:
            import bitsandbytes as _bnb_probe
            _probe_p = torch.nn.Parameter(
                torch.zeros(1, device=_probe_dev, dtype=torch.float32),
            )
            _probe_opt = _bnb_probe.optim.AdamW8bit([_probe_p], lr=1e-5)
            _probe_p.grad = torch.zeros_like(_probe_p)
            _probe_opt.step()
            torch.cuda.synchronize()
            del _probe_opt, _probe_p
            torch.cuda.empty_cache()
            if is_main_process():
                logger.info(
                    "bitsandbytes early probe OK (version=%s) on %s",
                    getattr(_bnb_probe, "__version__", "?"), _probe_dev,
                )
        except (ImportError, OSError, RuntimeError, AttributeError) as _bnb_e:
            if is_main_process():
                logger.error(
                    "bitsandbytes early probe FAILED (%s: %s). "
                    "Auto-disabling use_8bit_optimizer for THIS run and "
                    "falling back to torch.AdamW (fp32). This typically "
                    "means the installed bnb wheel was compiled against a "
                    "different CUDA minor version than this node's "
                    "runtime — common signature is `OSError: libcublasLt.so.X.Y: "
                    "cannot open shared object file`. To re-enable bnb, "
                    "install a matching wheel (see bitsandbytes-foundation "
                    "release notes for your CUDA version). Continuing with "
                    "fp32 AdamW so this training run isn't lost.",
                    type(_bnb_e).__name__, _bnb_e,
                )
            want_8bit = False
            cfg["use_8bit_optimizer"] = False

    if want_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, weight_decay=wd,
            )
            if is_main_process():
                logger.info(
                    "Optimizer: bitsandbytes.AdamW8bit | wd=%s | probe=OK", wd,
                )
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            # Should be unreachable when the probe above passed, but keep
            # the catch as defense-in-depth (e.g. probe used cuda:0 while
            # model.parameters() live elsewhere, or DDP rank-specific state).
            if is_main_process():
                logger.warning(
                    "bnb construction failed AFTER passing probe "
                    "(%s: %s). Falling back to torch.optim.AdamW (fp32).",
                    type(e).__name__, e,
                )
    if optimizer is None:
        # PyTorch defaults CUDA AdamW to the foreach implementation when this
        # argument is None.  For full-FT 7B, foreach needs an additional
        # tensor-list roughly the size of the parameters at optimizer.step(),
        # which can OOM an otherwise valid 80GB run.  Keep the historical
        # default when the key is absent, but allow an explicit, cfg-recorded
        # false value for memory-bounded runs.
        adamw_foreach = cfg.get("adamw_foreach")
        if adamw_foreach is not None:
            adamw_foreach = bool(adamw_foreach)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd,
            foreach=adamw_foreach,
        )
        if is_main_process():
            logger.info(
                "Optimizer: torch.AdamW (fp32) | wd=%s | want_8bit=%s | "
                "foreach=%s",
                wd, want_8bit, adamw_foreach,
            )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    if is_main_process():
        logger.info(
            "LR schedule: cosine | selected=%d | steps/epoch=%d | epochs=%d "
            "| total_steps=%d | warmup_steps=%d | min_lr_ratio=%.3f "
            "| effective_batch=%d",
            planned_selected, planned_steps_per_epoch, train_epochs,
            total_steps, warmup_steps, min_lr_ratio,
            batch_size * grad_accum * max(1, world_size()),
        )

    # ---------- resume: restore optimizer/scheduler/anchor/metrics ----------
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
        metrics_json = run_dir / "metrics.json"
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

        with timer.phase(f"selection_epoch{epoch}", "selection"):
            selected, extras = select_indices(
                method,
                model=model,
                anchor=anchor,
                dataset=dataset,
                cfg=cfg,
                epoch=epoch,
                seed=seed,
                device=device,
                mvf_ctx=mvf_ctx,
                tag_ctx=tag_ctx,
            )
            save_selection(run_dir, epoch, selected)

        if len(selected) == 0:
            raise RuntimeError(
                f"Epoch {epoch}: selected indices is empty. SFT would produce "
                "0 batches and DDP all_reduce at end of empty loop is a known "
                "hang source. Check selection_ratio and dataset size.",
            )

        # NCCL reinit workaround: OPT-IN via TAG_NCCL_REINIT=1.
        # Empirically `destroy_process_group` itself can hang when the
        # NCCL communicator is in a deeply broken state — making this
        # "recovery" path itself a deadlock. Default is OFF; rely on
        # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=99999 + async-error env vars
        # to prevent the communicator from dying in the first place.
        if (
            method in ("selection", "data_agent")
            and use_ddp
            and os.environ.get("TAG_NCCL_REINIT", "0") == "1"
        ):
            model = _reinit_ddp_after_long_idle(model, use_ddp)

        subset = Subset(dataset, selected)
        loader = make_dataloader(
            subset, batch_size=batch_size, shuffle=True, seed=seed, epoch=epoch,
        )
        with timer.phase(f"sft_epoch{epoch}", "sft"):
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
        # Bug history: training with tag/data_agent under DDP was crashing
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

            # epoch_last/ layout (2026-05-16): instead of epoch_1, epoch_2, ...,
            # every save overwrites a single `epoch_last/` directory. The
            # actual epoch number lives in two places:
            #   - <run>/metrics.json (per-epoch rows, last row's "epoch" field)
            #   - <run>/epoch_last/_complete sentinel content (single int)
            # cfg.json snapshot stores the run's target train_epochs, so
            # "is training done?" = (sealed_epoch_from_sentinel >= train_epochs).
            # Eval picks <run>/epoch_last/ unconditionally — no need to scan
            # for the max numeric.
            ckpt_path = run_dir / "epoch_last"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            save_errors: list = []
            _ckpt_timer_phase = timer.phase(f"checkpoint_epoch{epoch}", "checkpoint")
            _ckpt_timer_phase.__enter__()

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
                                            run_dir / "metrics.json"))

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
            _ckpt_timer_phase.__exit__(None, None, None)

            # keep_last_n_checkpoints is now a no-op with the epoch_last layout
            # (only one ckpt dir per run by construction). Left intact in cfg
            # for backward compat with older YAMLs that set it. If a run dir
            # has legacy epoch_N/ subdirs alongside the new epoch_last/ (from
            # before this migration), the user can clean them manually — we
            # don't auto-delete to keep the migration explicit.
            _keep = int(cfg.get("keep_last_n_checkpoints", 0))
            if _keep > 0:
                logger.info(
                    "keep_last_n_checkpoints=%d ignored: epoch_last/ layout "
                    "stores only one ckpt per run.", _keep,
                )

            # Update _latest pointer after each completed epoch save so an
            # eval can fire mid-training (after epoch 1 finishes, before
            # epoch 2 starts) and pick up the freshly sealed checkpoint.
            if ok_model and ok_opt:
                try:
                    mech = update_latest(output_dir, run_tag)
                    logger.info("_latest -> runs/%s (%s)", run_tag, mech)
                except Exception as exc:
                    logger.warning("Failed to update _latest pointer: %s", exc)

        # Workers MUST hit the barrier even if rank 0 raised inside the save
        # block — otherwise NCCL stalls until its timeout and rank 0's exit
        # code is masked by a generic collective failure on every worker.
        if use_ddp:
            dist.barrier()

    if is_main_process():
        timer.save_report(run_dir / "timing_breakdown.json")
        timer.log_table()
        logger.info("Training complete (%d epochs).", train_epochs)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
