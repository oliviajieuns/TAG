"""NAIT (ICLR 2026) baseline training.

Usage:
    python -m tads.baselines.nait.train \\
        --config configs/methods/nait.yaml \\
        --seed_path seeds/mix.json --tag NAIT-Mix
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

# transformers 5.0 eager-imports `from torchvision.io import VideoReader`
# via its video model registry, which fails on torchvision builds without
# ffmpeg support — even though our LLM-only training never touches video.
# Stub the missing attribute BEFORE any transformers import (incl. via
# tads.modeling.loader) so the import resolves to a harmless placeholder.
# Same pattern as tads.train + every other tads/baselines/<m>/train.py.
try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import Subset
from tads.core.schedulers import get_cosine_schedule_with_warmup
from tads.core.utils import (
    clear_runtime_caches,
    disable_coredumps,
    load_config,
    set_seed,
    setup_logger,
)
from tads.data.alpaca import build_alpaca_dataset
from tads.modeling.loader import load_model, load_tokenizer
from tads.baselines.nait.direction import (
    extract_delta_from_seed,
    fit_directions,
    score_candidates,
)
from tads.pipelines.sft import make_dataloader, sft_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--seed_path",
        default=None,
        help=(
            "Path to NAIT seed JSON. If omitted (or the file doesn't exist), "
            "auto-build from cfg.data_files (= ALPACA_DATA_FILES) — random "
            "sample of `n_seeds` rows, cached to seeds/mix_auto.json for reuse."
        ),
    )
    p.add_argument("--tag", required=True, help="Variant tag, e.g. NAIT-Mix.")
    p.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help=(
            "Seed sample size. Default: cfg.nait.n_seeds. For auto-build, "
            "NAIT paper Table 7 'mix' variant uses 1500."
        ),
    )
    return p.parse_args()


def _ensure_seed_file(
    seed_path,
    *,
    data_files_spec: str,
    n_seeds: int,
    seed: int,
    logger,
) -> str:
    """Return a usable seed-JSON path; auto-build from Alpaca if missing.

    Resolution order:
        1. `seed_path` (CLI --seed_path) exists           → use as-is.
        2. `seed_path` given but missing                  → build at that path.
        3. `seed_path` is None  → use `seeds/mix_auto.json`; build if absent.
    """
    import glob as _glob

    if seed_path and Path(seed_path).exists():
        logger.info("Using user-provided seed file: %s", seed_path)
        return seed_path

    target = Path(seed_path) if seed_path else Path("seeds") / "mix_auto.json"
    if target.exists():
        logger.info("Reusing cached auto-built seed file: %s", target)
        return str(target)

    matches = sorted(_glob.glob(data_files_spec))
    if not matches:
        raise FileNotFoundError(
            f"NAIT seed auto-build failed: cfg.data_files glob {data_files_spec!r} "
            f"matched no files. Either pass --seed_path <existing.json>, or "
            f"export ALPACA_DATA_FILES to a valid path before launching."
        )
    logger.info(
        "Auto-building NAIT seed file from %d Alpaca file(s); target=%s, n=%d",
        len(matches), target, n_seeds,
    )
    # Cross-process lock so two concurrent training jobs sharing this seeds/
    # dir don't race on the cache write. filelock is already a pinned dep
    # (requirements.txt) used by tads/data/alpaca.py for the same purpose.
    try:
        from filelock import FileLock  # type: ignore
    except Exception:
        FileLock = None  # type: ignore
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(target) + ".lock"

    def _build_and_write():
        # Re-check inside the lock — the other process may have just written it.
        if target.exists():
            logger.info("Reusing cached seed file built by a concurrent job: %s", target)
            return
        # Read via the shared loader so JSON / JSONL / Parquet all work.
        from tads.core.data_io import read_records
        records = []
        for f in matches:
            records.extend(read_records(f))
        rng = random.Random(seed)
        sample = rng.sample(records, n_seeds) if len(records) > n_seeds else records
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as h:
            json.dump(sample, h)
        os.replace(tmp, target)
        logger.info("Wrote %d seed items to %s", len(sample), target)

    if FileLock is not None:
        with FileLock(lock_path, timeout=300):
            _build_and_write()
    else:
        _build_and_write()
    return str(target)


def main() -> None:
    # Cap RLIMIT_CORE on this process — see tads.train.main for rationale.
    disable_coredumps()

    # Clean process-local cache state — see clear_runtime_caches() docs.
    clear_runtime_caches()

    # OFFLINE BY DEFAULT — see tads.train.main for rationale.
    import os as _os
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

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
    # Default to 1500 (NAIT paper Table 7 'mix' variant). cfg.nait.n_seeds in
    # the shipped methods/nait.yaml is set to a very large cap (100000) so it
    # acts as no-op; users wanting a different mix size should pass --n_seeds.
    n_seeds = int(args.n_seeds or cfg.get("nait", {}).get("n_seeds", 1500))
    seed_path = _ensure_seed_file(
        args.seed_path,
        data_files_spec=str(cfg["data_files"]),
        n_seeds=n_seeds,
        seed=seed,
        logger=logger,
    )
    with open(seed_path) as f:
        seed_items = json.load(f)
    if len(seed_items) > n_seeds:
        rng = random.Random(seed)
        seed_items = rng.sample(seed_items, n_seeds)
    logger.info("Loaded %d seed items from %s", len(seed_items), seed_path)

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
        # `or "alpaca_default"` (not just `.get(..., "alpaca_default")`)
        # so a YAML that sets prompt_style: "" or null falls back instead
        # of passing the empty string down to tokenize_alpaca, which would
        # then raise ValueError("Unknown prompt_style=''").
        prompt_style=str(cfg.get("prompt_style") or "alpaca_default"),
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
