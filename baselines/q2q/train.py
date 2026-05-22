"""Q2Q / Cherry_LLM (Li et al., 2024 NAACL) baseline training.

Compute IFD = PPL(y | x) / PPL(y) for every Alpaca-GPT4 sample using the
base model directly (paper §3.2 uses a brief-experience precursor; we
simplify to the base model — set `q2q.use_precursor: true` in the future
to enable the full pipeline). Then filter to IFD ∈ [ifd_low, ifd_high] and
take top-`selection_ratio` by IFD, then SFT.

Usage:
    source scripts/setup_env.sh
    CUDA_VISIBLE_DEVICES=0 python -m baselines.q2q.train \\
        --config configs/experiments/main_7b/llama2/q2q_10.yaml \\
        --tag Q2Q-Top10
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

try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        _tv_io.VideoReader = type("VideoReader", (), {})
except Exception:
    pass

import torch
from torch.utils.data import Subset

from tads.core.data_io import read_records_glob
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
from tads.data.sft_prompts import alpaca_input_part
from tads.modeling.loader import load_model, load_tokenizer
from tads.pipelines.sft import make_dataloader, sft_one_epoch

from .score import compute_ifd_scores, select_top_proportion_by_ifd

logger = logging.getLogger(__name__)


def _atomic_json_dump(obj, path) -> None:
    """tmp + fsync + rename — same pattern as tads.train and nait seed cache.
    Prevents a kill mid-`json.dump` from leaving a truncated cache file the
    resume path then chokes on."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


_ALPACA_PROMPT_PREFIX = (
    "Below is an instruction that describes a task"
    "{input_part}. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def _make_alpaca_prefix(ins: str) -> str:
    return _ALPACA_PROMPT_PREFIX.format(
        input_part=alpaca_input_part(""),
        instruction=ins,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="Q2Q", help="Variant tag (e.g. Q2Q-Top10).")
    return p.parse_args()


def _load_alpaca_raw(spec: str) -> list:
    """Load Alpaca-GPT4 raw records via the shared JSON/JSONL/Parquet helper."""
    return read_records_glob(spec)


def _extract_ins_res(rec) -> tuple:
    if "instruction" in rec and "output" in rec:
        ins = rec["instruction"]
        inp = rec.get("input") or ""
        if inp:
            ins = f"{ins}\n{inp}"
        return ins, rec["output"]
    if "conversations" in rec and isinstance(rec["conversations"], list):
        c = rec["conversations"]
        if len(c) < 2:
            raise ValueError(
                f"conversations record needs >=2 turns; got {len(c)}"
            )

        def _txt(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return v.get("value") or v.get("content") or ""
            return str(v)

        return _txt(c[0]), _txt(c[1])
    raise KeyError(f"No instruction/output in record; keys={list(rec.keys())}")


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
    output_dir = Path(cfg["output_root"]) / cfg["output_subdir"] / f"q2q_{tag_slug}"
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(log_dir), name=f"q2q_{tag_slug}_{ts}")
    logger.info("Q2Q baseline | tag=%s | output_dir=%s", args.tag, output_dir)

    timer = PhaseTimer(log=logger, method="q2q")

    q2q_cfg = cfg.get("q2q", {}) or {}
    ifd_low = float(q2q_cfg.get("ifd_low", 0.5))
    ifd_high = float(q2q_cfg.get("ifd_high", 1.0))
    proportion = float(cfg.get("selection_ratio", 0.1))

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
            f"Q2Q: raw ({len(raw_records)}) vs tokenised dataset "
            f"({len(dataset)}) length mismatch."
        )
    logger.info("Alpaca-GPT4 size: %d", len(dataset))

    # ---------- IFD scoring (resumable) ----------
    selected_path = output_dir / "selected_indices.json"
    scores_path = output_dir / "ifd_scores.json"
    if selected_path.exists() and scores_path.exists():
        logger.info("RESUME: loading cached selection from %s", selected_path)
        with open(selected_path) as f:
            selected_indices = json.load(f)
    else:
        instructions, responses = [], []
        for rec in raw_records:
            ins, res = _extract_ins_res(rec)
            instructions.append(ins)
            responses.append(res)

        logger.info(
            "Q2Q IFD scoring | proportion=%.3f | ifd_low=%.2f | ifd_high=%.2f | n=%d",
            proportion, ifd_low, ifd_high, len(instructions),
        )
        with timer.phase("q2q.ifd_scoring", "selection"):
            scores = compute_ifd_scores(
                model=model,
                tokenizer=tokenizer,
                device=device,
                instructions=instructions,
                responses=responses,
                prompt_format=_make_alpaca_prefix,
                max_length=int(cfg.get("max_seq_len", 2048)),
            )
        with timer.phase("q2q.topk_filter", "selection"):
            # Atomic write so a kill mid-dump doesn't leave a truncated JSON
            # that the resume path then chokes on.
            _atomic_json_dump(scores, scores_path)
            selected_indices = select_top_proportion_by_ifd(
                scores, proportion=proportion, ifd_low=ifd_low, ifd_high=ifd_high,
            )
            _atomic_json_dump(selected_indices, selected_path)
        logger.info("Selected %d / %d", len(selected_indices), len(scores))

    # ---------- SFT on selected subset ----------
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
        logger.info("=== Q2Q epoch %d/%d ===", epoch, train_epochs)
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
    logger.info("Q2Q training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
