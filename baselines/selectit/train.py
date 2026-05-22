"""SelectIT (Liu et al., 2024b) baseline training.

Faithful port of github.com/Blue-Raincoat/SelectIT — uncertainty-aware
self-reflection scoring of every Alpaca example, top-K by score, then SFT
on the selected subset.

Pipeline (mirrors baselines/nait/train.py):
    1. Load Alpaca-GPT4 (HF Dataset, tokenised) AND its raw records (raw
       (instruction, output) text needed by SelectIT scoring).
    2. Run forward-only scoring per sample using `selectit_scores` (token
       or sentence level).
    3. Top-`selection_ratio` indices → SFT one-epoch loop with the shared
       `tads.pipelines.sft.sft_one_epoch`.

Usage:
    source scripts/setup_env.sh
    CUDA_VISIBLE_DEVICES=0 python -m baselines.selectit.train \\
        --config configs/experiments/main_7b/llama2/selectit_10.yaml \\
        --tag SelectIT-Token

Eval (writes under SELECTIT_EVAL_RESULTS_ROOT):
    python -m tads.eval --config <same cfg> \\
        --ckpt ${OUTPUT_ROOT}/main_7b/llama2/selectit_10/selectit_selectit_token/epoch_3 \\
        --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \\
        --out_dir ${SELECTIT_EVAL_RESULTS_ROOT}/llama2/selectit_10/
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
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

# transformers 5.0+ 의 video / object-detection 모델 registry 가 import-time
# 에 `torch.ops.torchvision.nms` 를 조회. torch ↔ torchvision 버전 mismatch
# (또는 partial install) 인 경우 이 op 가 registry 에 없어서 모델 등록이
# 실패 → transformers 의 lazy loader 가 그 뒤 모든 *ForCausalLM lookup 을
# "could not import module 'XForCausalLM'" 으로 답하는 cascade 발생
# (Llama / Qwen / Mistral / DeepSeek 4개 모두 동일 증상).
# 본 stub 은 dummy nms 를 torch op registry 에 등록해서 cascade 차단.
# LLM SFT / scoring 코드는 NMS 호출 안 함 → safe.
try:
    _ = torch.ops.torchvision.nms  # probe; missing 시 RuntimeError
except (RuntimeError, AttributeError):
    try:
        torch.library.define(
            "torchvision::nms",
            "(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
        )

        def _stub_nms(dets, scores, iou_threshold):
            # NMS returns kept-box indices; empty default = "no detections".
            # 실제 NMS 가 필요한 caller 가 있으면 0개 결과로 graceful degrade
            # (cryptic op-registry 에러 대신).
            return torch.empty(0, dtype=torch.long, device=dets.device)

        for _backend in ("CPU", "CUDA", "Meta"):
            try:
                torch.library.impl("torchvision::nms", _backend)(_stub_nms)
            except Exception:
                pass
    except Exception:
        # torch.library 미지원 (torch <2.0) 또는 op 가 partial 등록된 상태 —
        # 그냥 그대로 진행해서 transformers 가 처리하게 둠
        pass
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
from tads.modeling.loader import load_model, load_tokenizer
from tads.pipelines.sft import make_dataloader, sft_one_epoch

from .score import select_top_proportion, selectit_scores


def _atomic_json_dump(obj, path) -> None:
    """tmp + fsync + rename so a kill mid-dump can't leave a truncated cache."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

logger = logging.getLogger(__name__)


_DEFAULT_RATING_PROMPTS = Path(__file__).parent / "rating_prompts.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--tag",
        default="SelectIT",
        help="Variant tag (e.g. SelectIT-Token, SelectIT-Sentence).",
    )
    p.add_argument(
        "--rating-prompts-file",
        default=str(_DEFAULT_RATING_PROMPTS),
        help="One rating prompt per line. Default: the 9 prompts shipped with this package.",
    )
    return p.parse_args()


def _load_rating_prompts(path: str) -> list:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No non-empty rating prompts in {path}")
    return lines


def _extract_ins_res(rec) -> tuple:
    """Pull (instruction, response) text from one raw Alpaca-GPT4 record.

    Alpaca-GPT4 canonical schema: {'instruction', 'input', 'output'}.
    Some mirrors use the ShareGPT-style {'conversations': [...]} shape —
    where each element can be a plain string OR a dict with a 'value' key.
    """
    if not isinstance(rec, dict):
        raise TypeError(f"raw record is not a dict: {type(rec)}")
    if "instruction" in rec and "output" in rec:
        ins = rec["instruction"]
        inp = rec.get("input") or ""
        if inp:
            ins = f"{ins}\n{inp}"
        return ins, rec["output"]
    if "conversations" in rec and isinstance(rec["conversations"], list):
        convs = rec["conversations"]
        if len(convs) < 2:
            raise ValueError(
                f"conversations record needs >=2 turns; got {len(convs)}"
            )

        def _txt(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return v.get("value") or v.get("content") or ""
            return str(v)

        return _txt(convs[0]), _txt(convs[1])
    raise KeyError(
        f"Cannot find instruction/output in record; keys={list(rec.keys())}"
    )


def _load_raw_records(data_files_spec: str) -> list:
    """Load raw records via the shared JSON / JSONL / Parquet helper."""
    return read_records_glob(data_files_spec)


def main() -> None:
    disable_coredumps()
    clear_runtime_caches()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)
    random.seed(seed)

    tag_slug = args.tag.lower().replace("-", "_")
    output_dir = (
        Path(cfg["output_root"]) / cfg["output_subdir"] / f"selectit_{tag_slug}"
    )
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", output_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(log_dir), name=f"selectit_{tag_slug}_{ts}")
    logger.info("SelectIT baseline | tag=%s | output_dir=%s", args.tag, output_dir)

    timer = PhaseTimer(log=logger, method="selectit")

    selectit_cfg = cfg.get("selectit", {}) or {}
    level = str(selectit_cfg.get("level", "token"))
    alpha = float(selectit_cfg.get("alpha", 0.2))
    proportion = float(cfg.get("selection_ratio", 0.1))

    # ---------- model + tokeniser ----------
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

    # ---------- dataset (tokenised) + raw records (for scoring text) ----------
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
        raw_records = _load_raw_records(str(cfg["data_files"]))
    if len(raw_records) != len(dataset):
        raise RuntimeError(
            f"SelectIT: raw records ({len(raw_records)}) and tokenised dataset "
            f"({len(dataset)}) length mismatch — per-index alignment broken. "
            f"Likely cause: HF datasets filtered or de-duplicated rows during "
            f"`build_alpaca_dataset`; the SelectIT scoring path assumes the "
            f"raw on-disk JSON order matches the tokenised Dataset order."
        )
    logger.info("Dataset size: %d (tokenised + raw aligned)", len(dataset))

    # ---------- score ----------
    rating_prompts = _load_rating_prompts(args.rating_prompts_file)
    logger.info(
        "SelectIT level=%s | proportion=%.3f | alpha=%.3f | rating_prompts=%d",
        level, proportion, alpha, len(rating_prompts),
    )

    selected_path = output_dir / "selected_indices.json"
    scores_path = output_dir / "scores.json"
    if selected_path.exists() and scores_path.exists():
        logger.info("RESUME: loading cached selection from %s", selected_path)
        with open(selected_path) as f:
            selected_indices = json.load(f)
    else:
        instructions = []
        responses = []
        for rec in raw_records:
            ins, res = _extract_ins_res(rec)
            instructions.append(ins)
            responses.append(res)

        with timer.phase(f"selectit.score_{level}", "selection"):
            scores = selectit_scores(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rating_templates=rating_prompts,
                instructions=instructions,
                responses=responses,
                level=level,
                alpha=alpha,
                max_length=int(cfg.get("max_seq_len", 2048)),
                batch_size=int(selectit_cfg.get("batch_size", 8)),
            )
        with timer.phase("selectit.topk", "selection"):
            _atomic_json_dump(scores, scores_path)
            selected_indices = select_top_proportion(scores, proportion=proportion)
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
        logger.info("=== SelectIT epoch %d/%d ===", epoch, train_epochs)
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
        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "selected": len(selected_indices),
        }
        metrics_log.append(metrics)
        logger.info("Epoch %d done | %s", epoch, metrics)

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
    logger.info("SelectIT training complete. Tag: %s", args.tag)


if __name__ == "__main__":
    main()
