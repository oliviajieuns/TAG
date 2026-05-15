"""SFT loop — single epoch over a selected subset, DDP-aware."""
from __future__ import annotations

import gc
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..core.utils import cuda_mem_str, is_main_process, rank as _rank, world_size

logger = logging.getLogger(__name__)


# Always log the first N steps (regardless of log_every) so a hang in the
# no_sync window or at the first grad_accum boundary is visible immediately.
_ALWAYS_LOG_FIRST = 10


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _collate(batch):
    return {
        "input_ids": torch.stack([torch.as_tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack(
            [torch.as_tensor(x["attention_mask"]) for x in batch],
        ),
        "labels": torch.stack([torch.as_tensor(x["labels"]) for x in batch]),
    }


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    sampler: Optional[object] = None,
    epoch: int = 0,
) -> DataLoader:
    """Deterministic dataloader, DDP-aware when sampler is None and dist is up.

    ``epoch`` is folded into the generator seed so single-GPU runs (where
    DistributedSampler is bypassed and its ``set_epoch`` mechanism doesn't
    apply) still produce a different shuffle order each epoch. Without
    this, callers that re-construct the loader once per epoch — like the
    main trainer — would see the exact same shuffled batch sequence on
    epoch 1, 2, 3, etc., silently undoing shuffle entirely. DDP runs are
    unaffected because the DistributedSampler branch sets its own seed
    and the trainer calls ``sampler.set_epoch(epoch)`` separately.
    """
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)

    if sampler is None and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False

    def _seed_worker(worker_id: int) -> None:
        random.seed(seed + epoch * 100 + worker_id)
        np.random.seed(seed + epoch * 100 + worker_id)

    # Allow disabling the DataLoader's background workers via env to rule
    # them out as a hang source. Set TADS_DL_NUM_WORKERS=0 to keep loading
    # in the main process — slightly slower but the background worker
    # pinning / pickling path is a known DDP hang surface.
    _nw_env = os.environ.get("TADS_DL_NUM_WORKERS")
    if _nw_env is not None:
        try:
            num_workers = int(_nw_env)
        except ValueError:
            pass

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g if sampler is None else None,
        collate_fn=_collate,
    )


def sft_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    grad_accum: int,
    grad_clip: float,
    device,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    log_every: int = 50,
) -> float:
    """Run one SFT epoch and return the mean per-step loss."""
    if logger is None:
        logger = logging.getLogger(__name__)
    model.train()

    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

    total_loss = 0.0
    n_steps = 0
    optimizer.zero_grad()

    # NO dist.barrier here. After rank 0's solo collect_episode the NCCL
    # communicator can hang on the next collective even when every rank
    # arrives. We rely on the FIRST forward / backward's all_reduce to
    # naturally align ranks. Print each rank's entry so a real hang is
    # visible without going through a fragile collective.
    r = _rank()
    ws = world_size()
    print(
        f"[sft] rank={r} ENTER sft_one_epoch | epoch={epoch} | "
        f"ws={ws} | mem={cuda_mem_str()}",
        flush=True,
    )

    # DDP grad-accum: skip the all-reduce on intermediate micro-batches with
    # model.no_sync(), and only sync on the boundary step that actually calls
    # optimizer.step(). The context manager is a no-op for non-DDP modules.
    #
    # Default is OFF (every step syncs) because no_sync has been linked to
    # rare-but-real DDP hangs on the very first grad_accum boundary — exactly
    # the symptom user reported ("SFT prints step=0 once then stalls"). To
    # restore the ~4x communication savings when the run is known stable,
    # set TADS_ENABLE_NO_SYNC=1 before launching.
    no_sync_enabled = _env_truthy("TADS_ENABLE_NO_SYNC")
    no_sync_cm = getattr(model, "no_sync", None) if no_sync_enabled else None
    n_batches = len(loader)
    logger.info(
        "SFT loop start | rank=%d | epoch=%d | n_batches=%d | grad_accum=%d "
        "| no_sync_enabled=%s | %s",
        r, epoch, n_batches, grad_accum, no_sync_enabled, cuda_mem_str(),
    )

    # Free anything left over from collect_episode / agent.update so the SFT
    # forward starts on a clean allocator and the first step's memory peak
    # isn't competing with stale CPU buffers.
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:  # cleanup is best-effort
        logger.warning("Pre-SFT cleanup hiccup: %s", exc)

    # First few optimizer.step boundaries also get an explicit dist.barrier
    # after the all-reduce, so a desync across ranks surfaces immediately
    # rather than as a silent stall on the next collective. After three
    # successful boundaries we trust the pipeline.
    _ALWAYS_BARRIER_FIRST_N_BOUNDARIES = 3
    n_boundaries_seen = 0

    for step, batch in enumerate(loader):
        is_boundary = ((step + 1) % grad_accum == 0) or ((step + 1) == n_batches)
        verbose_step = step < _ALWAYS_LOG_FIRST

        if verbose_step:
            logger.info(
                "SFT step entry | rank=%d | epoch=%d | step=%d/%d | "
                "is_boundary=%s | %s",
                r, epoch, step, n_batches, is_boundary, cuda_mem_str(),
            )

        def _forward_backward():
            # non_blocking=True lets the H→D copy overlap with the prior
            # micro-batch's forward. Only effective when DataLoader uses
            # pin_memory=True (it does — see make_dataloader). Without the
            # flag the CPU stalls on every transfer and pin_memory is wasted.
            o = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True),
            )
            (o.loss / grad_accum).backward()
            return o

        if (not is_boundary) and no_sync_cm is not None:
            with no_sync_cm():
                out = _forward_backward()
        else:
            out = _forward_backward()

        # Force any pending CUDA work to drain on the first few steps so a
        # silent CUDA error (illegal memory access, NaN-only state, etc.)
        # surfaces as a Python exception here instead of as a downstream
        # NCCL hang. Keeps the wall-time impact bounded — only first 10.
        if verbose_step and torch.cuda.is_available():
            torch.cuda.synchronize()

        if verbose_step:
            logger.info(
                "SFT step backward done | rank=%d | step=%d | loss=%.4f | %s",
                r, step, out.loss.item(), cuda_mem_str(),
            )

        total_loss += out.loss.item()
        n_steps += 1

        if is_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            # set_to_none=True frees the .grad tensors entirely rather than
            # zeroing them in place; less memory pressure between steps and
            # safer with bitsandbytes 8-bit optimisers.
            optimizer.zero_grad(set_to_none=True)
            n_boundaries_seen += 1
            if verbose_step:
                logger.info(
                    "SFT step optimizer.step done | rank=%d | step=%d | "
                    "lr=%.2e | %s",
                    r, step, scheduler.get_last_lr()[0], cuda_mem_str(),
                )
            # Explicit barrier on the first few boundaries: every rank pings
            # before continuing. A stall here points at a real DDP desync
            # (not a slow batch). After this many successful boundaries we
            # trust the pipeline and skip the extra barrier.
            if (
                n_boundaries_seen <= _ALWAYS_BARRIER_FIRST_N_BOUNDARIES
                and dist.is_initialized()
                and world_size() > 1
            ):
                t_b = time.time()
                dist.barrier()
                logger.info(
                    "SFT boundary barrier #%d | rank=%d | step=%d | wait=%.2fs",
                    n_boundaries_seen, r, step, time.time() - t_b,
                )

        if is_main_process() and (verbose_step or step % log_every == 0):
            logger.info(
                "SFT | epoch=%d | step=%d/%d | loss=%.4f | lr=%.2e | %s",
                epoch, step, n_batches,
                out.loss.item(), scheduler.get_last_lr()[0], cuda_mem_str(),
            )

    logger.info(
        "SFT loop done | rank=%d | epoch=%d | n_steps=%d | %s",
        r, epoch, n_steps, cuda_mem_str(),
    )

    # Aggregate the mean per-step loss across DDP ranks so the returned
    # number is a true global mean rather than a single rank's view.
    mean_loss = total_loss / max(1, n_steps)
    if dist.is_initialized() and world_size() > 1:
        # Log entry into the collective so a stall here is distinguishable
        # from a stall inside the loop. The communicator's timeout
        # (NCCL_TIMEOUT 120 min via train.py) bounds the worst case.
        logger.info(
            "SFT final loss all_reduce | rank=%d | local_mean=%.4f",
            r, mean_loss,
        )
        t = torch.tensor([mean_loss], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        mean_loss = (t.item() / world_size())
        logger.info(
            "SFT final loss all_reduce DONE | rank=%d | global_mean=%.4f",
            r, mean_loss,
        )
    return mean_loss
