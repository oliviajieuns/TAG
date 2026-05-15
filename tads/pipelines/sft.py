"""SFT loop — single epoch over a selected subset, DDP-aware."""
from __future__ import annotations

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
    num_workers: int = 2,
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

    # Align every rank at the SFT entry point so the no_sync window cannot
    # span a "rank 0 already on step 2 while rank 3 still finishing forward
    # of step 0" situation. Without this, asymmetric arrival times leak
    # into the first grad_accum boundary (step 3 by default) where DDP
    # all-reduce expects every rank to have the same bucket state — a
    # known hang trigger when SFT prints "step=0" once and then stalls.
    r = _rank()
    ws = world_size()
    if dist.is_initialized() and ws > 1:
        t0_barrier = time.time()
        dist.barrier()
        logger.info(
            "SFT entry barrier | rank=%d | epoch=%d | wait=%.2fs",
            r, epoch, time.time() - t0_barrier,
        )

    # DDP grad-accum: skip the all-reduce on intermediate micro-batches with
    # model.no_sync(), and only sync on the boundary step that actually calls
    # optimizer.step(). With grad_accum=4 this cuts inter-GPU communication
    # by ~4x. The context manager is a no-op for non-DDP modules.
    no_sync_cm = getattr(model, "no_sync", None)
    n_batches = len(loader)
    logger.info(
        "SFT loop start | rank=%d | epoch=%d | n_batches=%d | grad_accum=%d "
        "| no_sync_avail=%s | %s",
        r, epoch, n_batches, grad_accum, no_sync_cm is not None, cuda_mem_str(),
    )

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
            o = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (o.loss / grad_accum).backward()
            return o

        if (not is_boundary) and no_sync_cm is not None:
            with no_sync_cm():
                out = _forward_backward()
        else:
            out = _forward_backward()

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
            optimizer.zero_grad()
            if verbose_step:
                logger.info(
                    "SFT step optimizer.step done | rank=%d | step=%d | "
                    "lr=%.2e | %s",
                    r, step, scheduler.get_last_lr()[0], cuda_mem_str(),
                )

        if is_main_process() and (verbose_step or step % log_every == 0):
            logger.info(
                "SFT | epoch=%d | step=%d/%d | loss=%.4f | lr=%.2e | %s",
                epoch, step, n_batches,
                out.loss.item(), scheduler.get_last_lr()[0], cuda_mem_str(),
            )

    # Aggregate the mean per-step loss across DDP ranks so the returned
    # number is a true global mean rather than a single rank's view.
    mean_loss = total_loss / max(1, n_steps)
    if dist.is_initialized() and world_size() > 1:
        t = torch.tensor([mean_loss], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        mean_loss = (t.item() / world_size())
    return mean_loss
