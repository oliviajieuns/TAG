"""SFT loop — single epoch over a selected subset, DDP-aware."""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..core.utils import cuda_mem_str, is_main_process, world_size

logger = logging.getLogger(__name__)


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
) -> DataLoader:
    """Deterministic dataloader, DDP-aware when sampler is None and dist is up."""
    g = torch.Generator()
    g.manual_seed(seed)

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
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)

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

    # DDP grad-accum: skip the all-reduce on intermediate micro-batches with
    # model.no_sync(), and only sync on the boundary step that actually calls
    # optimizer.step(). With grad_accum=4 this cuts inter-GPU communication
    # by ~4x. The context manager is a no-op for non-DDP modules.
    no_sync_cm = getattr(model, "no_sync", None)

    for step, batch in enumerate(loader):
        is_boundary = ((step + 1) % grad_accum == 0) or ((step + 1) == len(loader))

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

        total_loss += out.loss.item()
        n_steps += 1

        if is_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if is_main_process() and step % log_every == 0:
            logger.info(
                "SFT | epoch=%d | step=%d/%d | loss=%.4f | lr=%.2e | %s",
                epoch, step, len(loader),
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
