"""Learning-rate schedulers — vendored from HF transformers.

`transformers.get_cosine_schedule_with_warmup` used to be a stable public API,
but 5.0 reorganised the optimization module and broke `from transformers
import …`. We don't actually need anything from transformers here — the
function is a thin wrapper around `torch.optim.lr_scheduler.LambdaLR`. Vendor
it so our training loop is decoupled from transformers' version churn, then
add two trainer-specific correctness controls: an optional terminal LR floor
and exact DDP optimiser-step planning.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def optimizer_steps_per_epoch(
    n_samples: int,
    batch_size: int,
    grad_accum: int,
    world_size: int = 1,
) -> int:
    """Exact optimiser-step count for the trainer's DDP data layout.

    ``DistributedSampler(drop_last=False)`` first pads every rank to
    ``ceil(n_samples / world_size)`` records.  The DataLoader and gradient
    accumulator then each keep their final partial group, so both divisions
    must also use ``ceil``.  The old trainer floored one combined division;
    for the Table-2 TAG cell that planned 120 steps although it executes 123,
    leaving the final updates at a zero learning rate.
    """
    values = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "world_size": world_size,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    per_rank_samples = math.ceil(int(n_samples) / int(world_size))
    per_rank_batches = math.ceil(per_rank_samples / int(batch_size))
    return math.ceil(per_rank_batches / int(grad_accum))


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup → cosine decay to a configurable LR floor.

    Args:
        optimizer: torch / bitsandbytes optimizer (anything LambdaLR accepts).
        num_warmup_steps: steps to linearly warm up from 0 → base_lr.
        num_training_steps: total optimizer steps; cosine decays over the
            remaining `num_training_steps - num_warmup_steps` steps.
        num_cycles: 0.5 = single half-cosine to zero (default; matches
            transformers' behaviour). 1.0 would oscillate.
        last_epoch: passed to LambdaLR for resuming.
        min_lr_ratio: final LR divided by the optimiser's base LR.  ``0``
            preserves the historical schedule; ``0.1`` keeps late dynamic
            selections trainable instead of decaying their updates to zero.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the cosine-with-warmup schedule.
    """
    if not (0.0 <= float(min_lr_ratio) <= 1.0):
        raise ValueError(
            f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}",
        )

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Exact step planning should keep progress <= 1.  Clamp anyway so a
        # resumed/extended run cannot enter the next cosine lobe and silently
        # raise the LR again after reaching its planned endpoint.
        progress = min(max(progress, 0.0), 1.0)
        cosine = max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
        floor = float(min_lr_ratio)
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch)
