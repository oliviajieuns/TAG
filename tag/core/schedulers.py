"""Learning-rate schedulers — vendored from HF transformers.

`transformers.get_cosine_schedule_with_warmup` used to be a stable public API,
but 5.0 reorganised the optimization module and broke `from transformers
import …`. We don't actually need anything from transformers here — the
function is a thin wrapper around `torch.optim.lr_scheduler.LambdaLR`. Vendor
it so our training loop is decoupled from transformers' version churn.

The function is byte-equivalent to the implementation in
`transformers/src/transformers/optimization.py` (Apache-2.0).
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup → cosine decay to 0.

    Args:
        optimizer: torch / bitsandbytes optimizer (anything LambdaLR accepts).
        num_warmup_steps: steps to linearly warm up from 0 → base_lr.
        num_training_steps: total optimizer steps; cosine decays over the
            remaining `num_training_steps - num_warmup_steps` steps.
        num_cycles: 0.5 = single half-cosine to zero (default; matches
            transformers' behaviour). 1.0 would oscillate.
        last_epoch: passed to LambdaLR for resuming.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the cosine-with-warmup schedule.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
