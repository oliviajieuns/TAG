"""Distributed utilities for TAG selection.

Fixes the _broadcast_selection bug class:
  * is_main_process() vs rank==0 mismatch (local vs global predicates)
  * dist.broadcast called inside a rank-gated branch (NCCL collective deadlock)
  * top-B taken over a per-shard score tensor instead of the full pool

Single source of truth for "am I the source for selection broadcast?":
    rank == 0   (== is_global_main()).
NEVER use accelerator.is_local_main_process, transformers.is_local_process_zero,
or os.environ["LOCAL_RANK"] == "0" here — they are True on every node's local
rank 0 in multi-node setups and disagree with dist.broadcast(src=0).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# rank / world helpers
# ---------------------------------------------------------------------------

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """Global rank across all nodes. The ONLY rank notion used in this module."""
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_global_main() -> bool:
    """True iff this process is global rank 0. Equivalent to rank == 0.

    Use this — NOT accelerator.is_local_main_process — wherever the predicate
    must agree with dist.broadcast(..., src=0).
    """
    return get_global_rank() == 0


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def log_rank_predicates(extra: Optional[Callable[[], bool]] = None) -> None:
    """One-shot diagnostic. Call once after init_process_group.

    Prints global rank, local rank, world size, and the value of is_global_main()
    on every process. Used to verify that any codebase-local is_main_process()
    helper agrees with rank == 0 on every node.
    """
    rank = get_global_rank()
    world = get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    node = os.environ.get("GROUP_RANK", "0")
    extra_str = "" if extra is None else f" extra_pred={extra()}"
    print(
        f"[tag.dist] node={node} rank={rank} local_rank={local} "
        f"world={world} is_global_main={is_global_main()}{extra_str}",
        flush=True,
    )
    if is_dist_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# gather (pool scores across DDP shards)
# ---------------------------------------------------------------------------

@torch.no_grad()
def all_gather_concat(local_tensor: torch.Tensor) -> torch.Tensor:
    """Gather per-rank 1-D tensors of EQUAL length into a single (world*K,) tensor
    that is identical on every rank.

    Assumes every rank holds `local_tensor.shape[0] == K` for the same K.
    Pad your pool shards upstream if N is not divisible by world_size; otherwise
    NCCL will reject the all_gather with a shape-mismatch error.

    Use this BEFORE broadcast_selection if your forward over the candidate pool
    is sharded across DDP ranks (the common case for the selection Step 1 forward).
    """
    if not is_dist_initialized():
        return local_tensor
    local_tensor = local_tensor.contiguous()
    ws = get_world_size()
    buf = [torch.empty_like(local_tensor) for _ in range(ws)]
    dist.all_gather(buf, local_tensor)
    return torch.cat(buf, dim=0)


# ---------------------------------------------------------------------------
# broadcast_selection — the fixed version
# ---------------------------------------------------------------------------

@torch.no_grad()
def broadcast_selection(
    scores: torch.Tensor,
    B: int,
    *,
    src_global_rank: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute top-B on `src_global_rank` and broadcast the index tensor to all
    ranks. Returns an identical (B,) int64 tensor on every rank.

    Contract (any violation = silent corruption or deadlock):

      1. ALL ranks in the default process group MUST call this function. Do not
         wrap the call site in `if is_main_process(): ...`. dist.broadcast is a
         collective.
      2. `scores` on the source rank must be the FULL pool score tensor (length N,
         not N/world_size). all_gather_concat() it upstream if needed.
      3. The predicate "am I the source" is rank == src_global_rank, where rank
         is the GLOBAL rank. Local-flavor predicates (is_local_main_process,
         LOCAL_RANK == 0) are forbidden in this function and at its call site.
      4. The broadcast tensor is detached, contiguous, int64, on the calling
         rank's device. Mismatch in any of these → NCCL error.
    """
    if device is None:
        device = scores.device
    rank = get_global_rank()

    if rank == src_global_rank:
        # Source rank: compute the actual selection.
        if scores.dim() != 1:
            raise ValueError(
                f"broadcast_selection expects 1-D scores, got shape {tuple(scores.shape)}"
            )
        if B > scores.numel():
            raise ValueError(
                f"B={B} exceeds pool size N={scores.numel()}; "
                "did you forget all_gather_concat()?"
            )
        S_t = (
            torch.topk(scores, B).indices
            .detach()
            .to(device=device, dtype=torch.long)
            .contiguous()
        )
    else:
        # Non-source rank: allocate matching buffer to receive the broadcast.
        S_t = torch.empty(B, dtype=torch.long, device=device)

    if is_dist_initialized():
        dist.broadcast(S_t, src=src_global_rank)

    return S_t
