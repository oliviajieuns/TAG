"""Distributed unit test for broadcast_selection / all_gather_concat.

Run with 2 GPUs (or CPU+gloo):
    torchrun --standalone --nproc-per-node=2 tests/test_broadcast_selection.py

The test deliberately perturbs rank>0's score tensor so that, if the broadcast
is wrong, the resulting S_t will differ across ranks or won't equal rank-0's
expected top-B. A correct implementation returns identical S_t on every rank
equal to the top-B over rank-0's scores.

Three failure modes this test catches:

  (A) `_broadcast_selection` runs `dist.broadcast` inside `if is_main_process():`
      → non-source ranks hang on the next collective. PyTest will time out
      (default NCCL_BLOCKING_WAIT=10min). Run with NCCL_DEBUG=INFO to see the
      stalled rank.

  (B) Predicate is local-flavor (e.g., LOCAL_RANK==0) but src=0 is global.
      On single-node this still passes; on multi-node, run with 2 nodes to
      surface the divergence.

  (C) Source rank takes top-B over its local shard instead of the full pool.
      The pool here is uniform across ranks (each rank holds the same N),
      so this test does NOT cover the shard-vs-full issue — see
      `test_all_gather_concat` below for that.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

try:
    import pytest
except ImportError:  # Keep direct `torchrun ... test_broadcast_selection.py` usable.
    pytest = None

# Allow `python tests/test_broadcast_selection.py` from repo root without install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tag.core.dist_utils import (  # noqa: E402
    all_gather_concat,
    broadcast_selection,
    get_global_rank,
    get_world_size,
    is_dist_initialized,
)


def _setup() -> torch.device:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _teardown() -> None:
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _has_torchrun_env() -> bool:
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


if pytest is not None:

    @pytest.fixture(scope="module")
    def device():
        if not _has_torchrun_env():
            pytest.skip(
                "distributed test requires torchrun, e.g. "
                "`torchrun --standalone --nproc-per-node=2 tests/test_broadcast_selection.py`",
            )
        dev = _setup()
        try:
            yield dev
        finally:
            _teardown()


def test_broadcast_selection_uses_rank0_scores(device: torch.device) -> None:
    """Rank 0's top-B must win regardless of what other ranks hold."""
    rank = get_global_rank()
    N, B = 100, 5

    # Rank 0 has ascending scores [0, 1, ..., 99]; expected top-B = [99,98,97,96,95].
    # Other ranks have a wildly different score profile; if the broadcast leaks
    # their selection, S_t will not match the expected tensor.
    scores = torch.arange(N, dtype=torch.float32, device=device)
    if rank != 0:
        scores = scores.flip(0) + 10_000.0   # would prefer indices [0,1,2,3,4]

    S_t = broadcast_selection(scores, B, src_global_rank=0, device=device)

    expected = torch.tensor([99, 98, 97, 96, 95], dtype=torch.long, device=device)
    assert S_t.dtype == torch.long, f"S_t dtype is {S_t.dtype}, expected int64"
    assert S_t.device == device, f"S_t device is {S_t.device}, expected {device}"
    assert torch.equal(S_t, expected), (
        f"rank {rank}: S_t = {S_t.tolist()}, expected {expected.tolist()}. "
        "Selection was not taken from global rank 0."
    )
    print(f"[ok] broadcast: rank={rank} S_t={S_t.tolist()}", flush=True)


def test_all_gather_concat_stitches_shards(device: torch.device) -> None:
    """all_gather_concat must return identical (world*K,) tensor on every rank."""
    rank = get_global_rank()
    ws = get_world_size()
    K = 4

    local = torch.arange(rank * K, (rank + 1) * K, dtype=torch.float32, device=device)
    full = all_gather_concat(local)

    expected = torch.arange(0, ws * K, dtype=torch.float32, device=device)
    assert full.shape == (ws * K,), f"shape {full.shape}, expected {(ws * K,)}"
    assert torch.equal(full, expected), (
        f"rank {rank}: full = {full.tolist()}, expected {expected.tolist()}"
    )
    print(f"[ok] all_gather: rank={rank} full[:4]={full[:4].tolist()}", flush=True)


def main() -> None:
    device = _setup()
    try:
        test_broadcast_selection_uses_rank0_scores(device)
        test_all_gather_concat_stitches_shards(device)
        if get_global_rank() == 0:
            print("PASS: broadcast_selection + all_gather_concat", flush=True)
    finally:
        _teardown()


if __name__ == "__main__":
    main()
