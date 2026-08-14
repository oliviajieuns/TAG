"""Minimal DDP smoke test — isolates the bug from our pipeline.

If THIS script hangs/fails, the cluster's distributed stack is broken and
no amount of application-side code change will help. If it passes, the bug is
inside our training pipeline (model loading, gradient_checkpointing,
PEFT/bnb integration, etc.) and we can localize from there.

Run with torchrun (4 GPU, mirrors our real launch):

    torchrun --standalone --nproc-per-node=4 tests/test_ddp_smoke.py

Or with a specific backend (nccl / gloo):

    TAG_DDP_BACKEND=gloo torchrun --standalone --nproc-per-node=4 \\
        tests/test_ddp_smoke.py

Pass criteria — all four log lines visible within ~30 seconds:

    [rank 0] init OK | backend=nccl | world_size=4
    [rank 0] forward OK | loss=...
    [rank 0] backward+all_reduce OK
    [rank 0] PASS

If you see "init OK" + "forward OK" but never "backward+all_reduce OK"
the DDP gradient sync is dead — same signature as the production hang
but in a 1-MB model, so it's the cluster, not us.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn


def _log(msg: str, rank: int = -1) -> None:
    if rank == -1:
        rank = int(os.environ.get("RANK", "0"))
    print(f"[rank {rank}] {msg}", flush=True)


def main() -> None:
    backend = os.environ.get("TAG_DDP_BACKEND", "nccl").lower()
    if backend not in ("nccl", "gloo"):
        backend = "nccl"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    _log(f"about to init_process_group | backend={backend} | "
         f"local_rank={local_rank} | world_size={world_size}", rank)

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=10))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available() else torch.device("cpu")
    )
    _log(f"init OK | backend={backend} | world_size={world_size}", rank)

    # ---- toy model: 1024 -> 1024 -> 1024 (~3M params, bf16) -----------------
    model = nn.Sequential(
        nn.Linear(1024, 1024, bias=False),
        nn.GELU(),
        nn.Linear(1024, 1024, bias=False),
        nn.GELU(),
        nn.Linear(1024, 1024, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    _log("DDP wrap OK", rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ---- forward / backward / step round-trip --------------------------------
    for step in range(3):
        t0 = time.time()
        x = torch.randn(8, 1024, device=device, dtype=torch.bfloat16)
        y = model(x)
        loss = y.float().pow(2).mean()
        _log(f"forward OK | step={step} | loss={loss.item():.4f} "
             f"| dt={time.time() - t0:.2f}s", rank)
        t0 = time.time()
        loss.backward()
        if dist.is_initialized() and world_size > 1:
            # extra explicit all_reduce so we can see it succeed independent
            # of DDP's internal gradient sync.
            probe = torch.tensor([float(rank)], device=device)
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
            expected = float(sum(range(world_size)))
            if abs(probe.item() - expected) > 1e-3:
                _log(f"FAIL | explicit all_reduce wrong | got={probe.item()} "
                     f"expected={expected}", rank)
                dist.destroy_process_group()
                sys.exit(1)
        _log(f"backward+all_reduce OK | step={step} | dt={time.time() - t0:.2f}s",
             rank)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    dist.barrier()
    if rank == 0:
        _log("PASS — DDP forward+backward+all_reduce works on this cluster.",
             rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
