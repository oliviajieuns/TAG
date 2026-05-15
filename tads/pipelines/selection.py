"""Per-epoch sample selection dispatch.

Wraps the four selection methods behind a single function. For data_agent
and tads the heavy collect_episode runs on rank 0 only; other ranks share
the resulting indices through a filesystem sentinel + poll, NOT through
an NCCL barrier — that was the deadlock that crashed runs after epoch 1.

Why polling, not dist.barrier:
    Rank 0 spends 30+ minutes inside collect_episode (52K samples × 32
    decoder layers × chunked rewards). While that runs, the other DDP
    ranks would be stuck inside dist.barrier() inside _broadcast_selection,
    and any of them hitting the NCCL collective watchdog (120 min default
    now, less previously) tears down the communicator. The next forward
    pass then fails on every rank, and rank 0 — which never reached the
    barrier — exits before saving any checkpoint.

    The fix is to remove the NCCL barriers from this path entirely. Rank 0
    writes the selection atomically (tmp + fsync + rename), then writes
    a separate `.ready` sentinel; workers poll on disk for the sentinel
    and read once it appears. The only collective in this module is a
    single barrier at the very end, after everyone has the data — so it
    always completes immediately.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..core.agent import PPOAgent
from ..core.selector import collect_episode
from ..core.trajectory_anchor import TrajectoryAnchor
from ..core.utils import is_main_process, local_rank, rank, world_size

logger = logging.getLogger(__name__)


# Workers poll this often while waiting for rank-0's collect_episode.
_POLL_INTERVAL_SEC = 2.0
# Hard ceiling on how long workers will wait. Set generously — episodes
# can legitimately take an hour at the 7B scale.
_POLL_TIMEOUT_SEC = 6 * 60 * 60  # 6 hours


def _random_indices(n_total, ratio, seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)
    perm = torch.randperm(n_total, generator=g).tolist()
    k = max(1, int(n_total * ratio))
    return perm[:k]


def _broadcast_selection(selected, *, epoch=0, output_dir=None):
    """File-poll selection share — no inter-write/read NCCL barrier.

    Every rank takes the same code path:
      - rank 0 atomically writes the indices, then atomically touches a
        `.ready` sentinel.
      - all other ranks poll for the sentinel on disk and read once it
        exists. They never call a collective while rank 0 is busy.
    A single dist.barrier() at the very end keeps the SFT phase in step
    even if some worker reads a few milliseconds before rank 0 exits its
    write — and it always completes immediately because everyone has
    already converged here.
    """
    r = dist.get_rank() if dist.is_initialized() else 0

    if not dist.is_initialized():
        if hasattr(selected, "tolist"):
            return selected.tolist()
        return list(selected) if not isinstance(selected, list) else selected

    SRC = 0
    base = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    sel_path = base / f"_selection_epoch{epoch}.json"
    ready_path = base / f"_selection_epoch{epoch}.ready"
    ready_tmp = base / f"_selection_epoch{epoch}.ready.tmp"

    if r == SRC:
        # Clean up any stale sentinel from a previous run before writing.
        for stale in (ready_path, ready_tmp):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

        if hasattr(selected, "tolist"):
            selected = selected.tolist()
        elif not isinstance(selected, list):
            selected = list(selected)
        selected = [int(x) for x in selected]
        logger.info(
            "[sel-share] rank=0 normalized selection | len=%d | first5=%s",
            len(selected), selected[:5],
        )

        # 1) atomic write of the selection itself.
        tmp_path = sel_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(selected, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, sel_path)

        # 2) atomic write of the `.ready` sentinel — workers only start
        # reading once this exists, so we never race a half-written
        # selection file.
        with open(ready_tmp, "w") as f:
            f.write(str(epoch))
            f.flush()
            os.fsync(f.fileno())
        os.replace(ready_tmp, ready_path)

        logger.info(
            "[sel-share] rank=0 WROTE %s (%d ids) + ready sentinel",
            sel_path, len(selected),
        )
        result = selected
    else:
        # Workers poll on disk. No NCCL collective during the wait, so
        # rank 0's long collect_episode can't trigger a collective watchdog.
        t_start = time.time()
        last_log = t_start
        while not ready_path.exists():
            now = time.time()
            if now - t_start > _POLL_TIMEOUT_SEC:
                raise RuntimeError(
                    f"[rank {r}] timed out after "
                    f"{int(now - t_start)}s waiting for "
                    f"{ready_path}. Rank 0 likely crashed before writing.",
                )
            if now - last_log > 60.0:
                logger.info(
                    "[sel-share] rank=%d polling for selection (%.0fs elapsed)",
                    r, now - t_start,
                )
                last_log = now
            time.sleep(_POLL_INTERVAL_SEC)

        if not sel_path.exists():
            raise RuntimeError(
                f"[rank {r}] ready sentinel present but selection file "
                f"{sel_path} missing.",
            )
        with open(sel_path, "r") as f:
            result = json.load(f)
        if not isinstance(result, list):
            raise RuntimeError(
                f"[rank {r}] selection file has wrong shape: "
                f"type={type(result).__name__}",
            )
        logger.info(
            "[sel-share] rank=%d READ %s len=%d", r, sel_path, len(result),
        )

    # NO dist.barrier here. After rank 0's 30+ minute solo collect_episode
    # the NCCL communicator can be in a state where the next collective
    # hangs even when every rank reaches it — the communicator's
    # background socket has effectively died. Removing the barrier means
    # ranks proceed straight to SFT, and the very first DDP all_reduce
    # inside backward() doubles as the alignment point.
    print(
        f"[sel-share] rank={r} EXIT _broadcast_selection (no barrier)",
        flush=True,
    )

    if r == SRC:
        # Best-effort cleanup; missing files are fine.
        for stale in (sel_path, ready_path):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    return result


def select_indices(method, *, model, agent, anchor, dataset, cfg, epoch, seed, device):
    """Return (selected_indices, extras) for the given epoch."""
    n_total = len(dataset)
    ratio = float(cfg["selection_ratio"])
    extras = {}

    if method == "full":
        selected = list(range(n_total))
        logger.info("Full dataset selection | k=%d", len(selected))
        return selected, extras

    if method == "random":
        selected = _random_indices(n_total, ratio, seed, epoch)
        logger.info("Random selection | k=%d/%d", len(selected), n_total)
        return selected, extras

    if method not in ("tads", "data_agent"):
        raise ValueError("Unknown method: " + repr(method))

    # ---------- selection cache: skip collect_episode if a prior run
    # ---------- already produced selected_indices_epoch{N}.json
    # collect_episode for tads/data_agent takes 30+ min on 7B. If a previous
    # run made it through scoring but hung in the post-broadcast NCCL step,
    # the indices already exist on disk and we can reuse them directly. This
    # path is ONLY hit when the file is present; a fresh start still runs
    # the full episode.
    _output_dir_raw = cfg.get("output_dir") or cfg.get("output_root")
    if _output_dir_raw is not None:
        _cached_path = Path(_output_dir_raw) / f"selected_indices_epoch{epoch}.json"
        if _cached_path.exists():
            try:
                with open(_cached_path) as _f:
                    _cached = json.load(_f)
                if isinstance(_cached, list) and len(_cached) > 0:
                    logger.info(
                        "REUSING cached selection from %s (%d indices) — "
                        "skipping collect_episode for epoch %d.",
                        _cached_path, len(_cached), epoch,
                    )
                    # Broadcast the cached indices to all ranks via the same
                    # file-polling mechanism so workers also get them.
                    if is_main_process():
                        selected = [int(x) for x in _cached]
                    else:
                        selected = []
                    selected = _broadcast_selection(
                        selected, epoch=epoch,
                        output_dir=_output_dir_raw,
                    )
                    extras["selection_cache_reused"] = True
                    return selected, extras
            except Exception as _e:
                logger.warning(
                    "Could not reuse cached selection at %s (%s); "
                    "running full collect_episode.",
                    _cached_path, _e,
                )

    if is_main_process():
        print("[trace] rank=0 ENTER main branch | method=" + method
              + " | anchor=" + ("set" if anchor is not None else "None"), flush=True)
        import traceback as _tb
        try:
            if method == "tads" and anchor is not None:
                logger.info("Updating trajectory anchor ...")
                print("[trace] rank=0 BEFORE anchor.update", flush=True)
                anchor_stats = anchor.update(
                    model=model, dataset=dataset, seed=seed, epoch=epoch,
                )
                _akeys = list(anchor_stats.keys()) if anchor_stats else None
                print("[trace] rank=0 AFTER anchor.update | stats_keys="
                      + str(_akeys), flush=True)
                extras["anchor_stats"] = anchor_stats

            tads_cfg = cfg.get("tads", {}) or {}
            exp_tag = str(cfg.get("model_key", "?")) + "/alpaca/" + method

            print("[trace] rank=0 BEFORE collect_episode", flush=True)
            episode = collect_episode(
                model=model,
                agent=agent,
                dataset=dataset,
                selection_ratio=ratio,
                trajectory_anchor=anchor if method == "tads" else None,
                lam=float(tads_cfg.get("lam", 0.0)),
                use_anchor=bool(tads_cfg.get("use_anchor", False)) and method == "tads",
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(device),
                seed=seed,
                epoch=epoch,
                exp_tag=exp_tag,
            )
            print("[trace] rank=0 AFTER collect_episode | episode_keys="
                  + str(list(episode.keys())), flush=True)
            selected = episode["selected_indices"]
            _slen = len(selected) if hasattr(selected, "__len__") else "?"
            print("[trace] rank=0 selected=" + type(selected).__name__
                  + " len=" + str(_slen), flush=True)

            extras.update({
                "r_loss_mean": episode["r_loss_mean"],
                "r_entropy_mean": episode["r_entropy_mean"],
                "r_weight": episode["r_weight"],
                "rdiff_mean": episode["rdiff_mean"],
                "rconf_mean": episode["rconf_mean"],
                "lam": episode["lam"],
                "use_anchor": episode["use_anchor"],
                "align_mean": episode["align_mean"],
                "align_std": episode["align_std"],
            })

            if agent is not None:
                actor_loss, critic_loss = agent.update(
                    states=episode["states"],
                    actions=episode["actions"],
                    old_log_probs=episode["log_probs"],
                    rewards=episode["rewards"],
                )
                extras.update({"actor_loss": actor_loss, "critic_loss": critic_loss})
                logger.info(
                    "PPO update | actor_loss=%.4f | critic_loss=%.4f",
                    actor_loss, critic_loss,
                )
        except Exception as _e:
            print("[trace] rank=0 EXCEPTION in main branch: "
                  + type(_e).__name__ + ": " + str(_e), flush=True)
            _tb.print_exc()
            import sys as _sys
            _sys.stdout.flush()
            _sys.stderr.flush()
            raise
    else:
        selected = []

    _output_dir = (
        cfg.get("output_dir")
        or cfg.get("output_root")
        or "/tmp/tads_selection_share"
    )
    selected = _broadcast_selection(
        selected, epoch=epoch, output_dir=_output_dir,
    )
    return selected, extras


def save_selection(output_dir, epoch, selected):
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / ("selected_indices_epoch" + str(epoch) + ".json"), "w") as f:
        json.dump(selected, f)
