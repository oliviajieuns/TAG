"""Per-epoch sample selection dispatch.

Wraps the four selection methods behind a single function. For data_agent
and tads the selection is performed on rank-0 only under DDP, and the
chosen indices are shared with other ranks via a small JSON file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..core.agent import PPOAgent
from ..core.selector import collect_episode
from ..core.trajectory_anchor import TrajectoryAnchor
from ..core.utils import is_main_process, local_rank, rank, world_size

logger = logging.getLogger(__name__)


def _random_indices(n_total, ratio, seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)
    perm = torch.randperm(n_total, generator=g).tolist()
    k = max(1, int(n_total * ratio))
    return perm[:k]


def _broadcast_selection(selected, *, epoch=0, output_dir=None):
    """Share selected indices from rank 0 to all ranks via filesystem.

    Robust replacement for the old dist.broadcast version: a barrier guarantees
    write-before-read, missing file raises a clear error, and there are no
    NCCL ordering / async-stream / process-group conflicts.
    """
    import os as _os
    import tempfile as _tempfile
    from pathlib import Path as _Path

    _r = dist.get_rank() if dist.is_initialized() else 0
    print("[sel-share] rank=" + str(_r) + " pid=" + str(_os.getpid())
          + " type=" + type(selected).__name__, flush=True)

    if not dist.is_initialized():
        if hasattr(selected, "tolist"):
            return selected.tolist()
        return list(selected) if not isinstance(selected, list) else selected

    SRC = 0
    base = _Path(output_dir) if output_dir is not None else _Path(_tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    sel_path = base / ("_selection_epoch" + str(epoch) + ".json")

    if _r == SRC:
        if hasattr(selected, "tolist"):
            selected = selected.tolist()
        elif not isinstance(selected, list):
            selected = list(selected)
        selected = [int(x) for x in selected]
        print("[sel-share] rank=0 NORMALIZED type=list len=" + str(len(selected))
              + " first5=" + str(selected[:5]), flush=True)
        tmp_path = sel_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(selected, f)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp_path, sel_path)
        print("[sel-share] rank=0 WROTE " + str(sel_path)
              + " (" + str(len(selected)) + " ids)", flush=True)

    dist.barrier()

    if not sel_path.exists():
        raise RuntimeError(
            "[rank " + str(_r) + "] selection file missing after barrier: "
            + str(sel_path) + ". Rank 0 likely crashed before writing."
        )
    with open(sel_path, "r") as f:
        result = json.load(f)
    if not isinstance(result, list):
        raise RuntimeError(
            "[rank " + str(_r) + "] selection file has wrong shape: type="
            + type(result).__name__
        )
    print("[sel-share] rank=" + str(_r) + " READ " + str(sel_path)
          + " len=" + str(len(result)), flush=True)

    dist.barrier()
    if _r == SRC:
        try:
            sel_path.unlink()
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
