"""Per-epoch sample selection dispatch.

Wraps the four selection methods behind a single function. For ``data_agent``
and ``tads`` the selection is performed on rank-0 only under DDP, and the
chosen indices are broadcast to other ranks so that all workers train on
the same subset without each rank rerunning the whole episode forward pass.
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


def _random_indices(n_total: int, ratio: float, seed: int, epoch: int) -> List[int]:
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)
    perm = torch.randperm(n_total, generator=g).tolist()
    k = max(1, int(n_total * ratio))
    return perm[:k]


def _broadcast_selection(selected) -> List[int]:
    """Broadcast selected indices from GLOBAL rank 0 to all ranks (defensive)."""
    import os as _os
    _r_enter = dist.get_rank() if dist.is_initialized() else 0
    print(f"[bcast-enter] rank={_r_enter} pid={_os.getpid()} type={type(selected).__name__}", flush=True)
    if not dist.is_initialized():
        if hasattr(selected, "tolist"):
            return selected.tolist()
        return list(selected)
    device = (
        torch.device(f"cuda:{local_rank()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    SRC = 0
    rank = dist.get_rank()
    if rank == SRC:
        _t = type(selected).__name__
        _sh = getattr(selected, "shape", None)
        _dev = getattr(selected, "device", None)
        if hasattr(selected, "shape"):
            _repr = f"shape={_sh} device={_dev}"
        else:
            _repr = repr(selected)[:200]
        print(f"[bcast] rank=0 BEFORE-NORMALIZE: type={_t} {_repr}", flush=True)
        if hasattr(selected, "tolist"):
            selected = selected.tolist()
        elif not isinstance(selected, list):
            selected = list(selected)
        _post_t = type(selected).__name__
        if hasattr(selected, "__len__"):
            _post_len = len(selected)
        else:
            _post_len = "NO-LEN"
        if isinstance(selected, list):
            _post_first5 = selected[:5]
        else:
            _post_first5 = "NOT-LIST"
        print(f"[bcast] rank=0 AFTER-NORMALIZE: type={_post_t} len={_post_len} first5={_post_first5}", flush=True)
        length_val = len(selected)
    else:
        length_val = 0
    length = torch.tensor([length_val], dtype=torch.long, device=device).contiguous()
    dist.broadcast(length, src=SRC)
    n = int(length.item())
    print(f"[bcast] rank={rank} after-bcast n={n} device={device}", flush=True)
    if n < 0 or n > 10_000_000:
        raise RuntimeError(
            f"[rank {rank}] _broadcast_selection garbage length n={n}. "
            f"Source rank selected appears corrupted. Check BEFORE-NORMALIZE log above."
        )
    if rank == SRC:
        payload = torch.tensor(selected, dtype=torch.long, device=device).contiguous()
    else:
        payload = torch.zeros(n, dtype=torch.long, device=device)
    dist.broadcast(payload, src=SRC)
    return payload.cpu().tolist()


def select_indices(
    method: str,
    *,
    model,
    agent: Optional[PPOAgent],
    anchor: Optional[TrajectoryAnchor],
    dataset,
    cfg: Dict[str, Any],
    epoch: int,
    seed: int,
    device,
) -> Tuple[List[int], Dict[str, Any]]:
    """Return ``(selected_indices, extras)`` for the given epoch."""
    n_total = len(dataset)
    ratio = float(cfg["selection_ratio"])
    extras: Dict[str, Any] = {}

    if method == "full":
        selected = list(range(n_total))
        logger.info("Full dataset selection | k=%d", len(selected))
        return selected, extras

    if method == "random":
        selected = _random_indices(n_total, ratio, seed, epoch)
        logger.info("Random selection | k=%d/%d", len(selected), n_total)
        return selected, extras

    if method not in ("tads", "data_agent"):
        raise ValueError(f"Unknown method: {method!r}")

    # --- data_agent or tads: episode collection (rank-0 only under DDP) ---
    if is_main_process():
        print(f"[trace] rank=0 ENTER main branch | method={method} | anchor={'set' if anchor is not None else 'None'}", flush=True)
        import traceback as _tb
        try:
            if method == "tads" and anchor is not None:
                logger.info("Updating trajectory anchor ...")
                print(f"[trace] rank=0 BEFORE anchor.update", flush=True)
                anchor_stats = anchor.update(
                    model=model, dataset=dataset, seed=seed, epoch=epoch,
                )
                print(f"[trace] rank=0 AFTER anchor.update | stats_keys={list(anchor_stats.keys()) if anchor_stats else None}", flush=True)
                extras["anchor_stats"] = anchor_stats

            tads_cfg = cfg.get("tads", {}) or {}
            exp_tag = f"{cfg.get('model_key','?')}/alpaca/{method}"

            print(f"[trace] rank=0 BEFORE collect_episode", flush=True)
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
            print(f"[trace] rank=0 AFTER collect_episode | episode_keys={list(episode.keys())}", flush=True)
            selected = episode["selected_indices"]
            print(f"[trace] rank=0 selected={type(selected).__name__} len={len(selected) if hasattr(selected,'__len__') else '?'}", flush=True)

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

            # PPO update (rank-0 only).
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
            print(f"[trace] rank=0 EXCEPTION in main branch: {type(_e).__name__}: {_e}", flush=True)
            _tb.print_exc()
            import sys as _sys
            _sys.stdout.flush(); _sys.stderr.flush()
            raise
    else:
        selected = []  # placeholder, will be broadcast

    selected = _broadcast_selection(selected)
    return selected, extras


def save_selection(output_dir: Path, epoch: int, selected: List[int]) -> None:
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"selected_indices_epoch{epoch}.json", "w") as f:
        json.dump(selected, f)
