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


def _broadcast_selection(selected: List[int]) -> List[int]:
    """Broadcast ``selected`` (a python list of ints) from rank 0 to all."""
    if not dist.is_initialized():
        return selected
    device = f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
    # Encode length first, then the payload.
    if is_main_process():
        length = torch.tensor([len(selected)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(length, src=0)
    n = int(length.item())
    if is_main_process():
        payload = torch.tensor(selected, dtype=torch.long, device=device)
    else:
        payload = torch.zeros(n, dtype=torch.long, device=device)
    dist.broadcast(payload, src=0)
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
        if method == "tads" and anchor is not None:
            logger.info("Updating trajectory anchor ...")
            anchor_stats = anchor.update(
                model=model, dataset=dataset, seed=seed, epoch=epoch,
            )
            extras["anchor_stats"] = anchor_stats

        tads_cfg = cfg.get("tads", {}) or {}
        exp_tag = f"{cfg.get('model_key','?')}/alpaca/{method}"

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
        selected = episode["selected_indices"]

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
