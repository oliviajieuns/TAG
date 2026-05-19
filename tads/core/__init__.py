"""Core algorithms: reward / scorer (paper Eq.2-3, 7, 8), selector,
trajectory anchor, utils."""
from .reward import compute_rewards, composite_reward
from .scorer import (
    calibrated_utility,
    normalize_alignment,
    pool_reward,
    select_top_b,
    tads_score,
)
from .selector import collect_episode
from .trajectory_anchor import TrajectoryAnchor
from .utils import cuda_mem_str, load_config, set_seed, setup_logger

__all__ = [
    "calibrated_utility",
    "collect_episode",
    "composite_reward",
    "compute_rewards",
    "cuda_mem_str",
    "load_config",
    "normalize_alignment",
    "pool_reward",
    "select_top_b",
    "set_seed",
    "setup_logger",
    "tads_score",
    "TrajectoryAnchor",
]
