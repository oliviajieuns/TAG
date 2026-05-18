"""TADS — Trajectory-Anchored Data Selection for Instruction Tuning.

Top-level convenience re-exports. Keep this file minimal — heavy modules
(modeling, data, evals) are imported lazily by their respective subpackages.
"""
__version__ = "0.1.0"

from tads.core.reward import compute_rewards
from tads.core.scorer import (
    calibrated_utility,
    normalize_alignment,
    pool_reward,
    select_top_b,
    tads_score,
)
from tads.core.selector import collect_episode
from tads.core.trajectory_anchor import TrajectoryAnchor
from tads.core.utils import cuda_mem_str, load_config, set_seed, setup_logger

__all__ = [
    "__version__",
    "calibrated_utility",
    "collect_episode",
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
