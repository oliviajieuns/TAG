"""TAG — Training-Adaptive data selection with a reliability Gate.

Top-level convenience re-exports. Keep this file minimal — heavy modules
(modeling, data, evals) are imported lazily by their respective subpackages.
"""
__version__ = "0.1.0"

from tag.core.reward import compute_rewards
from tag.core.scorer import (
    calibrated_utility,
    legacy_score,
    normalize_alignment,
    pool_reward,
    select_top_b,
)
from tag.core.selector import collect_episode
from tag.core.trajectory_anchor import TrajectoryAnchor
from tag.core.utils import cuda_mem_str, load_config, set_seed, setup_logger

__all__ = [
    "__version__",
    "calibrated_utility",
    "collect_episode",
    "compute_rewards",
    "cuda_mem_str",
    "legacy_score",
    "load_config",
    "normalize_alignment",
    "pool_reward",
    "select_top_b",
    "set_seed",
    "setup_logger",
    "TrajectoryAnchor",
]
