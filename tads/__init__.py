"""TADS — Trajectory-Anchored Data Selection for Instruction Tuning.

Top-level convenience re-exports. Keep this file minimal — heavy modules
(modeling, data, evals) are imported lazily by their respective subpackages.
"""
__version__ = "0.1.0"

from tads.core.agent import PPOAgent
from tads.core.reward import compute_rewards, composite_reward
from tads.core.selector import collect_episode
from tads.core.trajectory_anchor import TrajectoryAnchor
from tads.core.utils import set_seed, setup_logger, load_config, cuda_mem_str

__all__ = [
    "__version__",
    "PPOAgent",
    "compute_rewards",
    "composite_reward",
    "collect_episode",
    "TrajectoryAnchor",
    "set_seed",
    "setup_logger",
    "load_config",
    "cuda_mem_str",
]
