"""SelectIT training entrypoint — NOT YET IMPLEMENTED.

Paper:
    Liu et al., 2024b. "SelectIT: Selective Instruction Tuning for Large
    Language Models via Uncertainty-Aware Self-Reflection."
    https://arxiv.org/abs/2402.16705

Expected CLI (planned):
    python -m tads.baselines.selectit.train \\
        --config configs/experiments/main_7b/llama2/selectit_10.yaml \\
        --seed_path <seeds.json>

See `tads/baselines/nait/train.py` for the reference baseline structure
(seed-driven direction extraction → top-K selection → SFT).
"""
from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError(
        "SelectIT baseline is not yet implemented. Track the issue in the "
        "project board or refer to tads.baselines.nait.train for the "
        "reference baseline layout to mirror."
    )


if __name__ == "__main__":
    sys.exit(main())
