"""LIMA (Zhou et al., 2023) — pure data-replacement baseline.

LIMA is not a data-selection algorithm; it's a hand-curated 1030-sample
instruction-tuning dataset. This subpackage loads GAIR/lima (or a local
mirror) and runs the shared SFT loop.
"""
