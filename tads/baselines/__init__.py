"""Comparison baselines for TADS.

Each subpackage is a self-contained baseline method with its own training
entrypoint, kept separate from `tads.train` (which orchestrates the four
TADS-family methods: random / full / data_agent / tads).

Currently provided:
    tads.baselines.nait     — NAIT (Chen et al., ICLR 2026)
    tads.baselines.selectit — SelectIT (Liu et al., 2024b) [STUB]
"""
