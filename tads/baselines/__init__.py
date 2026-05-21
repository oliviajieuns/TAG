"""Comparison baselines for TADS.

Each subpackage is a self-contained baseline method with its own training
entrypoint, kept separate from `tads.train` (which orchestrates the four
TADS-family methods: random / full / data_agent / tads).

Currently provided:
    tads.baselines.data_agent — Data Agent (Yang et al., ICML 2026, PPO + Beta actor)
    tads.baselines.nait       — NAIT (Chen et al., ICLR 2026)
    tads.baselines.selectit   — SelectIT (Liu et al., 2024b)
    tads.baselines.lima       — LIMA (Zhou et al., 2023)
    tads.baselines.alpagasus  — AlpaGasus (Chen et al., 2023)
    tads.baselines.q2q        — Q2Q / Cherry-LLM IFD (Li et al., 2023)
"""
