"""Comparison baselines for TADS.

Each subpackage is a self-contained baseline method with its own training
entrypoint, kept separate from `tads.train` (which orchestrates the four
TADS-family methods: random / full / data_agent / tads).

Currently provided:
    baselines.data_agent — Data Agent (Yang et al., ICML 2026, PPO + Beta actor)
    baselines.nait       — NAIT (Chen et al., ICLR 2026)
    baselines.selectit   — SelectIT (Liu et al., 2024b)
    baselines.lima       — LIMA (Zhou et al., 2023)
    baselines.alpagasus  — AlpaGasus (Chen et al., 2023)
    baselines.q2q        — Q2Q / Cherry-LLM IFD (Li et al., 2023)
"""
