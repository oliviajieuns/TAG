# TADS — Trajectory-Anchored Data Selection

Training-adaptive instruction-data selection for LLM fine-tuning. Each
refresh (epoch) scores the full candidate pool under the current
checkpoint and trains on the top-B samples.

## Scoring modes

**Legacy (`score_mode: tads`, paper Eq. 10)** — composite reward ×
trajectory-anchor alignment:

    s_i = R_i · (1 + λ · ã_i),   R_i = w·L_i + (1−w)·H_i

**MVF v3 (`score_mode: mvf`)** — reliability-gated multi-view fusion for
low-quality pools (`docs/plan_low_quality_multiview.md` §2). Uncertainty is
not treated as quality; three views from genuinely distinct information
sources are fused as a weighted log-opinion pool with a non-compensatory
gate:

    D'_i  = d_floor + (1 − d_floor) · D_i^t
    S_i^t = (Q_i · c_i + ε)^γ · (D'_i + ε) · (1 + λ_t · ã_i^t)

| View | Signal | Source |
|---|---|---|
| Consistency `Q_i` | counterfactual instruction fidelity, zero-anchored calibrated gate `clip(2·(σ(ΔL_i/s) − 0.5), 0, 1)` with `ΔL_i = L(y_i\|x_i⁻) − L(y_i\|x_i)`; `s` calibrated once per backbone on a clean reference pool; cached at the base checkpoint (hard error if missing later); completeness gate `c_i` (raw-text heuristic AND label-EOS) | separate counterfactual forward pass (K > 1 pools → dispersion-discounted Q) |
| Dynamics `D_i^t` | `rank01(L^t) · (η + (1−η)·P̂)`, progress `P̂` ranked within the previous refresh's SELECTED set only (unselected = neutral 0.5 — no gradient evidence, no verdict); `d_floor` compresses the factor to [0.5, 1] so it modulates rather than overrides the gate | cross-refresh loss dynamics |
| Geometry `ã_i^t` | anchor hidden-state alignment, pool-CDF (rank01) normalised in MVF mode; `λ_t = λ0 ·` anchor-stability when `adaptive_lam` is on | layer-wise hidden-state geometry |

Duplicated instructions are handled outside the score: near-duplicate
clusters (MinHash) admit at most one selection each.

## Low-quality-pool experiments

Generate a corrupted pool + ground-truth manifest (+ counterfactual pool
and dedup clusters):

    python scripts/make_corrupted_pool.py \
        --input alpaca_gpt4.json --out-dir pools/composite20 \
        --preset composite20 --duplicate-frac 0.05 --seed 42 \
        --emit-counterfactual --emit-dedup-clusters

Corruption types: instruction–response mismatch, noisy response,
truncated response, duplicate instructions, wrong numeric answers, and
multi-source imbalance (`--source-scale`).

Forward-only diagnosis (Dirty@K / AP / per-type rejection per signal, no
training):

    python scripts/score_pool.py \
        --config configs/experiments/lowq/light_tads_mvf_05b.yaml \
        --manifest pools/composite20/corruption_manifest.json \
        --out pools/composite20/score_report.json

End-to-end training on the corrupted pool:

    export ALPACA_DATA_FILES=pools/composite20/pool.json
    export TADS_CF_FILES=pools/composite20/counterfactual.json
    export TADS_DEDUP_FILE=pools/composite20/dedup_clusters.json
    python -m tads.train --config configs/experiments/lowq/light_tads_mvf_05b.yaml

## Layout

    tads/core/        scorer, selector, reliability, dedup, trajectory anchor
    tads/data/        Alpaca loading + corruption generation
    tads/pipelines/   per-epoch selection dispatch + SFT loop
    baselines/        data_agent / nait / lima / selectit / alpagasus / q2q
    configs/          composable YAML (base / methods / models / modes / experiments)
    scripts/          pool generation, forward-only scoring, run helpers
    docs/             active plan: plan_low_quality_multiview.md
