# TADS — Trajectory-Anchored Data Selection

Training-adaptive instruction-data selection for LLM fine-tuning. Each
refresh (epoch) scores the full candidate pool under the current
checkpoint and trains on the top-B samples.

## Scoring modes

**Legacy (`score_mode: tads`, paper Eq. 10)** — composite reward ×
trajectory-anchor alignment:

    s_i = R_i · (1 + λ · ã_i),   R_i = w·L_i + (1−w)·H_i

**MVF (`score_mode: mvf`)** — quality-gated multi-view fusion for
low-quality pools (`docs/plan_low_quality_multiview.md`). Uncertainty is
not treated as quality; instead three views from genuinely distinct
information sources are fused:

    S_i^t = (Q_i · c_i + ε)^γ · (D_i^t + ε) · (1 + λ · ã_i^t)

| View | Signal | Source |
|---|---|---|
| Reliability `Q_i` | counterfactual instruction fidelity `rank01[L(y_i\|x_i⁻) − L(y_i\|x_i)]`, cached at the base checkpoint; completeness gate `c_i` (EOS check) | separate counterfactual forward pass |
| Learnability `D_i^t` | `rank01(L^t) · (η + (1−η)·rank01([L^{t−1}−L^t]₊))` | cross-refresh loss dynamics |
| Alignment `ã_i^t` | trajectory-anchor hidden-state alignment (unchanged) | layer-wise hidden-state geometry |

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
