# TAG — Training-Adaptive Data Selection with a Reliability Gate

Training-adaptive instruction-data selection for LLM fine-tuning. Each
refresh (epoch) re-scores the full candidate pool under the *current*
checkpoint and trains on the top-B samples — an example's usefulness is a
function of where the model is on its trajectory, not a fixed property.

Real instruction pools are low-quality (mismatched pairs, truncated
responses, subtly wrong answers), and the difficulty–uncertainty signals
that drive dynamic selection actively *prefer* that corruption — broken
samples look hard. TAG adds one static, model-intrinsic **reliability
gate** in front of the dynamic score, fused non-compensatorily: because
both dynamic factors are bounded, a zeroed gate cannot be bought back by
any amount of difficulty or alignment evidence. Reliability is a veto,
not a vote.

No external judges, no reward models, no PPO-style selector training.

## Score

Three model-intrinsic evidence views, one multiplicative fusion:

    s_i^(t) = G_i · R_i^(t) · (1 + λ · ã_i^(t))

| View | Symbol | When | Signal |
|---|---|---|---|
| **Reliability** (static gate) | `G_i ∈ [0,1]` | once, base checkpoint | counterfactual likelihood contrast `ΔL_i = L(y_i\|x_i⁻) − L(y_i\|x_i)` — how much the true instruction improves the response's likelihood over an unrelated one — mapped to a zero-anchored gate `clip(2·(σ(ΔL_i/s) − 0.5), 0, 1)`, scale `s` calibrated once per backbone on a clean reference pool; combined with a completeness gate `c_i`. Cached, so refreshes cost nothing extra. |
| **Difficulty–uncertainty carrier** | `R_i^(t)` | every refresh | per-sample loss / predictive entropy under the current checkpoint (bounded before fusion) |
| **Trajectory-anchor alignment** | `ã_i^(t) ∈ [0,1]` | every refresh | checkpoint-to-checkpoint shifts in layer-wise hidden representations, projected on the anchor direction; supported by a local stability analysis |

`ΔL = 0` is a physical reference point ("the true instruction does not
help predict this response at all") independent of pool composition, which
is why the gate is anchored there rather than at a pool quantile — a rank
gate would suppress half of a clean pool and pass the top of an 80%-
corrupted one.

### What the code actually computes

The package name and config keys are still `tads` / `mvf` (rename pending).
Two scoring modes ship:

**`score_mode: tads`** — ungated legacy trajectory-anchored score:

    s_i = R_i · (1 + λ · ã_i),   R_i = w·L_i + (1−w)·H_i     (paper Eq. 10)

**`score_mode: mvf`** — the gated form (`configs/methods/tads_mvf.yaml`):

    D'_i  = d_floor + (1 − d_floor) · D_i^t
    S_i^t = (Q_i · c_i + ε)^γ · (D'_i + ε) · (1 + λ_t · ã_i^t)

mapping onto the equation above as `G_i = (Q_i·c_i + ε)^γ` and
`R_i^(t) = D'_i + ε`, with:

- `Q_i` — the zero-anchored calibrated gate above; computed once at the
  base checkpoint and cached (hard error if the cache is missing at
  epoch > 1 — recomputing at a later checkpoint silently changes what the
  view means). `K > 1` counterfactual pools give a dispersion-discounted
  `Q`. `reliability_mode: rank` / `reliability_rezero: false` are ablation
  arms.
- `c_i` — completeness: raw-text heuristic AND label-EOS check (EOS alone
  cannot see textual truncation, since tokenisation appends EOS
  unconditionally).
- `D_i^t = rank01(L^t) · (η + (1−η)·P̂)` — progress `P̂` ranked *within the
  previous refresh's selected set only*; unselected samples get a neutral
  0.5 (no gradient evidence, no verdict). `d_floor` compresses the factor
  to `[0.5, 1]` so the carrier modulates the ranking among gated samples
  instead of overriding the gate.
- `ã_i^t` — anchor alignment, pool-CDF (`rank01`) normalised in `mvf` mode
  (min-max in the legacy path); `λ_t = λ0 · anchor-stability` when
  `adaptive_lam` is on.

**Known manuscript↔code delta:** the abstract describes `R^(t)` as the
difficulty–uncertainty carrier inherited unchanged from trajectory-anchored
selection (`w·L + (1−w)·H`). The shipped `mvf` carrier is difficulty-only
(`rank01(L^t)` × progress, range-compressed); entropy is logged as a
diagnostic but does not enter the score. Reconcile before submission —
either restore entropy into the fused carrier, or state the substitution
explicitly in the method section. See
`docs/plan_low_quality_multiview.md` §2.0.

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
truncated response, duplicate instructions, wrong numeric answers,
fluent-but-wrong responses, and multi-source imbalance (`--source-scale`).
Every corruption is recorded in a verifiable manifest, so detection can be
scored per type.

Forward-only diagnosis (Dirty@K / AUPRC / per-type rejection per signal,
no training) — includes the `entropy`, `ppl`, and `ifd` comparison rows:

    python scripts/score_pool.py \
        --config configs/experiments/lowq/light_tads_mvf_05b.yaml \
        --manifest pools/composite20/corruption_manifest.json \
        --out pools/composite20/score_report.json

End-to-end training on the corrupted pool:

    export ALPACA_DATA_FILES=pools/composite20/pool.json
    export TADS_CF_FILES=pools/composite20/counterfactual.json
    export TADS_DEDUP_FILE=pools/composite20/dedup_clusters.json
    python -m tads.train --config configs/experiments/lowq/light_tads_mvf_05b.yaml

## Results status

Detection (Dirty@K / AUPRC vs. entropy-, perplexity-, and IFD-based
selection) and end-to-end seed-paired SFT comparisons are **pending** —
see `docs/plan_low_quality_multiview.md` §4–§5 for the pre-registered
endpoints and statistics. No number from the earlier CIKM submission may
be carried into the new manuscript (`docs/cikm-review-revision-audit.md`
§2).

## Layout

    tads/core/        scorer, selector, reliability, dedup, trajectory anchor
    tads/data/        Alpaca loading + corruption generation
    tads/pipelines/   per-epoch selection dispatch + SFT loop
    baselines/        data_agent / nait / lima / selectit / alpagasus / q2q
    configs/          composable YAML (base / methods / models / modes / experiments)
    scripts/          pool generation, forward-only scoring, run helpers
    docs/             active plan: plan_low_quality_multiview.md
