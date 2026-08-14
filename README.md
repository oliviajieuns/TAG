# TAG — Training-Adaptive Data Selection with a Reliability Gate

Training-adaptive instruction-data selection for LLM fine-tuning. Each
refresh (epoch) re-scores the full candidate pool under the *current*
checkpoint and trains on the top-B samples — an example's usefulness is a
function of where the model is on its trajectory, not a fixed property.

Real instruction pools are low-quality (mismatched pairs, truncated
responses, subtly wrong answers), and the difficulty–uncertainty signals
that drive dynamic selection actively *prefer* that corruption — broken
samples look hard. TAG adds one static, model-intrinsic **reliability
gate** in front of the dynamic score, fused non-compensatorily: a zeroed
gate cannot be bought back by any amount of difficulty or alignment
evidence. Reliability is a weight whose floor is attainable, not a vote.

No external judges, no reward models, no PPO-style selector training.

## Score

Three model-intrinsic evidence views, one multiplicative fusion:

    s_i^(t) = G_i · R_i^(t) · (1 + λ · ã_i^(t))

| View | Symbol | When | Signal |
|---|---|---|---|
| **Reliability** (static gate) | `G_i ∈ [0,1]` | once, base checkpoint | counterfactual likelihood contrast: how much the true instruction improves the response's likelihood over an unrelated one, as a *relative* gain aggregated over the whole response and over its worst span, then squashed by a zero-anchored sigmoid whose scale is calibrated on a clean reference pool. Cached, so refreshes cost nothing extra. |
| **Difficulty–uncertainty carrier** | `R_i^(t)` | every refresh | `w·L_i + (1−w)·H_i` — per-sample loss and predictive entropy under the current checkpoint, inherited unchanged from the legacy selector |
| **Trajectory-anchor alignment** | `ã_i^(t) ∈ [0,1]` | every refresh | checkpoint-to-checkpoint shifts in layer-wise hidden representations, projected on the anchor direction; supported by a local stability analysis |

`Δ̂ = 0` is a physical reference point ("the true instruction does not help
predict this response at all") independent of pool composition, which is
why the gate is anchored there rather than at a pool quantile — a rank gate
would suppress half of a clean pool and pass the top of an 80%-corrupted
one. That absolute anchor, not the contrast itself, is what separates the
gate from IFD-style pool-relative rankings.

The gate's floor is exact: `G_i = 0 ⟹ s_i = 0` whatever the dynamic
factors say. For a small but non-zero gate, the achievable compensation is
governed by the pool's reward ratio times the anchor factor's `1+λ` — so
the exact-zero case needs only finiteness, while boundedness is what constrains
the *graded* case (see `docs/tag-paper-deltas.md` C2).

### The gate, concretely (paper Eqs. 2-6)

`tads/core/gate.py`. Per response token `k`, contrast the NLL under the true
instruction against the NLL under an unrelated one, then aggregate at two
granularities:

    δ_{i,k} = ℓ_k(y_i|x_i⁻) − ℓ_k(y_i|x_i)                          (Eq. 2)
    Δ̄_i    = 1 − L(y_i|x_i) / L(y_i|x_i⁻)                           (Eq. 3)
    Δ_{i,m} = 1 − Σ_{k∈S_m} ℓ_k(y_i|x_i) / Σ_{k∈S_m} ℓ_k(y_i|x_i⁻)  (Eq. 4)
    Δ^min_i = min over admissible spans S_m ∈ C_i                    (Eq. 5)
    Δ̂_i     = min(Δ̄_i, Δ^min_i) − μ(M_i)                            (Eq. 5′)
    G_i     = c_i · (2σ(Δ̂_i/s) − 1)₊                                 (Eq. 6)

The **ratio** makes the statistic scale-free, so an intrinsically hard
response is not mistaken for a large gain. The **span minimum** is what
catches localized corruption: a response whose final answer is wrong still
gains on every other token, so `Δ̄` stays healthy while `Δ^min` collapses.
`C_i` excludes low-information spans — boilerplate carries no instruction
dependency by nature and must not drive the gate down.

**Eq. 5′** is an amendment the implementation forced. `Δ^min` is a minimum
over `M = ⌈n/W⌉` spans, so its null *location* falls as responses get
longer; tested against a fixed zero it put 60% of a **clean** 7B reference
at the floor, almost all of it long, while `Δ̄` averaged a healthy +0.108.
`μ(M)` is the `target_zero_rate`-quantile of the uncorrected statistic on a
clean reference at the same span count, which makes the share of clean data
at the floor a dial (5% by default) and uniform in length. `μ` is fit on
clean data only, so it cannot absorb corruption signal — there is no
in-pool fallback for it. See `docs/tag-paper-deltas.md` A5, and the
`tag_nonull_7b` ablation arm.

**`G` is a continuous weight, not a binary gate.** `(·)₊` is a ReLU-style
kink, not a step: `2σ(z)−1 → 0` as `z → 0⁺`, so `G` is continuous in `Δ̂`
and about half the pool receives a weight strictly between the bounds. What
is distinctive is that the floor is *attainable* — `σ(0)=½` makes
`Δ̂ ≤ 0 → G = 0` exactly, where an ordinary sigmoid gate only approaches 0.
That attainability, not thresholding, is what makes the fusion
non-compensatory; it also means an estimation error at the floor is
unrecoverable, which is why Eq. 5′ is required rather than optional.

Because response token IDs are identical under both instructions (prompt
and response are tokenised separately), the two per-token vectors are
index-aligned by construction; only the length budget can differ, so both
are trimmed to their common prefix first.

### What the code actually computes

The package name and most config keys are still `tads` (rename pending).
Three scoring modes ship:

**`score_mode: tads`** — ungated legacy trajectory-anchored score:

    s_i = R_i · (1 + λ · ã_i),   R_i = w·L_i + (1−w)·H_i     (legacy Eq. 10)

**`score_mode: tag`** — the paper's method (`configs/methods/tag.yaml`):
exactly the legacy score above with `G_i` multiplied in front, so `G ≡ 1`
reproduces the legacy ranking bit-for-bit and the gate is a clean ablation.
`G` is computed once at the base checkpoint and cached in
`tag_gate_cache.pt`; a missing cache after the base checkpoint is a hard
error rather than a silent recomputation at the wrong checkpoint, since `G`
is defined as a property of the data.

**`score_mode: mvf`** — the earlier multi-view fusion, kept as a comparison
arm (`configs/methods/tads_mvf.yaml`):

    S_i^t = (Q_i · c_i + ε)^γ · (D'_i + ε) · (1 + λ_t · ã_i^t)

Unlike TAG this *replaces* the composite reward with a difficulty-only
carrier `D'` and drops entropy. The two modes are mutually exclusive.

Notes shared by both gated modes:

- `c_i` — completeness: raw-text heuristic AND label-EOS check (EOS alone
  cannot see textual truncation, since tokenisation appends EOS
  unconditionally).
- `K > 1` counterfactual pools give a dispersion-discounted gate; the gate
  is applied per pairing and then averaged, not the reverse (the clamp is
  convex, so gate-of-mean would collapse straddling evidence to an exact zero).
- Near-duplicate clusters (MinHash) admit at most one selection each. The
  legacy path never deduplicated, so TAG threads `cluster_ids` explicitly.
- When fewer candidates pass the gate than the budget needs, the leftover
  slots go to the best zero-weight samples by the **ungated** score, with a loud
  warning — without that rule the exact-zero ties would be broken by pool
  file order. `score_pool.py` reports `budget_fits@K` so a shortfall cannot
  pass unnoticed.

**Proposed paper amendments** — places where making the equations
executable exposed an undefined case or an over-strong claim — are recorded
in `docs/tag-paper-deltas.md`. Four are correctness fixes (empty `C_i`, the
common-prefix trim, no-evidence samples, and the budget-shortfall rule);
the most consequential is the order-statistic drift of `Δ^min` with
response length, which makes the span width `W` a first-order
hyper-parameter rather than a detail.

## Running it

On a fresh GPU box (Lambda / RunPod / vast.ai / a bare VM) — everything lands
under one workspace, no cluster mounts needed:

    source scripts/gpu_cloud/env.sh          # workspace + env vars
    bash   scripts/gpu_cloud/bootstrap.sh    # deps, weights, data, pools, calibration
    python scripts/gpu_cloud/preflight.py    # cheap checks before GPU hours
    bash   scripts/run_tag_lowq_05b.sh smoke # ~2 min, proves the epoch-2 path

See `docs/gpu-cloud-quickstart.md`. `scripts/setup_env.sh` is the n9-cluster
equivalent and points at `/group-volume` paths.

The `smoke` stage is worth running once per machine: `G` is defined at the
base checkpoint and cached, so the whole gate lifecycle — cache write, cache
hit, and the hard error when it is missing — only exercises at **epoch 2**.

## Low-quality-pool experiments

The whole TAG sequence (pools → calibrate → detection → SFT) is wrapped in

    bash scripts/run_tag_lowq_05b.sh all

Step by step, and what each step is for:

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

Calibrate the gate scale `s` on a **clean** reference pool. This is not
optional for a reported run: without it the gate self-calibrates in-pool,
which makes `G` depend on how dirty its neighbours are — exactly the pool
dependence that anchoring at `Δ̂ = 0` removes:

    python scripts/calibrate_reliability.py --mode tag \
        --config configs/experiments/lowq/light_tag_05b.yaml \
        --pool pools/clean_ref/pool.json \
        --counterfactual pools/clean_ref/counterfactual.json \
        --out pools/clean_ref/delta_hat_05b.pt
    export TADS_GATE_REF=pools/clean_ref/delta_hat_05b.pt

Forward-only diagnosis (Dirty@K / AUPRC / per-type rejection per signal, no
training) — includes the `entropy`, `ppl`, and `ifd` comparison rows, the
TAG rows, and the length-bias profile:

    export ALPACA_DATA_FILES=pools/composite20/pool.json
    export TADS_CF_FILES=pools/composite20/counterfactual.json
    export TADS_DEDUP_FILE=pools/composite20/dedup_clusters.json
    python scripts/score_pool.py \
        --config configs/experiments/lowq/light_tag_05b.yaml \
        --manifest pools/composite20/corruption_manifest.json \
        --out pools/composite20/score_report.json \
        --save-signals pools/composite20/signals.pt

`delta_bar` vs `delta_min` in that report **is** the span ablation: if the
tail gain does not beat the overall gain on the localized corruption types
(wrong-answer, fluent-wrong), Eqs. 4-5 are not earning their place.

End-to-end training on the corrupted pool:

    python -m tads.train --config configs/experiments/lowq/light_tag_05b.yaml

## Results status

Detection (Dirty@K / AUPRC vs. entropy-, perplexity-, and IFD-based
selection) and end-to-end seed-paired SFT comparisons are **pending** —
see `docs/plan_low_quality_multiview.md` §4–§5 for the pre-registered
endpoints and statistics. No number from the earlier CIKM submission may
be carried into the new manuscript (`docs/cikm-review-revision-audit.md`
§2).

## Layout

    tads/core/gate.py the TAG reliability gate (Eqs. 2-6)
    tads/core/        scorer, selector, reliability, dedup, trajectory anchor
    tads/data/        Alpaca loading + corruption generation
    tads/pipelines/   per-epoch selection dispatch + SFT loop
    baselines/        data_agent / nait / lima / selectit / alpagasus / q2q
    configs/          composable YAML (base / methods / models / modes / experiments)
    scripts/          pool generation, forward-only scoring, run helpers
    docs/             active plan: plan_low_quality_multiview.md
                      paper amendments: tag-paper-deltas.md
