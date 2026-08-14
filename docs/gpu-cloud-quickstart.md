# Running TAG on a GPU cloud box

For a fresh machine — Lambda, RunPod, vast.ai, a bare A100 VM — with nothing
but the repo and a GPU. `scripts/setup_env.sh` is for the n9 cluster and
points at `/group-volume` mounts that do not exist elsewhere; use
`scripts/gpu_cloud/` instead, which keeps everything under one workspace.

## The short version

```bash
git clone <repo> && cd TAG
source scripts/gpu_cloud/env.sh          # workspace + all env vars
bash   scripts/gpu_cloud/bootstrap.sh    # deps, weights, data, pools, calibration
python scripts/gpu_cloud/preflight.py    # cheap checks — do not skip this
bash   scripts/run_tag_lowq_05b.sh smoke     # ~2 min, proves the epoch-2 path
bash   scripts/run_tag_lowq_05b.sh phasea    # detection table
bash   scripts/run_tag_lowq_05b.sh phaseb    # full SFT run
```

Put the workspace on a scratch disk if `/` is small:

```bash
TAG_WORKSPACE=/mnt/data/tag source scripts/gpu_cloud/env.sh
```

Requirements: one GPU with ≥12 GB (0.5B + LoRA fits comfortably in 24 GB),
~10 GB disk, and outbound network for the bootstrap's download steps only.

## Why the smoke stage exists

`G` is defined at the **base checkpoint** and cached, so the entire gate
lifecycle — cache write, cache hit, and the hard error when the cache is
missing — only exercises at **epoch 2**. A single-epoch test proves nothing
about it. During development this repo had a bug where every run with the
shipped defaults trained happily through epoch 1 and died at the start of
epoch 2; a one-epoch check would not have caught it.

`smoke` runs 2 epochs on a small subset (`SMOKE_N`, default 512) and then
inspects the artifacts: gate cache present, two distinct selections written,
and the realised zero-weight accounting. Run it once on a new machine before
spending real GPU hours.

## What preflight checks, and why each one is there

| Check | Why it costs GPU time to miss |
|---|---|
| GPU / CUDA, bf16 | A CPU-only torch wheel is the most common cloud-image surprise; the loader defaults to bf16, which pre-Ampere cards emulate very slowly |
| dependencies | `peft` is only needed for LoRA — and every lowq arm is LoRA, so it fails at model load, minutes in |
| pool ↔ counterfactual alignment | Training only checks **lengths**. A stale counterfactual of the right length passes silently and the gate then contrasts responses against the wrong instructions — the run completes and the numbers are meaningless |
| counterfactual actually deranged | If instructions were not shuffled, Δ ≈ 0 everywhere and the gate zeroes the whole pool. Compared against the collision rate expected from repeated instructions, so a legitimately deranged pool on a repetitive corpus does not false-alarm |
| gate reference | Wrong *kind* of reference (an MVF `delta` in nats instead of a TAG `delta_hat` ratio) would scale `s` an order of magnitude too high and quietly disable the gate |
| gate reference span config | `s` is a quantile of Δ̂, whose distribution depends on the span partition. Calibrating at W=16 and gating at W=32 mis-scales every gate value, with no symptom but a wrong zero-weight rate |
| manifest ↔ pool | Phase A's Dirty@K is meaningless if the manifest describes a different pool |
| disk | The tokenised cache plus (with `store_token_losses`) an fp16 token-loss tensor per pool |

Exit code 0 means nothing failed; `--strict` also fails on warnings.

## Calibration is not optional

The bootstrap's `calibrate` step derives **two** things from a **clean**
reference pool, and they are not interchangeable:

- the gate scale `s` (Eq. 6). Skipping it makes the gate self-calibrate
  in-pool, which reintroduces the pool dependence that anchoring at `Δ̂ = 0`
  removes — `G` then depends on how dirty a sample's neighbours are. The code
  warns loudly at every layer but nothing stops you: fine for a smoke test,
  wrong for a reported run.
- the null curve `μ(M)` (Eq. 5′). This one has **no** in-pool fallback and no
  warning path — a missing curve is a hard error. Fitting `μ` on a 30%-dirty
  candidate pool would absorb the exact signal the gate looks for, so the
  only options the code offers are a clean reference or
  `null_correction: false` (the declared ablation).

If `TAG_GATE_REF_7B` names a file that does not exist, or one written before
the null curve existed, the run fails with the exact command to regenerate it
rather than silently falling back.

## Reading the smoke report

```
  ok    gate cache written (12.4 MB)
  ok    2 epochs selected; epoch1 n=520, epoch2 n=520, overlap 71%
  ok    epoch 1: gate_mean=0.7412 zero_frac=0.183 admissible=4247/520 n_zero_weight_selected=0
```

- `n_zero_weight_selected=0` is the one that matters: it is the run's own evidence
  that non-compensation held for every selected sample. Anything above 0 must be
  reported, not silently accepted.
- `zero_frac` above 0.9 almost always means the scale is wrong — usually the
  in-pool fallback on an uncalibrated run.
- 100% overlap between epochs means selection stopped responding to the
  trajectory, which is expected only for the `-static` control arm.

## The span-width sweep

`span_tokens` (W) is a first-order hyper-parameter, not a detail: Δ^min is a
minimum over `M ≈ n/W` spans, so the tail statistic drifts down with response
length, while per-span noise shrinks like `1/√W`. The two effects pull in
opposite directions. The 0.5B arm sets `store_token_losses: true` so the
sweep costs **no forward pass** — G is re-derived from cached token losses:

```bash
for W in 8 16 32 64; do
  python scripts/score_pool.py \
    --config configs/experiments/lowq/light_tag_05b.yaml \
    --manifest $POOLS/composite20/corruption_manifest.json \
    --out     $POOLS/composite20/report_W$W.json \
    --span-tokens $W
done
```

Note what `--span-tokens` does to calibration: `s` is a quantile of Δ̂ under a
*specific* partition, so a reference calibrated at the config's W no longer
applies. The flag therefore drops `gate_ref_file` for that sweep point and
self-calibrates in-pool, which is fine for **comparing detection across W**
(the quantity being compared is the ranking) and not fine for a reported
absolute zero-weight rate. Once W is chosen, recalibrate at that W and re-run.

Watch `tag.length_bias` in each report: the clean false-zero rate should not
swing much across response-length quantiles. See
`docs/tag-paper-deltas.md` §B2 for the quantitative argument.

## Multi-GPU

Selection runs on rank 0 only and shares its result through a filesystem
sentinel rather than an NCCL barrier (a barrier there deadlocked earlier
versions — rank 0 spends a long time inside `collect_episode` while the
other ranks sit in the collective). Training itself is ordinary DDP:

```bash
torchrun --nproc_per_node=4 -m tag.train \
    --config configs/experiments/lowq/light_tag_05b.yaml
```

At 0.5B a single GPU is usually faster than the DDP overhead.

## Troubleshooting

**`no usable gate cache at epoch N (> 1)`** — `G` must come from the base
checkpoint. Either restore `tag_gate_cache.pt` into the run dir, or start a
fresh run. `selection.tag.allow_late_gate: true` accepts a wrong-checkpoint `G`
explicitly, which is a deliberate escape hatch, not a fix.

**Over 90% of the pool zeroed** — with `null_correction: true` this cannot be
a scale problem, because the zero is decided at `Δ̂ ≤ 0`, before `s` is
consulted. Check in this order: (1) is the reference from THIS backbone and
THIS pool? (2) does the calibration's per-bin table read ~`target_zero_rate`
everywhere? (3) is `Δ̄` mean positive? A negative `Δ̄` means the reference is
contaminated or the counterfactuals are not actually unrelated, which no
correction fixes. `compute_gate` warns when the pool rate exceeds 50% or
falls below half of `target_zero_rate`.

**Most of the pool demoted to `c_trunc`** — the completeness heuristic
misfiring, not the gate. Run `scripts/audit_completeness.py --ablate`: if the
false-positive rate on the uncorrupted subset exceeds the true truncation
rate, the view is costing more than it buys (see `docs/tag-paper-deltas.md`
item B4).

**OOM during the gate forward** — lower `episode_batch_size`; the token-level
pass keeps a per-token vector per sample rather than a scalar.

**Cache seems stale after editing tokenisation** — the HF `map` fingerprint
has served stale tokenisations before. Set `TAG_FRESH_DATA_CACHE=1` for one
run.

---

# 7B on a multi-GPU box

```bash
source scripts/gpu_cloud/env.sh
bash   scripts/gpu_cloud/bootstrap.sh all7b      # +Qwen2.5-7B (~15 GB), 7B calibration
# env.sh already exported TAG_GATE_REF_7B=$POOLS/clean_ref/delta_hat_7b.pt.
# The 7B arms read TAG_GATE_REF_7B, NOT TAG_GATE_REF — the latter is the
# 0.5B reference, and a single shared variable made picking the wrong
# backbone's calibration silent.
export TAG_EPISODE_BS_7B=32

# Pick W from the calibration, on CPU, in seconds. The calibrate step keeps
# the per-token NLLs, so every W is re-derived without a forward pass.
python scripts/sweep_gate_config.py --ref $TAG_GATE_REF_7B \
    --span-tokens 16,32,64,128
# If the winner is not the config's W, refit the reference at that W (still
# no GPU) and edit selection.tag.span_tokens in configs/methods/tag.yaml to match:
#   python scripts/sweep_gate_config.py --ref $TAG_GATE_REF_7B \
#       --span-tokens 32 --refit-out $POOLS/clean_ref/delta_hat_7b_W32.pt
#   export TAG_GATE_REF_7B=$POOLS/clean_ref/delta_hat_7b_W32.pt

# Completeness is a five-fold demotion decided by a string heuristic; check
# its false-positive rate on THIS pool before running four arms on it.
python scripts/audit_completeness.py --ablate \
    --pool $POOLS/composite20/pool.json

python scripts/gpu_cloud/preflight.py --config configs/experiments/lowq/tag_7b.yaml

bash scripts/precompute_gate.sh configs/experiments/lowq/tag_7b.yaml
export TAG_GATE_CACHE=$POOLS/composite20/tag_gate_qwen2.5-7b.pt

SCALE=7b bash scripts/run_lowq_all_arms.sh 42
```

## Reading the calibration output

The calibrate step prints two zero-weight rates and they mean different things.

```
RAW  Δ̂ = min(Δ̄, Δ^min):  39.6% positive | Δ̄ mean +0.1083 | Δ^min mean -0.2653
  ^ Δ̄ is healthy but the RAW tail min is not: that is Eq. 5's
    order-statistic drift (min over M spans), not a contaminated reference.
Eq.5' null correction ON (target_zero_rate=0.050): clean zero-weight rate 5.0%
per-bin clean zero-weight rate (should all be ~5.0%):
  M in [1, 3]  | n=  8123 | mu=+0.0412 | zero=5.0%
  ...
```

The **raw** line diagnoses the data: a healthy `Δ̄` with a deeply negative
`Δ^min` is the length artifact Eq. 5′ exists to remove. A *negative* `Δ̄`
would be a different problem entirely — a contaminated reference pool or
counterfactuals that are not actually unrelated — and no correction fixes
that.

The **per-bin** table is the check that the correction worked. All bins
should read ~`target_zero_rate`; a bin that drifts is under-resolved, which means
the reference pool is too small at that length.

`target_zero_rate` is the dial for "how much should the gate reject". It applies
to the *clean reference*; the candidate pool's zero-weight rate will be higher by
roughly its dirty fraction, and `compute_gate` warns if it lands far outside
that expectation.

## Why one arm per GPU, not one arm across four

Selection runs on **rank 0 only** — `tag/pipelines/selection.py` explains
why (an NCCL barrier there deadlocked earlier versions: rank 0 sits inside a
30–90 minute scoring pass while the other ranks wait in the collective and
trip the watchdog). DDP therefore accelerates the SFT step but **not** the
scoring pass, so a 4-GPU DDP run leaves three cards idle for most of a 7B
epoch. Four concurrent single-GPU arms keep every card busy.

Per-GPU memory is the same either way: DDP replicates the full model,
gradients and optimizer state on every rank. What changes is `grad_accum`,
so the effective batch stays 128:

| layout | world_size | grad_accum | env |
|---|---|---|---|
| one arm per GPU (default) | 1 | 16 | — |
| one arm across 4 DDP ranks | 4 | 4 | `TAG_GRAD_ACCUM_7B=4` |

For the DDP layout:

```bash
export TAG_GRAD_ACCUM_7B=4
torchrun --nproc_per_node=4 -m tag.train \
    --config configs/experiments/lowq/tag_7b.yaml
```

7B full fine-tuning needs 8-bit AdamW (`use_8bit_optimizer: true`, inherited
from `configs/modes/full_ft.yaml`) — fp32 optimizer state alone is ~56 GB and
does not fit beside the weights on an 80 GB card.

## The shared gate cache

`G` is a function of (pool, base checkpoint, gate config) and nothing else —
not the seed, not the arm, not the epoch. `scripts/precompute_gate.sh`
computes it once, sharded across every GPU, and every arm and seed then reads
the same file. On the paper's 8-arm × 3-seed grid that collapses 24 redundant
gate computations into one; at 7B each of those is 1+K full pool forwards.

The shards are independent processes pinned to one GPU each, not torchrun —
no process group, no rendezvous, so one dead shard is re-runnable on its own
(the merge step prints the exact command) and the job survives pre-emption.

Because a shared cache is reachable by runs it was never meant for, the
producer stamps it with the pool and backbone it was computed on and the
consumer refuses a mismatch:

```
RuntimeError: TAG gate cache at ... does not belong to this run —
model_path: cache='.../qwen2.5-0.5b' run='.../qwen2.5-7b'
```

Verify a run is actually using it — you want **zero** gate forwards at every
epoch, including the first:

```
TAG gate: cache hit (config unchanged) — no forward pass.
```

## 7B-specific gotchas

**The calibration does not transfer from 0.5B.** Δ̂ is a property of the
backbone's likelihoods, so a 7B run needs `delta_hat_7b.pt`. The bootstrap's
`calibrate7b` step produces it; pointing a 7B run at the 0.5B reference
mis-scales every gate value.

**Pick W at 0.5B, carry it to 7B.** The span sweep is cheap at 0.5B (free
with `store_token_losses`) and expensive at 7B. The 7B arms ship with
`store_token_losses: false` for that reason.

**Base, not Instruct.** If `[tag-env] model` shows a `-Instruct` path, the
backbone is already instruction-tuned — the paper's setup is SFT from a base
checkpoint, and Δ̂'s distribution differs substantially. Either download the
base weights or state the deviation.

**Disk.** Qwen2.5-7B is ~15 GB, and full-FT checkpoints are ~28 GB each. Five
arms × 3 seeds of saved final checkpoints is well over 400 GB; keep the
workspace on a scratch volume and prune as you go.
