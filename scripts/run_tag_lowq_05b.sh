#!/usr/bin/env bash
# TAG on a low-quality pool, Qwen2.5-0.5B — the full ordered sequence.
#
#   0. pools          corrupted pool + manifest + counterfactual + dedup, and
#                     a CLEAN reference pool for calibration
#   1. calibrate      clean-reference Delta_hat -> the gate scale s (Eq. 6)
#   2. phase A        forward-only detection: Dirty@K / AUPRC per signal
#   3. phase B        end-to-end SFT for the TAG arm (and its controls)
#
# Step 1 is NOT optional for a reported run. Without it the gate falls back to
# in-pool self-calibration, which makes G depend on how dirty its neighbours
# are — exactly the pool dependence that anchoring at Delta_hat = 0 removes.
# The fallback logs a loud warning and is diagnostic-only.
#
# Usage:
#   bash scripts/run_tag_lowq_05b.sh <stage> [gpu_id]
#     stage: pools | calibrate | phasea | phaseb | all
#
# Env (source scripts/setup_env.sh first, then override as needed):
#   ALPACA_RAW_JSON  clean source corpus for pool generation
#   POOLS            root for generated pools           (default: ./pools)
#   OUTPUT_ROOT      run output root
set -euo pipefail

STAGE="${1:-all}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POOLS="${POOLS:-$REPO_ROOT/pools}"
ALPACA_RAW_JSON="${ALPACA_RAW_JSON:?set ALPACA_RAW_JSON to the clean source corpus}"
POOL_DIR="$POOLS/composite20"
CLEAN_DIR="$POOLS/clean_ref"
CFG_TAG="configs/experiments/lowq/light_tag_05b.yaml"

# The gate reference and the pool wiring must be exported for EVERY stage:
# the configs read them through ${oc.env:...}.
export ALPACA_DATA_FILES="$POOL_DIR/pool.json"
export TADS_CF_FILES="$POOL_DIR/counterfactual.json"
export TADS_DEDUP_FILE="$POOL_DIR/dedup_clusters.json"
export TADS_GATE_REF="${TADS_GATE_REF:-$CLEAN_DIR/delta_hat_05b.pt}"

log() { echo "[run_tag_lowq_05b] $*" >&2; }

stage_pools() {
  log "generating corrupted pool -> $POOL_DIR"
  python scripts/make_corrupted_pool.py \
    --input "$ALPACA_RAW_JSON" --out-dir "$POOL_DIR" \
    --preset composite20 --duplicate-frac 0.05 --seed 42 \
    --emit-counterfactual --emit-dedup-clusters

  # The calibration pool must be CLEAN: no --preset, no corruption fractions.
  # Only the counterfactual pairing is emitted.
  log "generating clean reference pool -> $CLEAN_DIR"
  python scripts/make_corrupted_pool.py \
    --input "$ALPACA_RAW_JSON" --out-dir "$CLEAN_DIR" \
    --emit-counterfactual --seed 42
}

stage_calibrate() {
  log "calibrating the gate scale s on the CLEAN reference pool"
  # --mode tag emits {'delta_hat': ...}; the MVF artifact ({'delta': ...}, raw
  # nats) is NOT interchangeable and the loader rejects it.
  python scripts/calibrate_reliability.py --mode tag \
    --config "$CFG_TAG" \
    --pool "$CLEAN_DIR/pool.json" \
    --counterfactual "$CLEAN_DIR/counterfactual.json" \
    --out "$TADS_GATE_REF"
  log "gate reference written to $TADS_GATE_REF"
}

stage_phasea() {
  log "phase A: forward-only detection (no training)"
  # --save-signals keeps the per-sample vectors the figures need; the JSON
  # report alone is aggregate and cannot produce a reliability diagram.
  python scripts/score_pool.py \
    --config "$CFG_TAG" \
    --manifest "$POOL_DIR/corruption_manifest.json" \
    --out "$POOL_DIR/score_report_tag.json" \
    --save-signals "$POOL_DIR/signals_tag.pt" \
    --ks 0.05,0.1,0.2 \
    ${UNCOND_LOSS:+--uncond-loss "$UNCOND_LOSS"}
  log "report: $POOL_DIR/score_report_tag.json"
  python - <<'PY'
import json, os, sys
p = os.path.join(os.environ["POOLS"], "composite20", "score_report_tag.json")
r = json.load(open(p))
tag = r.get("tag", {})
print(f"\n  dirty base rate: {r['dirty_base_rate']:.3f}")
print(f"  gate: mean={tag.get('gate_mean')} zero_frac={tag.get('gate_zero_frac')}")
print(f"  admissible: {tag.get('n_admissible')}/{r['n']} "
      f"({tag.get('admissible_frac', 0):.1%})")
for k, v in tag.items():
    if k.startswith("budget_fits@") and not v:
        print(f"  !! {k} is FALSE — the budget exceeds the gated set; the "
              f"veto cannot hold for every slot at that ratio.")
rows = sorted(r["signals"].items(),
              key=lambda kv: kv[1]["ap_dirty_from_rejection"], reverse=True)
print(f"\n  {'signal':<14} {'AP(dirty)':>10} {'dirty@0.1':>10}")
for name, e in rows:
    print(f"  {name:<14} {e['ap_dirty_from_rejection']:>10.4f} "
          f"{e.get('dirty@0.1', float('nan')):>10.3f}")
PY
}

stage_phaseb() {
  log "phase B: end-to-end SFT (TAG arm)"
  python -m tads.train --config "$CFG_TAG"
}

case "$STAGE" in
  pools)     stage_pools ;;
  calibrate) stage_calibrate ;;
  phasea)    stage_phasea ;;
  phaseb)    stage_phaseb ;;
  all)       stage_pools; stage_calibrate; stage_phasea; stage_phaseb ;;
  *) echo "unknown stage: $STAGE (pools|calibrate|phasea|phaseb|all)" >&2; exit 2 ;;
esac
log "stage '$STAGE' done"
