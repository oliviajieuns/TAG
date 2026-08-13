#!/usr/bin/env bash
# One-shot setup for a fresh GPU box: dependencies, weights, data, pools,
# and the gate calibration. Idempotent — every step skips if already done,
# so re-running after a failure resumes rather than redoing.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/gpu_cloud/bootstrap.sh [step]
#
#     step: deps | model | data | pools | calibrate | all   (default: all)
#
# Needs outbound network for `deps`, `model`, and `data`. The remaining
# steps are offline. Total: a few minutes plus ~1 GB of downloads; the
# calibrate step is the only one that touches the GPU (two pool forwards).
set -euo pipefail

STEP="${1:-all}"

if [ -z "${TAG_WORKSPACE:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi
cd "$TAG_REPO_ROOT"

log() { echo "[bootstrap] $*" >&2; }

step_deps() {
  log "installing python dependencies"
  python -m pip install -q --upgrade pip
  # torch first: on most GPU images it is preinstalled with the right CUDA
  # build, and letting pip resolve it from our requirements can silently
  # swap in a CPU wheel.
  if python -c "import torch" 2>/dev/null; then
    log "torch already present: $(python -c 'import torch; print(torch.__version__)')"
  else
    log "torch not found — installing the default CUDA wheel"
    python -m pip install -q torch
  fi
  python -m pip install -q -r requirements.txt
  python -m pip install -q pytest
  log "deps done"
}

step_model() {
  if [ -f "$MODEL_PATH_QWEN25_05B/config.json" ]; then
    log "model already at $MODEL_PATH_QWEN25_05B — skipping"
    return
  fi
  log "downloading Qwen2.5-0.5B (base) -> $MODEL_PATH_QWEN25_05B"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    bash scripts/download_qwen25_05b.sh "$MODEL_PATH_QWEN25_05B"
}

step_data() {
  if [ -f "$ALPACA_RAW_JSON" ]; then
    log "raw corpus already at $ALPACA_RAW_JSON — skipping"
    return
  fi
  log "downloading Alpaca-GPT4 -> $ALPACA_RAW_JSON"
  mkdir -p "$(dirname "$ALPACA_RAW_JSON")"
  # vicgalle/alpaca-gpt4 ships the standard instruction/input/output columns.
  # The liangxin mirror uses a non-standard `conversations` schema that the
  # Alpaca loader does not read.
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python - "$ALPACA_RAW_JSON" <<'PY'
import json, sys
from datasets import load_dataset
out = sys.argv[1]
ds = load_dataset("vicgalle/alpaca-gpt4")["train"]
recs = [
    {"instruction": r["instruction"], "input": r.get("input") or "", "output": r["output"]}
    for r in ds
]
with open(out, "w") as f:
    json.dump(recs, f, ensure_ascii=False)
print(f"[done] {len(recs)} records -> {out}")
PY
}

step_pools() {
  if [ -f "$POOLS/composite20/pool.json" ] && [ -f "$POOLS/clean_ref/counterfactual.json" ]; then
    log "pools already generated — skipping"
    return
  fi
  log "generating the corrupted pool -> $POOLS/composite20"
  python scripts/make_corrupted_pool.py \
    --input "$ALPACA_RAW_JSON" --out-dir "$POOLS/composite20" \
    --preset composite20 --duplicate-frac 0.05 --seed 42 \
    --emit-counterfactual --emit-dedup-clusters

  # The calibration pool must be CLEAN: no --preset and no corruption
  # fractions, so only the counterfactual pairing is emitted. Calibrating on
  # a corrupted pool would set s from contaminated Delta_hat and quietly
  # disable the gate.
  log "generating the CLEAN reference pool -> $POOLS/clean_ref"
  python scripts/make_corrupted_pool.py \
    --input "$ALPACA_RAW_JSON" --out-dir "$POOLS/clean_ref" \
    --emit-counterfactual --seed 42
}

step_calibrate() {
  if [ -f "$TADS_GATE_REF" ]; then
    log "gate reference already at $TADS_GATE_REF — skipping"
    return
  fi
  log "calibrating the gate scale on the clean pool (GPU, two pool forwards)"
  python scripts/calibrate_reliability.py --mode tag \
    --config configs/experiments/lowq/light_tag_05b.yaml \
    --pool "$POOLS/clean_ref/pool.json" \
    --counterfactual "$POOLS/clean_ref/counterfactual.json" \
    --out "$TADS_GATE_REF"
}

case "$STEP" in
  deps)      step_deps ;;
  model)     step_model ;;
  data)      step_data ;;
  pools)     step_pools ;;
  calibrate) step_calibrate ;;
  all)       step_deps; step_model; step_data; step_pools; step_calibrate ;;
  *) echo "unknown step: $STEP (deps|model|data|pools|calibrate|all)" >&2; exit 2 ;;
esac

log "step '$STEP' complete"
if [ "$STEP" = "all" ]; then
  echo ""
  echo "Next:"
  echo "  python scripts/gpu_cloud/preflight.py      # verify before burning GPU hours"
  echo "  bash scripts/run_tag_lowq_05b.sh smoke     # ~2 min end-to-end on a subset"
  echo "  bash scripts/run_tag_lowq_05b.sh phasea    # detection table"
  echo "  bash scripts/run_tag_lowq_05b.sh phaseb    # full SFT run"
fi
