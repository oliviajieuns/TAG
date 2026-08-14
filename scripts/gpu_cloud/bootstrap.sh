#!/usr/bin/env bash
# One-shot setup for a fresh GPU box: dependencies, weights, data, pools,
# and the gate calibration. Idempotent — every step skips if already done,
# so re-running after a failure resumes rather than redoing.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/gpu_cloud/bootstrap.sh [step]
#
#     step: deps | model | data | pools | calibrate | all       (0.5B)
#           model7b | calibrate7b | all7b                       (7B)
#     (default: all)
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
  # Always materialise Alpaca-GPT4 to a FIXED workspace path, independent of
  # whatever ALPACA_RAW_JSON currently points at: discovery may have found a
  # different corpus (alpaca-cleaned), and silently keeping that one would
  # break comparability with base.yaml and the prior TADS numbers.
  local out="$TAG_WORKSPACE/datasets/alpaca_gpt4.json"
  if [ -f "$out" ]; then
    log "Alpaca-GPT4 already at $out — skipping"
    log "  export ALPACA_RAW_JSON=$out"
    return
  fi
  mkdir -p "$(dirname "$out")"

  # Offline first: on a cluster the dataset is usually already in HF_HOME, and
  # the compute nodes often have no egress at all.
  if python - "$out" <<'DSPY'
import os, sys
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json
from datasets import load_dataset
out = sys.argv[1]
for repo in ("vicgalle/alpaca-gpt4", "liangxin/Alpaca_GPT4"):
    try:
        ds = load_dataset(repo)["train"]
    except Exception:
        continue
    cols = set(ds.column_names)
    if not {"instruction", "output"} <= cols:
        # liangxin ships a `conversations` schema the Alpaca loader cannot read.
        print(f"[skip] {repo}: columns {sorted(cols)}", file=sys.stderr)
        continue
    recs = [{"instruction": r["instruction"], "input": r.get("input") or "",
             "output": r["output"]} for r in ds]
    with open(out, "w") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"[done] {repo}: {len(recs)} records -> {out} (from cache, offline)")
    sys.exit(0)
sys.exit(1)
DSPY
  then
    log "Alpaca-GPT4 materialised from the local HF cache — no download"
  else
    log "not in the HF cache; downloading Alpaca-GPT4"
    HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python - "$out" <<'DLPY'
import json, sys
from datasets import load_dataset
out = sys.argv[1]
ds = load_dataset("vicgalle/alpaca-gpt4")["train"]
recs = [{"instruction": r["instruction"], "input": r.get("input") or "",
         "output": r["output"]} for r in ds]
with open(out, "w") as f:
    json.dump(recs, f, ensure_ascii=False)
print(f"[done] {len(recs)} records -> {out}")
DLPY
  fi
  log "Alpaca-GPT4 ready at $out"
  if [ "${ALPACA_RAW_JSON:-}" != "$out" ]; then
    log ""
    log "  ALPACA_RAW_JSON currently points at:"
    log "    ${ALPACA_RAW_JSON:-<unset>}"
    log "  To use Alpaca-GPT4 (what base.yaml and prior TADS numbers use):"
    log "    export ALPACA_RAW_JSON=$out"
    log "    bash scripts/gpu_cloud/n9_discover.sh --write   # or re-detect"
    log ""
  fi
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

step_model_7b() {
  if [ -f "$MODEL_PATH_QWEN25_7B/config.json" ]; then
    log "7B model already at $MODEL_PATH_QWEN25_7B — skipping"
    return
  fi
  log "downloading Qwen2.5-7B (base) -> $MODEL_PATH_QWEN25_7B (~15 GB)"
  mkdir -p "$(dirname "$MODEL_PATH_QWEN25_7B")"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python - "$MODEL_PATH_QWEN25_7B" <<'HFPY'
import sys
from huggingface_hub import snapshot_download
target = sys.argv[1]
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B", local_dir=target, local_dir_use_symlinks=False,
    allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*",
                    "merges.txt", "vocab.json", "*.model"],
)
print(f"[done] Qwen/Qwen2.5-7B -> {target}")
HFPY
}

step_calibrate_7b() {
  # Delta_hat is a property of THIS backbone's likelihoods, so a 0.5B
  # reference does not transfer — 7B needs its own calibration.
  local out="$POOLS/clean_ref/delta_hat_7b.pt"
  # Skip only if the artifact is COMPLETE. References written before the
  # token-loss payload existed look fine on disk but make the span-width
  # sweep impossible, and silently skipping them means discovering that
  # after the next 28-minute pass rather than before it.
  if [ -f "$out" ]; then
    if python - "$out" <<'CHK'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if "token_true" in p and "token_cf" in p else 1)
CHK
    then
      log "7B gate reference already at $out (with token losses) — skipping"
      return
    fi
    log "7B gate reference at $out predates the token-loss payload;"
    log "  recomputing so scripts/sweep_gate_config.py can re-derive Δ̂."
    mv "$out" "$out.notokens.bak"
  fi
  log "calibrating the 7B gate scale on the clean pool"
  TADS_GATE_REF="" python scripts/calibrate_reliability.py --mode tag \
    --config configs/experiments/lowq/tag_7b.yaml \
    --pool "$POOLS/clean_ref/pool.json" \
    --counterfactual "$POOLS/clean_ref/counterfactual.json" \
    --out "$out"
  # NOT TADS_GATE_REF — that one is the 0.5B reference. env.sh already
  # defaults TADS_GATE_REF_7B to this path; the echo is just a reminder of
  # which variable the 7B arms actually read.
  log "7B gate reference -> $out   (read via TADS_GATE_REF_7B)"

  # The Eq. 5' ablation arm needs a reference fit WITHOUT the correction.
  # It costs no GPU: the artifact above kept the per-token NLLs, so the
  # uncorrected calibration is re-derived from them.
  local nonull="$POOLS/clean_ref/delta_hat_7b_nonull.pt"
  if [ ! -f "$nonull" ]; then
    log "deriving the no-correction ablation reference (CPU, no forward)"
    python scripts/sweep_gate_config.py --ref "$out" \
      --span-tokens "$(python - "$out" <<'W'
import sys, torch
print(int((torch.load(sys.argv[1], map_location="cpu", weights_only=False)
           .get("gate_config") or {}).get("span_tokens", 16)))
W
)" --no-null-correction --refit-out "$nonull" >/dev/null \
      && log "ablation reference -> $nonull   (read via TADS_GATE_REF_7B_NONULL)" \
      || log "WARNING: ablation reference not written; tag_nonull_7b will not run"
  fi
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
  deps)         step_deps ;;
  model)        step_model ;;
  model7b)      step_model_7b ;;
  data)         step_data ;;
  pools)        step_pools ;;
  calibrate)    step_calibrate ;;
  calibrate7b)  step_calibrate_7b ;;
  all)          step_deps; step_model; step_data; step_pools; step_calibrate ;;
  all7b)        step_deps; step_model_7b; step_data; step_pools; step_calibrate_7b ;;
  *) echo "unknown step: $STEP (deps|model|model7b|data|pools|calibrate|calibrate7b|all|all7b)" >&2; exit 2 ;;
esac

log "step '$STEP' complete"
if [ "$STEP" = "all" ]; then
  echo ""
  echo "Next (0.5B):"
  echo "  python scripts/gpu_cloud/preflight.py      # verify before burning GPU hours"
  echo "  bash scripts/run_tag_lowq_05b.sh smoke     # ~2 min end-to-end on a subset"
  echo "  bash scripts/run_tag_lowq_05b.sh phasea    # detection table"
  echo "  bash scripts/run_lowq_all_arms.sh 42       # 4 arms, one per GPU"
fi
if [ "$STEP" = "all7b" ]; then
  echo ""
  echo "Next (7B):"
  echo "  # env.sh already set TADS_GATE_REF_7B=$POOLS/clean_ref/delta_hat_7b.pt"
  echo "  export TADS_EPISODE_BS_7B=32"
  echo "  # choose W from the calibration (CPU, seconds — no forward pass):"
  echo "  python scripts/sweep_gate_config.py --ref \$TADS_GATE_REF_7B \\"
  echo "      --span-tokens 16,32,64,128"
  echo "  # check the completeness heuristic's false-positive rate on this pool:"
  echo "  python scripts/audit_completeness.py --ablate \\"
  echo "      --pool $POOLS/composite20/pool.json \\"
  echo "      --manifest $POOLS/composite20/manifest.json"
  echo "  python scripts/gpu_cloud/preflight.py --config configs/experiments/lowq/tag_7b.yaml"
  echo "  bash scripts/precompute_gate.sh configs/experiments/lowq/tag_7b.yaml"
  echo "      # ^ shards the gate across every GPU, once; then:"
  echo "  export TADS_GATE_CACHE=$POOLS/composite20/tag_gate_qwen2.5-7b.pt"
  echo "  SCALE=7b bash scripts/run_lowq_all_arms.sh 42"
fi
