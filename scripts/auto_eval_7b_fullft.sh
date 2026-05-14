#!/usr/bin/env bash
# Watch checkpoints/7b_fullft/<run>/epoch_3 to appear, then run multi-benchmark eval.
# Usage: bash scripts/auto_eval_7b_fullft.sh <gpu_id> [run_subdirs ...]
set -euo pipefail

GPU="${1:-0}"
shift || true
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(tads_50 data_agent_50 random_50 full_100)
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-./results}"

while true; do
  for run in "${RUNS[@]}"; do
    ckpt="${OUTPUT_ROOT}/7b_fullft/${run}/epoch_3"
    done_marker="${RESULTS_ROOT}/7b_fullft/${run}/.eval_done"
    if [ -d "${ckpt}" ] && [ ! -f "${done_marker}" ]; then
      mkdir -p "${RESULTS_ROOT}/7b_fullft/${run}"
      CUDA_VISIBLE_DEVICES="${GPU}" python -m tads.eval \
        --config "configs/experiments/7b_fullft_$(echo ${run} | sed 's/_/_/').yaml" \
        --ckpt "${ckpt}" \
        --benchmarks mmlu,gsm8k,humaneval,tydiqa \
        --out_dir "${RESULTS_ROOT}/7b_fullft/${run}/" \
        || true
      touch "${done_marker}"
      # Optional: remove epoch_1/epoch_2 once epoch_3 evaluated.
      rm -rf "${OUTPUT_ROOT}/7b_fullft/${run}/epoch_1" "${OUTPUT_ROOT}/7b_fullft/${run}/epoch_2" || true
    fi
  done
  sleep 60
done
