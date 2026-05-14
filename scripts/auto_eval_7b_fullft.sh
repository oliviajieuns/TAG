#!/usr/bin/env bash
# Watch <OUTPUT_ROOT>/7b_fullft/<run>/epoch_3/ to appear, then run multi-benchmark
# eval and write results under <EVAL_RESULTS_ROOT>/7b_fullft/<run>/.
#
# Usage: bash scripts/auto_eval_7b_fullft.sh <gpu_id> [run_subdirs ...]
#
# Env: relies on OUTPUT_ROOT and EVAL_RESULTS_ROOT (the same names exported by
# scripts/setup_env.sh). Source that first if you don't already have them set.
#
# Notes:
#   - This script targets the LEGACY ``7b_fullft_*`` config layout. For the
#     current main matrix (``configs/experiments/main_7b/<model>/<method>.yaml``)
#     use scripts/run_eval_main_7b.sh instead.
#   - The optional epoch_1 / epoch_2 cleanup is OPT-IN: set CLEANUP_EARLY_EPOCHS=1
#     to enable. It used to run unconditionally, which destroyed checkpoints
#     even when eval didn't actually succeed (sentinel was touched on failure).
set -euo pipefail

GPU="${1:-0}"
shift || true
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(tads_50 data_agent_50 random_50 full_100)
fi

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  # shellcheck disable=SC1091
  source "$(dirname "$0")/setup_env.sh"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-./results}"
CLEANUP_EARLY_EPOCHS="${CLEANUP_EARLY_EPOCHS:-0}"

while true; do
  for run in "${RUNS[@]}"; do
    ckpt="${OUTPUT_ROOT}/7b_fullft/${run}/epoch_3"
    out_dir="${EVAL_RESULTS_ROOT}/7b_fullft/${run}"
    done_marker="${out_dir}/.eval_done"
    if [ -d "${ckpt}" ] && [ ! -f "${done_marker}" ]; then
      mkdir -p "${out_dir}"
      cfg="configs/experiments/7b_fullft_${run}.yaml"
      if [ ! -f "${cfg}" ]; then
        echo "[skip] missing config: ${cfg}" >&2
        continue
      fi
      # Run eval. Only mark done + (optionally) clean up earlier epochs if
      # eval actually succeeded — the previous version touched the marker
      # and rm'd epochs even on a failed run, destroying recoverable state.
      if CUDA_VISIBLE_DEVICES="${GPU}" python -m tads.eval \
          --config "${cfg}" \
          --ckpt "${ckpt}" \
          --benchmarks mmlu,gsm8k,humaneval,tydiqa \
          --out_dir "${out_dir}/"; then
        touch "${done_marker}"
        if [ "${CLEANUP_EARLY_EPOCHS}" = "1" ]; then
          rm -rf "${OUTPUT_ROOT}/7b_fullft/${run}/epoch_1" \
                 "${OUTPUT_ROOT}/7b_fullft/${run}/epoch_2" || true
        fi
      else
        echo "[warn] eval failed for ${run} — leaving checkpoints intact." >&2
      fi
    fi
  done
  sleep 60
done
