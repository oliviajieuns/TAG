#!/usr/bin/env bash
# The grounded-pool dose-response sweep: 10/20/30/40/50 % corruption.
#
#   source scripts/gpu_cloud/env.sh
#   INPUT=<clean alpaca json>  bash scripts/run_grounded_sweep.sh pools
#   bash scripts/run_grounded_sweep.sh gates     # caches + per-rate AP report
#   bash scripts/run_grounded_sweep.sh train     # one rate at a time, 2 arms/2 GPUs
#   bash scripts/run_grounded_sweep.sh purity    # the dose-response table, no GPU
#   bash scripts/run_grounded_sweep.sh all       # pools -> gates -> train -> purity
#
# RATES="10 30 50" narrows any stage. Rates whose artifact already exists are
# skipped, so the sweep is resumable after a dead box.
#
# What each stage buys, and what it costs:
#   pools    CPU minutes.   Five pools, counterfactuals, cited manifests.
#   gates    ~15 min/rate.  The cheapest scientific readout of the whole
#            sweep: gate_report prints detection AP per corruption type at
#            each rate — the dose-response of DETECTION costs no training.
#   train    ~1 h/job.      A queue of all (rate, arm) jobs, one job per
#            visible GPU, refilled as jobs finish — 3 GPUs run the 10-job
#            sweep in 4 waves. Needed for selection purity and any eval.
#   purity   CPU seconds.   Corrupted fraction of the selected subset per
#            rate and arm — the dose-response of SELECTION, and the figure
#            the family exists for.
#
# Downstream evaluation is deliberately NOT a stage: 10 runs x ~2.5 h of
# GPU is a decision, not a default. Evaluate a rate with:
#   SET=lowq_g30 MODELS= METHODS="legacy_7b tag_prefix_7b" \
#     BENCHMARKS="mmlu,bbh,svamp,gsm8k,mbpp,tydiqa,xquad" \
#     bash scripts/run_eval_main_7b.sh --gpus 0,1 --parallel
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${POOLS:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi

STAGE="${1:-all}"
RATES="${RATES:-10 20 30 40 50}"
SEED="${SEED:-42}"
GATE_KEY="tag_gate_qwen2.5-7b_prefix.pt"

_pool_dir()  { echo "$POOLS/grounded$1"; }
_cfg()       { echo "configs/experiments/lowq_g$1/$2.yaml"; }

stage_pools() {
  if [ -z "${INPUT:-}" ]; then
    echo "[sweep] INPUT=<clean alpaca json> is required for the pools stage" >&2
    exit 2
  fi
  for R in $RATES; do
    local d; d="$(_pool_dir "$R")"
    if [ -f "$d/pool.json" ]; then
      echo "[sweep] pools: grounded$R exists — skipped"
      continue
    fi
    echo "[sweep] pools: building grounded$R"
    python scripts/make_corrupted_pool.py \
      --input "$INPUT" --out-dir "$d" --preset "grounded$R" \
      --seed "$SEED" --emit-counterfactual --emit-dedup-clusters || exit 1
  done
}

stage_gates() {
  for R in $RATES; do
    local d cache; d="$(_pool_dir "$R")"; cache="$d/$GATE_KEY"
    [ -f "$d/pool.json" ] || { echo "[sweep] gates: no pool for rate $R — run pools first" >&2; exit 2; }
    python scripts/check_row_pair.py "$(_cfg "$R" legacy_7b)" "$(_cfg "$R" tag_prefix_7b)" \
      || { echo "[sweep] gates: row pair for rate $R is NOT comparable" >&2; exit 1; }
    if [ -f "$cache" ]; then
      echo "[sweep] gates: cache for grounded$R exists — skipped"
    else
      bash scripts/precompute_gate.sh "$(_cfg "$R" tag_prefix_7b)" "$cache" || exit 1
    fi
    # The detection dose-response, one rate at a time, kept on disk.
    python scripts/gate_report.py --gate "$cache" \
      --config "$(_cfg "$R" tag_prefix_7b)" | tee "$d/gate_report.txt"
  done
}

# One (rate, arm) job on one GPU. The queue below keeps every visible GPU
# busy across rate boundaries — with 3 GPUs the 10-job sweep runs in 4
# waves instead of 5 idle-thirded ones.
_train_done() {  # rate arm -> 0 if a seed-matched run finished all 3 epochs
  local sentinel
  for sentinel in "$OUTPUT_ROOT/lowq_g$1/$2/runs/"*"seed${SEED}"/epoch_last/_complete; do
    [ -f "$sentinel" ] && [ "$(cat "$sentinel" 2>/dev/null)" = "3" ] && return 0
  done
  return 1
}

stage_train() {
  local n_gpu
  n_gpu="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  [ "$n_gpu" -ge 1 ] || { echo "[sweep] train: no GPUs visible" >&2; exit 2; }

  local logdir="$TAG_WORKSPACE/logs/sweep"
  mkdir -p "$logdir"

  # Kill the whole fleet on Ctrl-C — a half-dead sweep holding 3 GPUs is
  # the failure mode this trap exists for.
  local -a _all_pids=()
  trap 'echo "[sweep] interrupt — stopping children" >&2;
        for p in ${_all_pids[@]+"${_all_pids[@]}"}; do kill "$p" 2>/dev/null; done;
        sleep 5;
        for p in ${_all_pids[@]+"${_all_pids[@]}"}; do kill -9 "$p" 2>/dev/null; done;
        exit 130' TERM INT

  # Build the queue: every (rate, arm) not already trained to 3 epochs.
  local -a queue=()
  local R arm
  for R in $RATES; do
    for arm in legacy_7b tag_prefix_7b; do
      if _train_done "$R" "$arm"; then
        echo "[sweep] train: lowq_g$R/$arm already complete — skipped"
      else
        queue+=("$R:$arm")
      fi
    done
  done
  [ ${#queue[@]} -eq 0 ] && { echo "[sweep] train: nothing to do"; return 0; }
  echo "[sweep] train: ${#queue[@]} job(s) on $n_gpu GPU(s), 1 job/GPU"

  local -a gpu_pid=() gpu_job=()
  local g job n_fail=0

  _reap() {  # blocks until at least one GPU frees; reports its job's outcome
    wait -n 2>/dev/null || true
    for g in $(seq 0 $((n_gpu - 1))); do
      local pid="${gpu_pid[$g]:-}"
      [ -n "$pid" ] || continue
      if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid" 2>/dev/null; then
          echo "[sweep] OK     ${gpu_job[$g]}  (gpu$g)"
        else
          echo "[sweep] FAILED ${gpu_job[$g]}  (gpu$g) — see $logdir/${gpu_job[$g]//:/_}.log" >&2
          tail -n 8 "$logdir/${gpu_job[$g]//:/_}.log" 2>/dev/null | sed 's/^/         | /' >&2
          n_fail=$((n_fail + 1))
        fi
        gpu_pid[$g]=""; gpu_job[$g]=""
      fi
    done
  }

  for job in "${queue[@]}"; do
    R="${job%%:*}"; arm="${job##*:}"
    # Find a free GPU, waiting for one when all are busy.
    while :; do
      for g in $(seq 0 $((n_gpu - 1))); do
        [ -z "${gpu_pid[$g]:-}" ] && break 2
      done
      _reap
    done
    echo "[sweep] gpu$g <- lowq_g$R/$arm   ($logdir/${R}_${arm}.log)"
    CUDA_VISIBLE_DEVICES="$g" \
      python -m tag.train --config "$(_cfg "$R" "$arm")" \
        --run_suffix "seed${SEED}" \
        --override "seed=${SEED}" ${OVERRIDES:-grad_accum=16} \
        > "$logdir/${R}_${arm}.log" 2>&1 &
    gpu_pid[$g]=$!; gpu_job[$g]="$job"; _all_pids+=($!)
  done
  # Drain.
  while :; do
    local busy=0
    for g in $(seq 0 $((n_gpu - 1))); do
      [ -n "${gpu_pid[$g]:-}" ] && busy=1
    done
    [ "$busy" -eq 0 ] && break
    _reap
  done
  trap - TERM INT
  echo "[sweep] train done — ${n_fail} failure(s)"
  [ "$n_fail" -eq 0 ] || exit 1
}

stage_purity() {
  for R in $RATES; do
    local d; d="$(_pool_dir "$R")"
    echo ""
    echo "===================== grounded$R ====================="
    python scripts/selection_purity.py \
      --manifest "$d/corruption_manifest.json" \
      "legacy_7b=$OUTPUT_ROOT/lowq_g$R/legacy_7b" \
      "tag_prefix_7b=$OUTPUT_ROOT/lowq_g$R/tag_prefix_7b" || true
  done
}

case "$STAGE" in
  pools)  stage_pools ;;
  gates)  stage_gates ;;
  train)  stage_train ;;
  purity) stage_purity ;;
  all)    stage_pools; stage_gates; stage_train; stage_purity ;;
  *) echo "[error] unknown stage: $STAGE (pools|gates|train|purity|all)" >&2; exit 2 ;;
esac
