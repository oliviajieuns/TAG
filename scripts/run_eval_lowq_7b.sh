#!/usr/bin/env bash
# Evaluate the low-quality-pool 7B arms and lay the results out so
# scripts/make_table_v2.py can aggregate them without a hand-written manifest.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/run_eval_lowq_7b.sh [seed] [arm ...]
#
# One arm per GPU, concurrently, same as run_lowq_all_arms.sh. Results land at
#
#   $EVAL_RESULTS_ROOT/lowq/qwen25-7b/<arm>/runs/<tag>/
#
# which is the <root>/<set>/<model>/<method>/runs/ shape make_table_v2's
# --results-root walks. The eval tag carries the seed so the table can pair
# arms seed-by-seed even when a training cfg.yaml is not readable from the
# eval side.
#
# Then:
#   python scripts/make_table_v2.py --results-root "$EVAL_RESULTS_ROOT" \
#       --benches mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad \
#       --pairs tag_prefix_7b:legacy_7b
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${1:-42}"
shift || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(tag_prefix_7b legacy_7b)

# The paper's Table 2, in its order.
BENCHMARKS="${BENCHMARKS:-mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad}"
LIMIT="${LIMIT:-}"
ARM_DIR="${ARM_DIR:-configs/experiments/lowq}"
SET_NAME="${SET_NAME:-lowq}"
MODEL_NAME="${MODEL_NAME:-qwen25-7b}"

if [ -z "${OUTPUT_ROOT:-}" ] || [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first (need OUTPUT_ROOT and EVAL_RESULTS_ROOT)" >&2
  exit 2
fi

# Refuse to start rather than fail one benchmark deep into a 7B eval. The
# check is on the FILES each evaluator opens, not on the directory existing —
# MMLU's shards live under all/ on some mirrors and directly in the dir on
# others, and pointing at the wrong one only surfaces an hour in.
if ! python scripts/check_eval_data.py --benchmarks "$BENCHMARKS"; then
  exit 2
fi

N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "$N_GPU" -eq 0 ] && { echo "[error] no GPUs visible" >&2; exit 2; }

LOGDIR="$TAG_WORKSPACE/logs/eval_lowq_7b_seed${SEED}"
mkdir -p "$LOGDIR"

_pids_all=()
_cleanup() {
  trap '' TERM INT
  echo "" >&2
  echo "[cleanup] stopping ${#_pids_all[@]} eval job(s)..." >&2
  for _p in ${_pids_all[@]+"${_pids_all[@]}"}; do kill "$_p" 2>/dev/null || true; done
  sleep 5
  for _p in ${_pids_all[@]+"${_pids_all[@]}"}; do kill -9 "$_p" 2>/dev/null || true; done
  exit 130
}
trap _cleanup TERM INT

echo "[eval] seed=$SEED  gpus=$N_GPU  benches=$BENCHMARKS"
echo "[eval] results -> $EVAL_RESULTS_ROOT/$SET_NAME/$MODEL_NAME/<arm>/"
echo "[eval] logs    -> $LOGDIR"

pids=(); names=(); i=0
for arm in "${ARMS[@]}"; do
  cfg="$ARM_DIR/${arm}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[eval] SKIP $arm — no such config: $cfg" >&2
    i=$((i+1)); continue
  fi
  # The checkpoint the arm actually produced. tag.train resolves _latest
  # itself when neither --ckpt nor --run_tag is given, which is what we want:
  # it picks the run this seed just wrote.
  gpu=$(( i % N_GPU ))
  out="$EVAL_RESULTS_ROOT/$SET_NAME/$MODEL_NAME/$arm"
  log="$LOGDIR/${arm}.log"
  echo "[eval] gpu$gpu <- $arm   ($log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m tag.eval --config "$cfg" \
      --benchmarks "$BENCHMARKS" \
      --out_dir "$out" \
      --eval_suffix "seed${SEED}" \
      ${LIMIT:+--limit "$LIMIT"} \
      > "$log" 2>&1 &
  pids+=($!); _pids_all+=($!); names+=("$arm")
  i=$((i+1))
done

if [ ${#pids[@]} -eq 0 ]; then
  echo "[eval] nothing launched" >&2
  exit 2
fi

echo "[eval] ${#pids[@]} job(s) launched; waiting..."
fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[eval] OK   ${names[$j]}"
  else
    echo "[eval] FAIL ${names[$j]} — see $LOGDIR/${names[$j]}.log" >&2
    fail=$((fail+1))
  fi
done

echo ""
if [ "$fail" -ne 0 ]; then
  echo "[eval] $fail job(s) failed; the table below will be missing those rows." >&2
fi
echo "[eval] build the table with:"
echo "  python scripts/make_table_v2.py --results-root \"$EVAL_RESULTS_ROOT\" \\"
echo "      --benches $BENCHMARKS --pairs tag_prefix_7b:legacy_7b"
exit "$fail"
