#!/usr/bin/env bash
# Evaluate ONE arm by splitting its benchmarks across GPUs.
#
#   source scripts/gpu_cloud/env.sh
#   MODELS=llama2 METHODS=tag_10 GPUS=0,1,2,3 bash scripts/run_eval_sharded.sh
#
# run_eval_main_7b.sh --parallel fans out over (model x method): with a single
# arm it launches ONE job that walks all eight benchmarks on one GPU, and the
# other GPUs sit idle. That is the right shape for a full matrix and the wrong
# one for "re-run one row and read the number". This script shards the other
# way — one process per GPU, each taking a slice of the benchmark list.
#
# The slices share ONE eval run dir, via a single --eval_tag computed here and
# passed to every shard. That matters: make_table_v2 reads exactly the run dir
# `_latest` names (read_run_dir globs *.json in it), so eight per-shard run
# dirs would each contribute a fraction of a row and the table would see only
# whichever one `_latest` happened to point at. Sharing the tag puts every
# per-bench JSON in one dir, which is what the aggregator expects.
#
# Each shard also writes <label>-eval_summary.json covering only its own
# benches. That is harmless — read_run_dir takes readings from the per-bench
# files and the summaries alike, and the values agree — but it does mean the
# summary file is NOT a whole-run summary when this script is used. Read
# t2_status.py / make_table_v2.py for completeness, not that file.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${TAG_WORKSPACE:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi
python -c "import torch" >/dev/null 2>&1 || {
  echo "[error] this python cannot import torch — activate the venv first:" >&2
  echo "        source /group-volume/jieuns.shin/venvs/exp/bin/activate" >&2
  exit 2
}

SET=${SET:-main_7b}
MODELS=${MODELS:-llama2}
METHODS=${METHODS:-tag_10}
BENCHMARKS=${BENCHMARKS:-"mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad"}
GPUS=${GPUS:-"0"}
LIMIT=${LIMIT:-}
# One tag for every shard. Override to add slices to an existing run dir.
EVAL_TAG=${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)        GPUS="$2"; shift 2 ;;
    --gpus=*)      GPUS="${1#*=}"; shift ;;
    --benchmarks)  BENCHMARKS="$2"; shift 2 ;;
    --eval_tag)    EVAL_TAG="$2"; shift 2 ;;
    --limit)       LIMIT="$2"; shift 2 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a _gpus <<< "$GPUS"
IFS=',' read -r -a _benches <<< "$BENCHMARKS"
n_gpus=${#_gpus[@]}

if ! python scripts/check_eval_data.py --benchmarks "$BENCHMARKS"; then
  echo "[error] benchmark data is not ready — not launching." >&2
  exit 2
fi

seg=""
[ "$MODELS" != "-" ] && seg="/${MODELS}"
CFG="configs/experiments/${SET}${seg}/${METHODS}.yaml"
CKPT_ROOT="${OUTPUT_ROOT}/${SET}${seg}/${METHODS}"
OUT_DIR="${EVAL_RESULTS_ROOT}/${SET}${seg}/${METHODS}"
[ -f "$CFG" ] || { echo "[error] missing config: $CFG" >&2; exit 2; }

# Newest sealed epoch under the newest run dir — same precedence as
# run_eval_main_7b.sh's latest_epoch().
latest_epoch() {
  local run_dir="$1"
  if [ -f "${run_dir}/epoch_last/_complete" ]; then
    echo "${run_dir}/epoch_last"; return
  fi
  local last=""
  while IFS= read -r p; do
    [ "$(basename "$p")" = "epoch_last" ] && continue
    [ -f "${p}/_complete" ] && last="$p"
  done < <(ls -1d "${run_dir}"/epoch_* 2>/dev/null | sort -V)
  echo "$last"
}

RUN_DIR="$(ls -1d "${CKPT_ROOT}"/runs/* 2>/dev/null | sort | tail -n 1)"
[ -n "$RUN_DIR" ] || { echo "[error] no training run under ${CKPT_ROOT}/runs" >&2; exit 2; }
CKPT="$(latest_epoch "$RUN_DIR")"
[ -n "$CKPT" ] || { echo "[error] no sealed epoch under $RUN_DIR" >&2; exit 2; }

echo "[eval-shard] arm      : ${SET}${seg}/${METHODS}"
echo "[eval-shard] ckpt     : $CKPT"
echo "[eval-shard] out      : $OUT_DIR/runs/$EVAL_TAG"
echo "[eval-shard] gpus     : $GPUS   benches: $BENCHMARKS"

mkdir -p logs
pids=()
names=()
for i in "${!_gpus[@]}"; do
  # Round-robin the benches onto this GPU, then hand them to ONE process:
  # two eval processes on one card would just contend for its memory.
  slice=""
  for j in "${!_benches[@]}"; do
    [ $((j % n_gpus)) -eq "$i" ] && slice="${slice:+$slice,}${_benches[$j]}"
  done
  [ -n "$slice" ] || continue
  gpu="${_gpus[$i]}"
  log="logs/eval_${SET}_${MODELS}_${METHODS}_gpu${gpu}.log"
  echo "[eval-shard] gpu${gpu} <- ${slice}"
  extra=()
  [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tag.eval \
    --config "$CFG" --ckpt "$CKPT" --benchmarks "$slice" \
    --out_dir "$OUT_DIR" --eval_tag "$EVAL_TAG" --cuda_device 0 \
    "${extra[@]}" >> "$log" 2>&1 &
  pids+=($!)
  names+=("gpu${gpu}:${slice}")
done

echo "[eval-shard] launched ${#pids[@]} shard(s); tail logs/eval_${SET}_${MODELS}_${METHODS}_gpu*.log"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[eval-shard] done  ${names[$i]}"
  else
    echo "[eval-shard] FAILED ${names[$i]}" >&2
    fail=1
  fi
done

echo "[eval-shard] run dir: $OUT_DIR/runs/$EVAL_TAG"
echo "[eval-shard] verify : python scripts/t2_status.py"
exit $fail
