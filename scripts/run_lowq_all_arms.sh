#!/usr/bin/env bash
# Run the lowq 0.5B arms in parallel, one per GPU.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/run_lowq_all_arms.sh [seed] [arm ...]
#
# OVERRIDES is passed through to `tag.train --override`, the same way
# run_main_7b.sh does it. The one that matters here: the 7B configs size
# grad_accum for DDP x 4 (8 x 4 x 4 = effective batch 128), so a
# 1-arm-per-GPU launch needs grad_accum=16 to train on the same effective
# batch:
#
#   OVERRIDES="grad_accum=16" ARM_DIR=configs/experiments/main_7b/llama2 \
#     SCALE=7b bash scripts/run_lowq_all_arms.sh 42 legacy_10 tag_10
#
# It applies to EVERY arm in the launch by construction, which is the
# property that keeps the arms comparable.
#
# At 0.5B a single H100 is far faster than DDP across four of them (the
# gradient sync dominates), so the right use of a 4-GPU box is four
# CONCURRENT arms rather than one distributed arm.
#
# Every arm inherits its training hyperparameters from
# _shared_light_05b.yaml, and TAG_EPISODE_BS is read from THIS shell, so
# all arms launched here share it by construction — which is the property
# plan §5.2 demands. Launching a stray arm later with a different value
# would break comparability; check cfg.yaml in each run dir if unsure.
set -uo pipefail

SEED="${1:-42}"
shift || true

# SCALE=7b runs the 7B grid; anything else runs the 0.5B one.
SCALE="${SCALE:-05b}"
if [ "$SCALE" = "7b" ]; then
  DEFAULT_ARMS=(tag_7b tag_static_7b legacy_7b random_7b)
else
  DEFAULT_ARMS=(
    light_tag_05b
    light_tag_static_05b
    light_legacy_05b
    light_random_05b
  )
fi
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${DEFAULT_ARMS[@]}")

# Where the arm configs live. The clean-pool control pair sits in
# configs/experiments/clean/ and is otherwise launched identically:
#   ARM_DIR=configs/experiments/clean bash scripts/run_lowq_all_arms.sh 42 \
#       tag_prefix_7b legacy_7b
ARM_DIR="${ARM_DIR:-configs/experiments/lowq}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${TAG_WORKSPACE:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi

# A shell that sourced env.sh but not the venv launches N processes that all
# die on `import torch` — after grabbing their GPUs. Refuse up front instead.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[error] this python cannot import torch — activate the venv first" >&2
  echo "        (source exp/bin/activate, then re-run)" >&2
  exit 2
fi

# Kill every child on Ctrl-C / SIGTERM. Without this a 7B arm that is
# interrupted — or one that hangs tearing down its CUDA context after an OOM,
# which is the common case — is left holding tens of GB on its GPU with no
# obvious owner, and the next launch OOMs for reasons that look unrelated.
_pids_all=()
_cleanup() {
  trap '' TERM INT
  echo "" >&2
  echo "[cleanup] stopping ${#_pids_all[@]} child process(es)..." >&2
  for _p in ${_pids_all[@]+"${_pids_all[@]}"}; do
    kill "$_p" 2>/dev/null || true
  done
  # Give CUDA a moment to release, then insist.
  sleep 5
  for _p in ${_pids_all[@]+"${_pids_all[@]}"}; do
    kill -9 "$_p" 2>/dev/null || true
  done
  echo "[cleanup] done; check nvidia-smi before relaunching." >&2
  exit 130
}
trap _cleanup TERM INT

N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "$N_GPU" -eq 0 ] && { echo "[error] no GPUs visible" >&2; exit 2; }

LOGDIR="$TAG_WORKSPACE/logs/arms_${SCALE}_seed${SEED}"
mkdir -p "$LOGDIR"

# A shared gate cache turns N redundant gate computations into zero: every
# TAG arm reads a file that was computed once. Without it each arm recomputes
# the gate on its own GPU, which at 7B is 1+K full pool forwards per arm.
#
# Which variable holds that file is PER ARM — the prefix and ablation arms
# each have their own G and so their own cache — so checking only
# TAG_GATE_CACHE stayed silent for exactly the arms most likely to be
# mis-wired. Read the variable name out of each config being launched.
for arm in "${ARMS[@]}"; do
  cfg="$ARM_DIR/${arm}.yaml"
  [ -f "$cfg" ] || continue
  var="$(sed -n 's/.*gate_cache_file:[[:space:]]*\${oc\.env:\([A-Z_0-9]*\).*/\1/p' "$cfg" | head -1)"
  [ -n "$var" ] || continue
  path="${!var:-}"
  if [ -z "$path" ]; then
    echo "[arms] WARNING: $arm reads \$$var, which is unset — it will compute" >&2
    echo "[arms]          the gate itself (1+K pool forwards). source env.sh?" >&2
  elif [ -f "$path" ]; then
    echo "[arms] gate cache: $arm <- \$$var = $path"
  else
    echo "[arms] WARNING: $arm reads \$$var = $path — that file does not exist." >&2
    echo "[arms]          It will recompute the gate. Run:" >&2
    echo "[arms]            bash scripts/precompute_gate.sh $cfg \$$var" >&2
  fi
done
unset arm cfg var path

echo "[arms] scale=$SCALE seed=$SEED gpus=$N_GPU arm_dir=$ARM_DIR"
if [ "$SCALE" = "7b" ]; then
  echo "[arms] episode_bs=${TAG_EPISODE_BS_7B:-8} (1 arm/GPU)"
  echo "[arms] overrides=${OVERRIDES:-<none>}"
  if [ "${OVERRIDES:-}" = "${OVERRIDES#*grad_accum}" ]; then
    echo "[arms] WARNING: no grad_accum override. The 7B configs set" >&2
    echo "[arms]          grad_accum=4 for DDP x 4 (effective batch 128); at" >&2
    echo "[arms]          1 arm/GPU that trains on 32, not 128. Pass" >&2
    echo "[arms]          OVERRIDES=\"grad_accum=16\" unless you mean it." >&2
  fi
else
  echo "[arms] episode_bs=${TAG_EPISODE_BS:-2}"
fi
echo "[arms] logs -> $LOGDIR"

if [ ${#ARMS[@]} -gt "$N_GPU" ]; then
  echo "[arms] note: ${#ARMS[@]} arms on $N_GPU GPUs — they will share cards." >&2
  echo "[arms]       At 7B that will OOM; run them in waves instead." >&2
fi

pids=()
names=()
i=0
for arm in "${ARMS[@]}"; do
  gpu=$(( i % N_GPU ))
  cfg="$ARM_DIR/${arm}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[arms] SKIP $arm — no such config: $cfg" >&2
    i=$((i+1)); continue
  fi
  log="$LOGDIR/${arm}.log"
  echo "[arms] gpu$gpu <- $arm   ($log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m tag.train --config "$cfg" \
      --run_suffix "seed${SEED}" \
      --override "seed=${SEED}" ${OVERRIDES:+$OVERRIDES} \
      > "$log" 2>&1 &
  pids+=($!)
  _pids_all+=($!)
  names+=("$arm")
  i=$((i+1))
done

echo "[arms] ${#pids[@]} run(s) launched; waiting..."
n_fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[arms] OK    ${names[$idx]}"
  else
    echo "[arms] FAILED ${names[$idx]} — see $LOGDIR/${names[$idx]}.log" >&2
    # Surface the actual error rather than making the user open the log.
    tail -n 15 "$LOGDIR/${names[$idx]}.log" | sed 's/^/         | /' >&2
    n_fail=$((n_fail+1))
  fi
done

echo ""
echo "[arms] seed $SEED done — ${n_fail}/${#pids[@]} arm(s) failed"
# Print the real paths, read out of each arm's own output_subdir. The old
# line hardcoded "lowq" and was wrong the moment ARM_DIR became a knob.
for _n in ${names[@]+"${names[@]}"}; do
  _sub="$(sed -n 's/^output_subdir:[[:space:]]*//p' "$ARM_DIR/${_n}.yaml" | head -1)"
  echo "       run dir: $OUTPUT_ROOT/${_sub:-$_n}/runs/*seed${SEED}"
done
unset _n _sub
if [ "$n_fail" -gt 0 ]; then
  echo "[arms] NOTE: a failed arm can leave a process holding GPU memory while" >&2
  echo "[arms]       its CUDA context tears down. Check nvidia-smi before" >&2
  echo "[arms]       relaunching; kill leftovers with:" >&2
  echo "[arms]         pkill -u \$USER -f 'tag\\.train'" >&2
fi
exit $([ "$n_fail" -gt 0 ] && echo 1 || echo 0)
