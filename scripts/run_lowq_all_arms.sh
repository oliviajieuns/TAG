#!/usr/bin/env bash
# Run the lowq 0.5B arms in parallel, one per GPU.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/run_lowq_all_arms.sh [seed] [arm ...]
#
# At 0.5B a single H100 is far faster than DDP across four of them (the
# gradient sync dominates), so the right use of a 4-GPU box is four
# CONCURRENT arms rather than one distributed arm.
#
# Every arm inherits its training hyperparameters from
# _shared_light_05b.yaml, and TADS_EPISODE_BS is read from THIS shell, so
# all arms launched here share it by construction — which is the property
# plan §5.2 demands. Launching a stray arm later with a different value
# would break comparability; check cfg.yaml in each run dir if unsure.
set -uo pipefail

SEED="${1:-42}"
shift || true

# SCALE=7b runs the 7B grid; anything else runs the 0.5B one.
SCALE="${SCALE:-05b}"
if [ "$SCALE" = "7b" ]; then
  DEFAULT_ARMS=(tag_7b tag_static_7b tads_legacy_7b random_7b)
else
  DEFAULT_ARMS=(
    light_tag_05b
    light_tag_static_05b
    light_tads_legacy_05b
    light_random_05b
  )
fi
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${DEFAULT_ARMS[@]}")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${TAG_WORKSPACE:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi

N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "$N_GPU" -eq 0 ] && { echo "[error] no GPUs visible" >&2; exit 2; }

LOGDIR="$TAG_WORKSPACE/logs/arms_${SCALE}_seed${SEED}"
mkdir -p "$LOGDIR"

# A shared gate cache turns N redundant gate computations into zero: every
# TAG arm here reads the same file. Without it each arm recomputes the gate
# on its own GPU, which at 7B is 1+K full pool forwards per arm.
if [ -n "${TADS_GATE_CACHE:-}" ]; then
  if [ -f "$TADS_GATE_CACHE" ]; then
    echo "[arms] shared gate cache: $TADS_GATE_CACHE"
  else
    echo "[arms] WARNING: TADS_GATE_CACHE=$TADS_GATE_CACHE does not exist." >&2
    echo "[arms]          Each TAG arm will recompute the gate separately." >&2
    echo "[arms]          Run: bash scripts/precompute_gate.sh <config>" >&2
  fi
fi

echo "[arms] scale=$SCALE seed=$SEED gpus=$N_GPU"
if [ "$SCALE" = "7b" ]; then
  echo "[arms] episode_bs=${TADS_EPISODE_BS_7B:-8} grad_accum=${TADS_GRAD_ACCUM_7B:-16} (1 arm/GPU)"
else
  echo "[arms] episode_bs=${TADS_EPISODE_BS:-2}"
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
  cfg="configs/experiments/lowq/${arm}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[arms] SKIP $arm — no such config: $cfg" >&2
    i=$((i+1)); continue
  fi
  log="$LOGDIR/${arm}.log"
  echo "[arms] gpu$gpu <- $arm   ($log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m tads.train --config "$cfg" \
      --run_suffix "seed${SEED}" \
      --override "seed=${SEED}" \
      > "$log" 2>&1 &
  pids+=($!)
  names+=("$arm")
  i=$((i+1))
done

echo "[arms] ${#pids[@]} run(s) launched; waiting..."
fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[arms] OK    ${names[$idx]}"
  else
    echo "[arms] FAILED ${names[$idx]} — see $LOGDIR/${names[$idx]}.log" >&2
    # Surface the actual error rather than making the user open the log.
    tail -n 15 "$LOGDIR/${names[$idx]}.log" | sed 's/^/         | /' >&2
    fail=1
  fi
done

echo ""
echo "[arms] seed $SEED done (fail=$fail)"
echo "       run dirs: $OUTPUT_ROOT/lowq/*/runs/*seed${SEED}"
exit $fail
