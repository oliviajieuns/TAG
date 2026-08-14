#!/usr/bin/env bash
# Compute the shared TAG gate cache across every visible GPU, then merge.
#
#   source scripts/gpu_cloud/env.sh
#   bash scripts/precompute_gate.sh [config] [out]
#
# Defaults to the 7B lowq arm and $POOLS/<pool>/tag_gate_<model_key>.pt.
#
# G depends only on (pool, base checkpoint, gate config), so this runs ONCE
# and every arm and seed reuses it — export TAG_GATE_CACHE=<out> before
# launching training. On the paper's 8-arm x 3-seed grid that turns 24
# redundant gate computations into one.
#
# Shards are independent processes, not torchrun: if one dies, re-run just
# that index (the script prints the exact command) instead of redoing the lot.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CFG="${1:-configs/experiments/lowq/tag_7b.yaml}"
OUT="${2:-}"

if [ -z "${TAG_WORKSPACE:-}" ]; then
  echo "[error] source scripts/gpu_cloud/env.sh first" >&2
  exit 2
fi

if [ -z "$OUT" ]; then
  KEY="$(python - "$CFG" <<'PY'
import sys
from tag.core.utils import load_config
c = load_config(sys.argv[1])
print(str(c.get("model_key", "model")))
PY
)"
  OUT="${POOLS}/composite20/tag_gate_${KEY}.pt"
fi

N="${NUM_SHARDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
[ "${N:-0}" -lt 1 ] && N=1

if [ -f "$OUT" ]; then
  echo "[gate] $OUT already exists — delete it to recompute."
  echo "[gate] export TAG_GATE_CACHE=$OUT"
  exit 0
fi

# Resolve the calibration first: a missing gate reference used to surface
# only at the merge, after every shard had already run.
if ! python - "$CFG" <<'PRECHK'
import sys
sys.path.insert(0, ".")
from tag.core.utils import load_config
from tag.pipelines.selection import _resolve_gate_calibration
cfg = load_config(sys.argv[1])
_resolve_gate_calibration((cfg.get("selection") or {}).get("tag") or {})
PRECHK
then
  echo "[gate] calibration reference unusable — not launching shards." >&2
  exit 1
fi

echo "[gate] config : $CFG"
echo "[gate] out    : $OUT"
echo "[gate] shards : $N (one per GPU)"

LOGDIR="$TAG_WORKSPACE/logs/gate"
mkdir -p "$LOGDIR"

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

pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/precompute_gate.py \
    --config "$CFG" --out "$OUT" --shard "$i" --num-shards "$N" \
    > "$LOGDIR/shard$i.log" 2>&1 &
  pids+=($!)
  _pids_all+=($!)
  echo "[gate] gpu$i -> shard $i  ($LOGDIR/shard$i.log)"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[gate] shard $i FAILED:" >&2
    tail -n 20 "$LOGDIR/shard$i.log" | sed 's/^/         | /' >&2
    fail=1
  fi
done
[ "$fail" -ne 0 ] && { echo "[gate] aborting before merge" >&2; exit 1; }

python scripts/precompute_gate.py --config "$CFG" --out "$OUT" \
  --num-shards "$N" --merge
