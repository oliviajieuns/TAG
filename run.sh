#!/usr/bin/env bash

# Short entry point for the TAG bs64 + LR-floor tuning arm.
#
#   S=1  bash run.sh          # launch
#   S=1  bash run.sh status   # inspect
#        bash run.sh test     # tests only

set -u

REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
LAUNCHER="$REPO/scripts/gpu_cloud/run_tag10_schedfloor_bs64.sh"

case "${1:-}" in
  test)
    cd "$REPO"
    exec "$PY" -m pytest -q \
      tests/test_training_schedule.py \
      tests/test_gate_params_forwarded.py
    ;;
  status)
    set -- --status
    ;;
  "")
    cd "$REPO"
    "$PY" -m pytest -q \
      tests/test_training_schedule.py \
      tests/test_gate_params_forwarded.py || exit $?
    set --
    ;;
  *)
    echo "usage: S={1|7|42} bash run.sh [status]" >&2
    echo "       bash run.sh test" >&2
    exit 2
    ;;
esac

export TAG10_TUNE_SEED=${S:-42}
exec bash "$LAUNCHER" "$@"
