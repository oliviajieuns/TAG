#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
exec "$PY" "$REPO/scripts/gpu_cloud/summarize_gate_sweep.py" \
  "${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}"
