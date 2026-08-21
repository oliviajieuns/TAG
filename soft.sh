#!/usr/bin/env bash
set -Eeuo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
export B=soft
exec bash "$REPO/ablate.sh" "$@"
