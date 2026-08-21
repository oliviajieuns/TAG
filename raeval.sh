#!/usr/bin/env bash

# Full eight-benchmark evaluation for the fresh legacy R x A repeat.

set -Eeuo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
export A=ra
exec bash "$REPO/all.sh" "$@"
