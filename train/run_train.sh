#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
taskset -c "${CPU_CORES:-0-15}" \
  python -m train.original --config "${CONFIG:-$REPO_ROOT/configs/config.yaml}" -w \
  > "${LOG_FILE:-$REPO_ROOT/record}" 2>&1 &
