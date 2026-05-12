#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SWEEP_ID="${SWEEP_ID:-nthu-expt-hep/pcres-sweep/mpm151hs}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/sweep.log}"
PID_FILE="${PID_FILE:-$REPO_ROOT/sweep.pid}"

cd "$REPO_ROOT"
nohup wandb agent "$SWEEP_ID" > "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

printf 'Started W&B sweep agent PID %s\n' "$pid"
printf 'Sweep: %s\n' "$SWEEP_ID"
printf 'Log: %s\n' "$LOG_FILE"
printf 'PID file: %s\n' "$PID_FILE"
