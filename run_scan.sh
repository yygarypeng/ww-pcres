#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-config.yaml}"
SCAN_ROOT="${SCAN_ROOT:-scans/latest}"

MAX_PARALLEL="${MAX_PARALLEL:-10}"
MAX_RETRIES="${MAX_RETRIES:-2}"
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-scan-pcres}"
BACKGROUND="${BACKGROUND:-1}"

if [[ "$CONFIG" = /* ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="$REPO_ROOT/$CONFIG"
fi

if [[ "$SCAN_ROOT" = /* ]]; then
  SCAN_ROOT_PATH="$(realpath -m "$SCAN_ROOT")"
else
  SCAN_ROOT_PATH="$(realpath -m "$REPO_ROOT/$SCAN_ROOT")"
fi

LOG_FILE="${LOG_FILE:-$SCAN_ROOT_PATH/run_scan.log}"
PID_FILE="${PID_FILE:-$SCAN_ROOT_PATH/run_scan.pid}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

case "$SCAN_ROOT_PATH" in
  ""|"/"|"$REPO_ROOT"|"$REPO_ROOT/"|"$REPO_ROOT/..")
    printf 'Refusing unsafe SCAN_ROOT: %s\n' "$SCAN_ROOT_PATH" >&2
    exit 1
    ;;
esac

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  old_cmd="$(ps -p "$old_pid" -o args= 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && [[ "$old_cmd" == *"scan.py"* ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf 'Stopping previous scan with PID %s\n' "$old_pid"
    kill "$old_pid"
    for _ in {1..60}; do
      if ! kill -0 "$old_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      printf 'Previous scan did not stop; sending SIGKILL to PID %s\n' "$old_pid"
      kill -9 "$old_pid"
    fi
  fi
fi

rm -rf "$SCAN_ROOT_PATH"
mkdir -p "$SCAN_ROOT_PATH"

cmd=(
  "$PYTHON" "$REPO_ROOT/scan.py"
  --config "$CONFIG_PATH"
  --scan-root "$SCAN_ROOT_PATH"
  --no-timestamp
  --max-parallel "$MAX_PARALLEL"
  --max-retries "$MAX_RETRIES"
  --gpus "$GPU_IDS"
  --wandb-project "$WANDB_PROJECT"
)

if [[ -n "$EPOCHS" ]]; then
  cmd+=(--override "parameters.epochs=$EPOCHS")
fi

if [[ "$WANDB_MODE" == "disabled" ]]; then
  cmd+=(--no-wandb)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
  BACKGROUND=0
fi

printf 'Scan command:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'Scan root: %s\n' "$SCAN_ROOT_PATH"
printf 'W&B project: %s\n' "$WANDB_PROJECT"

if [[ "$BACKGROUND" == "0" ]]; then
  "${cmd[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  printf 'Started scan in background with PID %s\n' "$pid"
  printf 'Log: %s\n' "$LOG_FILE"
  printf 'PID file: %s\n' "$PID_FILE"
fi
