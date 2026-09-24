#!/usr/bin/env bash
#
# Constrained sweep launcher. Execute it, do not source it.
# Validates in the foreground, then runs the workflow in the background;
# re-running resumes finished work. SWEEP_OUTPUT_DIR moves the log and pid file,
# so set it together with --output-dir.
#
# Foreground modes that never train:
#   ./sweep/run_sweep.sh --validate
#   ./sweep/run_sweep.sh --dry-run
#   ./sweep/run_sweep.sh --report

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
LOG="$OUTPUT_DIR/sweep.log"
PID_FILE="$OUTPUT_DIR/sweep.pid"
DB="$OUTPUT_DIR/study.db"
SELECTED="$OUTPUT_DIR/selected_config.yaml"
AFFINITY="0-9,12-31"

mkdir -p "$OUTPUT_DIR"

is_foreground=0
for arg in "$@"; do
    case "$arg" in
        --validate|--dry-run|--report)
            is_foreground=1
            break
            ;;
    esac
done

cd "$REPO_ROOT" || exit 1

if [ "$is_foreground" -eq 1 ]; then
    taskset -c "$AFFINITY" python -u -m sweep.optimize "$@"
    exit "$?"
fi

if [ -f "$PID_FILE" ]; then
    existing="$(cat "$PID_FILE")"
    if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
        echo "sweep already running (pid $existing); refusing a duplicate driver" >&2
        echo "pid file: $PID_FILE" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

taskset -c "$AFFINITY" python -u -m sweep.optimize --validate "$@" || exit 2

# Every CPU except the defective 10-11. setsid and ignored INT/HUP let the driver
# outlive its terminal, and `kill <pid>` stops it; setsid does not fork here, so $!
# is the driver. The log appends, so a crash traceback survives the resume.
printf '\n===== sweep launch %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
( trap '' INT HUP; exec setsid taskset -c "$AFFINITY" python -u -m sweep.optimize "$@" ) \
    < /dev/null &>> "$LOG" &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "started: pid $pid"
echo "log:     $LOG  (tail -f it)"
echo "db:      $DB"
echo "selected:$SELECTED (written when confirmation finishes)"
echo "resume:  ./sweep/run_sweep.sh $*"
