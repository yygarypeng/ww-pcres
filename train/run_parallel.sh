#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# Pin each run to a disjoint, known-stable CPU range.
# The 11th cpu is not stable in this BIOS version! (July 29, 2026)
V61_CPUS="0-9"
V4_CPUS="12-21"

echo "$(date +%H:%M:%S) LAUNCH v61 (cpus=$V61_CPUS)"
taskset -c "$V61_CPUS" python "${SCRIPT_DIR}/train.py" --config "${REPO_ROOT}/configs/config_v61.yaml" \
    &> "$REPO_ROOT/record_v61.log" &
V61_PID=$!

echo "$(date +%H:%M:%S) LAUNCH v4  (cpus=$V4_CPUS)"
taskset -c "$V4_CPUS" python "${SCRIPT_DIR}/train.py" --config "${REPO_ROOT}/configs/config_v4.yaml" \
    &> "$REPO_ROOT/record_v4.log" &
V4_PID=$!

echo "$(date +%H:%M:%S) launched v61 (pid $V61_PID) and v4 (pid $V4_PID)"