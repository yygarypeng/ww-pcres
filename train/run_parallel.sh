#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# Pin each run to a disjoint, known-stable CPU range.
# The 11th cpu is not stable in this BIOS version! (July 29, 2026)
ORG_CPUS="0-9"
NODMET_CPUS="12-21"

echo "$(date +%H:%M:%S) LAUNCH ORG (cpus=$ORG_CPUS)"
taskset -c "$ORG_CPUS" python "${SCRIPT_DIR}/train.py" --config "${REPO_ROOT}/configs/config.yaml" \
    &> "$REPO_ROOT/record.log" &
ORG_PID=$!

echo "$(date +%H:%M:%S) LAUNCH NODMET  (cpus=$NODMET_CPUS)"
taskset -c "$NODMET_CPUS" python "${SCRIPT_DIR}/train_no_dmet.py" --config "${REPO_ROOT}/configs/config_no_dmet.yaml" \
    &> "$REPO_ROOT/record_no_dmet.log" &
NODMET_PID=$!

echo "$(date +%H:%M:%S) launched ORG (pid $ORG_PID) and NODMET (pid $NODMET_PID)"
