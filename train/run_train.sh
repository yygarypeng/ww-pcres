#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# Keep the affinity mask off unstable CPUs 10-11 and E-cores 16-31.
taskset -c 0-9,12-15 python "${SCRIPT_DIR}/train.py" "$@" \
    &> "$REPO_ROOT/record.log" &
