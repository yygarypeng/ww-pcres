#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# TODO: UPDATE BIOS to fix the 11th CPU instability issue
# 11th cpu is not stable in this BIOS version! (July 29, 2026)
taskset -c 0-9 python "${SCRIPT_DIR}/train.py" "$@" \
    &> "$REPO_ROOT/record.log" &
