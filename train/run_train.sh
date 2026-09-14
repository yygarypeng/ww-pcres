#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# TODO: UPDATE BIOS to fix the 11th CPU instability issue
# 10th and 11th cpu are not stable in this BIOS version! (July 29, 2026)
# CPUs 0-15 are P-cores (5.7-6.0 GHz), 16-31 are E-cores (4.4 GHz) on this
# i9-14900KF. The step is bound by the main thread's kernel dispatch rate, and
# that thread runs ~2x slower on an E-core, so keep the job off 16-31 entirely.
taskset -c 0-9,12-15 python "${SCRIPT_DIR}/train.py" "$@" \
    &> "$REPO_ROOT/record.log" &
