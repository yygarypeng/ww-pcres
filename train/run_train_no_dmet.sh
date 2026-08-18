#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

taskset -c 12-21 python "${SCRIPT_DIR}/train_no_dmet.py" --config "${REPO_ROOT}/configs/config_no_dmet.yaml" \
    &> "$REPO_ROOT/record_no_dmet.log" &
