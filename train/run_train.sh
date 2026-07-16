#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

# taskset -c 6-15 python "$SCRIPT_DIR/train.py" --wandb &> "$REPO_ROOT/record.log" &
python "$SCRIPT_DIR/train.py" -w &> "$REPO_ROOT/record.log" &
