#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

command=(python "$SCRIPT_DIR/train.py" "$@")
if [[ -n "${CPU_CORES:-}" ]]; then
    command=(taskset -c "$CPU_CORES" "${command[@]}")
fi

"${command[@]}" &> "$REPO_ROOT/record.log" &
