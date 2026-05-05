#!/usr/bin/env bash
set -euo pipefail

taskset -c "${CPU_CORES:-0-15}" python train.py --config "${CONFIG:-config.yaml}" -w &> "${LOG_FILE:-record}" &
