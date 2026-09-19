#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
log="$REPO_ROOT/record_k_fold.log"

# Keep the affinity mask off unstable CPUs 10-11 and E-cores 16-31.
# A user-supplied --config later in "$@" overrides the default below.
taskset -c 0-9,12-15 python "${SCRIPT_DIR}/k_fold_train.py" \
    --config "$REPO_ROOT/configs/kfold_config.yaml" --wandb "$@" \
    &> "$log" &

echo "k-fold training started: pid $!"
echo "log: $log"
