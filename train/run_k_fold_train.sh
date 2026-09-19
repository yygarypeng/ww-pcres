#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

config_args=(--config "$REPO_ROOT/configs/kfold_config.yaml")
for arg in "$@"; do
    case "$arg" in
        -c | --config | --config=*)
            config_args=()
            break
            ;;
    esac
done

log="$REPO_ROOT/record_k_fold.log"

taskset -c 0-9,12-15 python "${SCRIPT_DIR}/k_fold_train.py" \
    "${config_args[@]}" --fold all --wandb "$@" \
    &> "$log" &

echo "k-fold training started: pid $!"
echo "log: $log"
