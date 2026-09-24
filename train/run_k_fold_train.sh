#!/usr/bin/env bash
#
# Background k-fold training launcher. Execute it, do not source it.
#
#   ./train/run_k_fold_train.sh                  # every fold of configs/kfold_config.yaml
#   ./train/run_k_fold_train.sh --fold 3         # one fold
#   ./train/run_k_fold_train.sh --overwrite      # also retrain existing folds
#
# Other arguments go to train/k_fold_train.py; a --config overrides the default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
log="$REPO_ROOT/record_k_fold.log"
pid_file="$REPO_ROOT/record_k_fold.pid"

overwrite=0
train_args=(--config "$REPO_ROOT/configs/kfold_config.yaml" --wandb)
for arg in "$@"; do
    if [ "$arg" = "--overwrite" ]; then
        overwrite=1
    else
        train_args+=("$arg")
    fi
done

if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "k-fold training already running (pid $(cat "$pid_file"))" >&2
    exit 1
fi

# Fail in the foreground; existing folds are refused because training deletes them.
OVERWRITE="$overwrite" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    taskset -c 0-9,12-15 python - "${train_args[@]}" <<'EOF' || exit 2
import os
import sys

import torch

from train import k_fold_train
from train import train as base_training

sys.argv[0] = "run_k_fold_train.sh"
args = k_fold_train.parse_args(sys.argv[1:])
cfg = base_training.load_config(args.config)
folds = k_fold_train.resolve_folds(cfg)
if args.fold != "all" and args.fold not in range(folds):
    sys.exit(f"--fold must be in 0..{folds - 1}, got {args.fold}")

data_path = base_training.resolve_repo_path(cfg["paths"]["data_path"])
if not data_path.is_file():
    sys.exit(f"data file not found: {data_path}")

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if not torch.cuda.is_available():
    sys.exit("no CUDA device is visible, so training would fall back to the CPU")

saved_path = base_training.resolve_repo_path(cfg["paths"]["saved_path"])
selected = range(folds) if args.fold == "all" else [args.fold]
targets = [saved_path / f"fold{fold}" for fold in selected]
existing = [path for path in targets if path.exists()]
if existing and os.environ["OVERWRITE"] != "1":
    listing = "\n".join(f"  {path}" for path in existing)
    sys.exit(
        f"training would delete these existing fold directories:\n{listing}\n"
        "point paths.saved_path at a new directory, or pass --overwrite to retrain them"
    )

print(f"config: {os.path.abspath(args.config)}")
print(f"folds:  {', '.join(map(str, selected))} of {folds}, into {saved_path}")
EOF

# Append, so a crashed run's traceback survives the next launch.
printf '\n===== k-fold launch %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"

# Keep the affinity mask off unstable CPUs 10-11 and E-cores 16-31.
# setsid and ignored INT/HUP let the run outlive its terminal; `kill <pid>` stops it.
( trap '' INT HUP; exec setsid taskset -c 0-9,12-15 python -u "${SCRIPT_DIR}/k_fold_train.py" \
    "${train_args[@]}" ) < /dev/null &>> "$log" &

pid="$!"
echo "$pid" > "$pid_file"

echo "k-fold training started: pid $pid"
echo "log:  $log"
echo "stop: kill $pid"
