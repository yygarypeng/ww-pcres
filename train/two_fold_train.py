import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import compute_neural_input_stats  # noqa: E402
from data import load_data as data  # noqa: E402
from data.data_module import WBosonDataModule  # noqa: E402
from train import train as base_training  # noqa: E402


def _select_parity(X, Y, parity, split_name, fold):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"{split_name} features and targets must have matching rows, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )

    mask = np.arange(X.shape[0]) % 2 == parity
    if not mask.any():
        raise ValueError(f"Fold {fold} {split_name.lower()} split is empty")
    return X[mask], Y[mask]


def select_fold_splits(splits, fold):
    """Select one cross-fitting fold using post-filtering row parity."""
    if fold not in (0, 1):
        raise ValueError(f"fold must be 0 or 1, got {fold}")
    if len(splits) != 6:
        raise ValueError(f"Expected six train/validation/test arrays, got {len(splits)}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = splits
    training_parity = 1 - fold
    inference_parity = fold
    X_train, Y_train = _select_parity(X_train, Y_train, training_parity, "Training", fold)
    X_val, Y_val = _select_parity(X_val, Y_val, training_parity, "Validation", fold)
    X_test, Y_test = _select_parity(X_test, Y_test, inference_parity, "Test", fold)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def build_fold_datamodule(cfg, data_path, fold):
    params = cfg["parameters"]
    splits = data.load_presplit_data(data_path, data_cfg=cfg.get("data", {}))
    splits = tuple(split.astype(np.float32) for split in splits)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = select_fold_splits(splits, fold)

    datamodule = WBosonDataModule(
        X_train,
        Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        batch_size=params["batch_size"],
        seed=params.get("seed", 114),
        num_workers=params.get("num_workers", 0),
        persistent_workers=params.get("persistent_workers", False),
        pin_memory=params.get("pin_memory", torch.cuda.is_available()),
        prefetch_factor=params.get("prefetch_factor", 2),
    )
    datamodule.setup()
    standardization = compute_neural_input_stats(X_train)
    return datamodule, X_train.shape[1], standardization


def fold_output_path(saved_path, fold):
    return Path(saved_path) / f"fold{fold}"


def configure_runtime(params, gpu):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Using GPU {gpu} (CUDA_VISIBLE_DEVICES={gpu})")

    num_threads = str(params.get("num_workers", 0))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = num_threads

    seed_everything(params.get("seed", 114), workers=True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")


def run_fold(cfg, fold, use_wandb):
    data_path = base_training.resolve_repo_path(cfg["paths"]["data_path"])
    datamodule, input_dim, standardization = build_fold_datamodule(cfg, data_path, fold)
    saved_path = fold_output_path(cfg["paths"]["saved_path"], fold)
    print(
        f"Fold {fold}: training on {'odd' if fold == 0 else 'even'} filtered rows; "
        f"testing on {'even' if fold == 0 else 'odd'} filtered rows."
    )
    base_training.run_training(
        cfg,
        datamodule,
        input_dim,
        standardization,
        saved_path,
        use_wandb,
    )


def parallel_commands(args, script_path=None):
    script_path = Path(__file__).resolve() if script_path is None else Path(script_path)
    commands = []
    for fold in (0, 1):
        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(args.config),
            "--fold",
            str(fold),
        ]
        if args.wandb:
            command.append("--wandb")
        if args.gpu is not None:
            command.extend(("--gpu", str(args.gpu)))
        commands.append(command)
    return commands


def wait_for_folds(processes):
    failures = []
    for fold, process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            failures.append(f"fold {fold} (exit {exit_code})")
    if failures:
        raise RuntimeError("Two-fold training failed: " + ", ".join(failures))


def run_parallel(args):
    print(
        "Warning: both folds will share one GPU. This may be slower than sequential training "
        "or run out of GPU memory."
    )
    processes = []
    try:
        for fold, command in zip((0, 1), parallel_commands(args), strict=True):
            processes.append((fold, subprocess.Popen(command)))
        wait_for_folds(processes)
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=str(base_training.DEFAULT_CONFIG),
        help="Path to YAML config file",
    )
    parser.add_argument("--fold", choices=("0", "1", "both"), default="both")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train both folds concurrently on the selected GPU",
    )
    parser.add_argument("--wandb", "-w", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--gpu",
        type=int,
        choices=(0, 1),
        help="Physical GPU used by each training process",
    )
    args = parser.parse_args(argv)
    if args.parallel and args.fold != "both":
        parser.error("--parallel requires --fold both")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        "Fold assignment uses post-filtering row indices, not HWWFrames eventNumber. "
        "Regenerating or filtering the HDF5 can change fold membership."
    )
    if args.parallel:
        run_parallel(args)
        return

    cfg = base_training.load_config(args.config)
    folds = (0, 1) if args.fold == "both" else (int(args.fold),)
    for fold in folds:
        configure_runtime(cfg["parameters"], args.gpu)
        run_fold(cfg, fold, args.wandb)


if __name__ == "__main__":
    main()
