import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_data as data  # noqa: E402
from train import train as base_training  # noqa: E402

DEFAULT_FOLDS = 2


def _fold_index(value):
    """argparse type for --fold: a fold index, or the literal 'all'."""
    if value == "all":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--fold must be an integer or 'all', got {value}")


def _select_rows(X, Y, mask, split_name, fold):
    if not mask.any():
        raise ValueError(f"Fold {fold} {split_name.lower()} split is empty")
    return X[mask], Y[mask]


def resolve_folds(cfg):
    """Number of cross-fitting folds, from parameters.folds, then the default."""
    value = cfg.get("parameters", {}).get("folds", DEFAULT_FOLDS)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"folds must be an integer, got {value!r}")
    folds = int(value)
    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    return folds


def _validate_fold(splits, fold, folds):
    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    if fold not in range(folds):
        raise ValueError(f"fold must be in 0..{folds - 1}, got {fold}")
    if len(splits) != 6:
        raise ValueError(f"Expected six train/validation/test arrays, got {len(splits)}")


def select_fold_splits(splits, fold, folds=DEFAULT_FOLDS):
    """Rotate validation through pooled train+validation rows; keep test unchanged."""
    _validate_fold(splits, fold, folds)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = splits
    X_pool = np.concatenate([X_train, X_val])
    Y_pool = np.concatenate([Y_train, Y_val])
    if X_pool.shape[0] != Y_pool.shape[0]:
        raise ValueError(
            f"Pooled features and targets must have matching rows, "
            f"got {X_pool.shape[0]} and {Y_pool.shape[0]}"
        )
    held_out = np.arange(X_pool.shape[0]) % folds == fold

    X_fit, Y_fit = _select_rows(X_pool, Y_pool, ~held_out, "Training", fold)
    X_check, Y_check = _select_rows(X_pool, Y_pool, held_out, "Validation", fold)
    return X_fit, Y_fit, X_check, Y_check, X_test, Y_test


def build_fold_datamodule(cfg, data_path, fold, folds=DEFAULT_FOLDS):
    """Load the pre-split arrays and keep only the rows belonging to one fold."""
    splits = data.load_presplit_data(data_path, data_cfg=cfg.get("data", {}))
    selected = select_fold_splits(splits, fold, folds)
    return base_training.datamodule_from_splits(cfg, selected)


def run_fold(cfg, fold, use_wandb, folds=DEFAULT_FOLDS):
    data_path = base_training.resolve_repo_path(cfg["paths"]["data_path"])
    datamodule, input_dim, standardization = build_fold_datamodule(cfg, data_path, fold, folds)
    saved_path = Path(cfg["paths"]["saved_path"]) / f"fold{fold}"
    print(
        f"Fold {fold} of {folds}: training on {100.0 * (folds - 1) / folds:.1f}% of the "
        f"pooled train+validation rows, validating on the rest, testing on all test rows."
    )
    base_training.run_training(
        cfg,
        datamodule,
        input_dim,
        standardization,
        saved_path,
        use_wandb,
    )


def parse_args(argv=None):
    parser = base_training.add_common_arguments(argparse.ArgumentParser())
    parser.add_argument(
        "--fold",
        default="all",
        type=_fold_index,
        help="Fold index, or 'all' to train every fold in turn (default)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(
        "Fold assignment uses post-filtering row indices, not HWWFrames eventNumber. "
        "Regenerating or filtering the HDF5 can change fold membership."
    )
    cfg = base_training.load_config(args.config)
    folds = resolve_folds(cfg)

    # One fold saturates the GPU, so folds run in turn rather than concurrently.
    selected = range(folds) if args.fold == "all" else (args.fold,)
    for fold in selected:
        base_training.configure_runtime(cfg["parameters"], args.gpu)
        run_fold(cfg, fold, args.wandb, folds)


if __name__ == "__main__":
    main()
