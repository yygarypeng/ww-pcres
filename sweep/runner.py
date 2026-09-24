"""Train and score one sweep trial on a fold's validation rows; the test group is never read."""

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from data import compute_neural_input_stats
from data.data_module import WBosonDataModule
from data.load_data import load_presplit_data
from sweep import metrics, resources
from sweep.lightning import FidelityMetrics, SweepLightningWBoson
from train import k_fold_train
from train import train as base_training

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPOCHS = 100
DEFAULT_WEIGHTS = {"bias": 1.0, "angular_1d": 1.0, "interangular_1d": 1.0}
OUTPUT_ROOT = REPO_ROOT / "sweep" / "outputs"

_FOLD_CACHE = {}


def fold_inputs(cfg, fold):
    """Fold arrays and input statistics, read from the HDF5 once and reused by every trial."""
    data_path = base_training.resolve_repo_path(cfg["paths"]["data_path"])
    folds = k_fold_train.resolve_folds(cfg)
    key = (str(data_path), int(fold), int(folds), json.dumps(cfg.get("data", {}), sort_keys=True))
    if key not in _FOLD_CACHE:
        _FOLD_CACHE.clear()  # one fold's arrays at a time is plenty for a sequential sweep
        raw = load_presplit_data(
            data_path,
            data_cfg=cfg.get("data", {}),
            with_event_numbers=True,
            splits=("train", "val"),
        )
        x_train, y_train, events_train, x_val, y_val, events_val = raw
        x_pool = np.concatenate((x_train, x_val))
        y_pool = np.concatenate((y_train, y_val))
        events_pool = np.concatenate((events_train, events_val))
        held_out = k_fold_train.fold_of_event(events_pool, folds) == fold
        splits = tuple(
            np.ascontiguousarray(split, dtype=np.float32)
            for split in (
                x_pool[~held_out],
                y_pool[~held_out],
                x_pool[held_out],
                y_pool[held_out],
            )
        )
        _FOLD_CACHE[key] = (splits, compute_neural_input_stats(splits[0]))
    return _FOLD_CACHE[key]


def build_datamodule(cfg, splits, num_workers=None):
    """Datamodule over the fold's train and validation rows."""
    params = cfg["parameters"]
    x_fit, y_fit, x_check, y_check = splits[:4]
    workers = params.get("num_workers", 0) if num_workers is None else num_workers
    workers = resources.safe_worker_count(cfg, splits[:4], workers)
    return WBosonDataModule(
        x_fit,
        y_fit,
        X_val=x_check,
        Y_val=y_check,
        batch_size=params["batch_size"],
        seed=params.get("seed", 114),
        num_workers=workers,
        # Pinned persistent workers leak pipe descriptors across trials in PyTorch 2.6.
        persistent_workers=False,
        pin_memory=params.get("pin_memory", torch.cuda.is_available()),
        prefetch_factor=params.get("prefetch_factor", 2),
        # Spawn avoids inheriting CUDA state from previous trials.
        multiprocessing_context="spawn",
    )


def build_model(cfg, input_dim, standardization):
    params = cfg["parameters"]
    std_mean_train, std_scale_train = standardization
    return SweepLightningWBoson(
        input_dim=input_dim,
        std_mean_train=std_mean_train,
        std_scale_train=std_scale_train,
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-4),
        loss_weights=params["loss_weights"],
        mmd_config=cfg.get("mmd", {}),
        adaptive_loss_weights=params.get("adaptive_loss_weights", False),
        log_loss_gradient_cosines=params.get("log_loss_gradient_cosines", False),
        d_model=params["d_model"],
        num_heads=params["n_heads"],
        attention_blocks=params.get("attention_blocks", 4),
        attention_dropout=params.get("attention_dropout", 0.1),
        decoder_dropout=params.get("decoder_dropout", 0.1),
        lr_plateau_factor=params.get("lr_plateau_factor", 1.0),
        lr_plateau_patience=params.get("lr_plateau_patience", 8),
        fourvec_loss=params.get("fourvec_loss", "l1"),
        huber_delta=params.get("huber_delta", 10.0),
    )


def prepare_trial_dir(trial_dir, output_root=None):
    """Empty one run directory, refusing anything outside ``output_root``."""
    root = Path(output_root or OUTPUT_ROOT).resolve()
    trial_dir = Path(trial_dir).resolve()
    if root not in trial_dir.parents:
        raise ValueError(f"refusing to clear {trial_dir}, which is outside {root}")
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True)
    return trial_dir


def predict_split(model, dataloader, device):
    """Inputs, targets, and predictions for a whole split, on the CPU."""
    model = model.to(device).eval()
    features, targets, predictions = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            predictions.append(model(x).detach().cpu())
            features.append(x.detach().cpu())
            targets.append(y.detach().cpu())
    return torch.cat(features), torch.cat(targets), torch.cat(predictions)


def _callbacks(cfg, fidelity, extra, full_epochs=False):
    params = cfg["parameters"]
    callbacks = [
        fidelity,
        *extra,
        LearningRateMonitor(logging_interval="epoch"),
        # The raw composite, since the selection score jumps as constraints flicker.
        EarlyStopping(
            monitor=FidelityMetrics.COMPOSITE_METRIC,
            patience=params.get("early_stopping_patience", 32),
            min_delta=params.get("early_stopping_min_delta", 0.0),
            mode="min",
            verbose=False,
        ),
        # The penalised score, so the kept epoch is feasible whenever one was.
        ModelCheckpoint(
            monitor=FidelityMetrics.SELECTION_METRIC,
            mode="min",
            save_top_k=1,
            save_last=False,
            filename="best-{epoch:03d}",
        ),
    ]
    return [cb for cb in callbacks if not full_epochs or not isinstance(cb, EarlyStopping)]


def run_trial(
    cfg,
    trial_dir,
    *,
    fold=0,
    epochs=DEFAULT_EPOCHS,
    weights=None,
    every_n_epochs=1,
    callback_factory=None,
    thresholds=None,
    num_workers=None,
    full_epochs=False,
    output_root=None,
):
    """Train one configuration on ``fold`` and return its fixed-ruler report.

    ``callback_factory`` maps the live :class:`FidelityMetrics` to extra callbacks,
    such as the Optuna pruner.
    """
    started = time.monotonic()
    weights = dict(weights or DEFAULT_WEIGHTS)
    trial_dir = prepare_trial_dir(trial_dir, output_root)
    params = cfg["parameters"]

    base_training.configure_runtime(params)
    seed_everything(params.get("seed", 114), workers=True)

    splits, standardization = fold_inputs(cfg, fold)
    datamodule = build_datamodule(cfg, splits, num_workers=num_workers)
    model = build_model(cfg, splits[0].shape[1], standardization)

    fidelity = FidelityMetrics(weights, thresholds=thresholds, every_n_epochs=every_n_epochs)
    extra = list(callback_factory(fidelity)) if callback_factory is not None else []
    callbacks = _callbacks(cfg, fidelity, extra, full_epochs=full_epochs)
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=trial_dir, name="logs", version=0),
        log_every_n_steps=max(1, len(datamodule.train_dataloader())),
        gradient_clip_val=params.get("gradient_clip_val", 1.0),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with open(trial_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    try:
        trainer.fit(model, datamodule=datamodule)
        report = _score_best_checkpoint(callbacks[-1], datamodule, weights, trainer.current_epoch)
    finally:
        # Pruning during validation leaves the training loader suspended.
        trainer.fit_loop.teardown()
        del model, trainer, datamodule
        gc.collect()
        torch.cuda.empty_cache()

    # The workflow budgets confirmation from the measured baseline wall clock.
    report["wall_seconds"] = float(time.monotonic() - started)
    with open(trial_dir / "metrics.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def _score_best_checkpoint(checkpoint, datamodule, weights, last_epoch):
    """Reload the best checkpoint and score it on the fold's validation rows."""
    if not checkpoint.best_model_path:
        raise RuntimeError("training finished without writing a checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The hyper-parameters hold numpy arrays, which weights_only=True rejects.
    best = SweepLightningWBoson.load_from_checkpoint(
        checkpoint.best_model_path,
        map_location=device,
        weights_only=False,
    )
    x, y_true, y_pred = predict_split(best, datamodule.val_dataloader(), device)
    report = metrics.evaluate(x, y_pred, y_true, with_floor=True)
    report["composite"] = metrics.composite(report["objectives"], weights)
    report["objective_weights"] = weights
    report["checkpoint"] = str(checkpoint.best_model_path)
    report["checkpoint_score"] = float(checkpoint.best_model_score)
    report["epochs_trained"] = int(last_epoch)
    del best
    return report
