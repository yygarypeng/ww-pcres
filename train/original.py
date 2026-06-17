import os
import argparse
import shutil
from pathlib import Path
import sys

import numpy as np
import yaml

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "configs/config.yaml"


from model import LightningWBoson
from data import load_data as data
from data.data_module import WBosonDataModule


def resolve_repo_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def clean_training_output(saved_path):
    saved_path = str(resolve_repo_path(saved_path))
    if not saved_path:
        raise ValueError("paths.saved_path must be a non-empty directory path")
    if os.path.abspath(saved_path) == os.path.abspath(os.sep):
        raise ValueError("Refusing to delete filesystem root as paths.saved_path")
    if os.path.exists(saved_path):
        if not os.path.isdir(saved_path):
            raise ValueError(f"paths.saved_path exists but is not a directory: {saved_path}")
        print(f"Found existing checkpoint at {saved_path}, deleting entire folder...")
        shutil.rmtree(saved_path)
    else:
        print("No existing checkpoint found, starting fresh...")


def load_config(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path).expanduser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def flatten_config(config, prefix=""):
    items = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.update(flatten_config(value, full_key))
        else:
            items[full_key] = value
    return items


def prime_csv_metric_header(csv_logger, model):
    metric_keys = {"epoch", "step"}
    loss_names = sorted(model.loss_weights.keys())
    active_loss_names = [
        name for name, weight in model.loss_weights.items()
        if weight != 0.0
    ]
    for prefix in ("", "val_", "test_"):
        metric_keys.add(f"{prefix}loss")
        metric_keys.update(f"{prefix}{name}_loss" for name in loss_names)
    if model.log_loss_gradient_cosines:
        metric_keys.update(
            f"grad_cos/{name}__total"
            for name in active_loss_names
        )
    metric_keys.update(
        f"loss_weight/{name}"
        for name in loss_names
    )

    writer = csv_logger.experiment
    existing_keys = set(getattr(writer, "metrics_keys", []))
    writer.metrics_keys = sorted(existing_keys | metric_keys)


def build_training_callbacks(trainer_cfg):
    monitor_metric = trainer_cfg.get("monitor_metric", "val_loss")
    monitor_mode = trainer_cfg.get("monitor_mode", "min")
    save_top_k = int(trainer_cfg.get("save_top_k", 3))
    save_last = bool(trainer_cfg.get("save_last", True))
    early_stop_patience = trainer_cfg.get("early_stop_patience", 128)
    early_stop_min_delta = float(trainer_cfg.get("early_stop_min_delta", 0.0))

    ckpt = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=save_top_k,
        save_last=save_last,
        filename=f"reg-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
    )
    callbacks = [ckpt]
    if early_stop_patience is not None and int(early_stop_patience) > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=int(early_stop_patience),
                min_delta=early_stop_min_delta,
                mode=monitor_mode,
                verbose=False,
            )
        )
    return ckpt, callbacks


def main(train=True, arg=None, config_path=DEFAULT_CONFIG):
    # ---------- load config ----------
    if arg is not None and hasattr(arg, "config"):
        config_path = arg.config
    _cfg = load_config(config_path)
    _param = _cfg["parameters"]
    SEED = _param.get("seed", 114)
    BATCH_SIZE = _param["batch_size"]
    EPOCHS = _param["epochs"]
    LEARNING_RATE = _param["learning_rate"]
    GRADIENT_CLIP_VAL = _param.get("gradient_clip_val", 1.0)
    LOSS_WEIGHTS = _param["loss_weights"]
    ADAPTIVE_LOSS_WEIGHTS = _param.get("adaptive_loss_weights", False)
    LOG_LOSS_GRADIENT_COSINES = _param.get("log_loss_gradient_cosines", False)
    D_MODEL = _param["d_model"]
    N_HEADS = _param["n_heads"]
    NUM_WORKERS = _param.get("num_workers", 0)
    PERSISTENT_WORKERS = _param.get("persistent_workers", False)
    PIN_MEMORY = _param.get("pin_memory", torch.cuda.is_available())
    PREFETCH_FACTOR = _param.get("prefetch_factor", 2)
    DATA_CFG = _cfg.get("data", {})
    TRAINER_CFG = _cfg.get("trainer", {})

    # some stable settings for dataloader and numpy
    os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS + 2)

    saved_path = str(resolve_repo_path(_cfg["paths"]["saved_path"]))
    data_path = str(resolve_repo_path(_cfg["paths"]["data_path"]))

    if train != True:
        print("Evaluation mode, loading checkpoints...")

    seed_everything(SEED, workers=True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium") # "high" is more accurate but slower

    llvv_train, ww_train, llvv_val, ww_val, llvv_test, ww_test = data.load_presplit_data(
        data_path,
        data_cfg=DATA_CFG,
    )

    X_train = llvv_train.astype(np.float32)
    Y_train = ww_train.astype(np.float32)
    X_val = llvv_val.astype(np.float32)
    Y_val = ww_val.astype(np.float32)
    X_test = llvv_test.astype(np.float32)
    Y_test = ww_test.astype(np.float32)

    dm = WBosonDataModule(
        X_train,
        Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
    ) 
    dm.setup()
    (std_mean_train, std_scale_train), _ = data.compute_standardization_stats(
        X_train,
        Y_train,
    )

    if train == True:
        print("Starting training...")
        input_dim = X_train.shape[1]
        print(f"Input dimension: {input_dim}")
        model = LightningWBoson(
            input_dim=input_dim,
            std_mean_train=std_mean_train, std_scale_train=std_scale_train,
            lr=LEARNING_RATE,
            loss_weights=LOSS_WEIGHTS,
            adaptive_loss_weights=ADAPTIVE_LOSS_WEIGHTS,
            log_loss_gradient_cosines=LOG_LOSS_GRADIENT_COSINES,
            d_model=D_MODEL,
            num_heads=N_HEADS
        )

        ckpt, callbacks = build_training_callbacks(TRAINER_CFG)

        steps_per_epoch = max(1, len(dm.train_dataloader()))
        clean_training_output(saved_path)
        use_wandb = bool(arg is not None and arg.wandb)
        if use_wandb:
            run_name = getattr(arg, "run_name", None) or os.path.basename(os.path.normpath(saved_path))
            wandb_project = getattr(arg, "wandb_project", None) or "PCRES-regressor"
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=run_name,
                save_dir=saved_path,
                log_model=True,
            )
            logged_config = flatten_config(_cfg)
            swept_config_keys = set(getattr(arg, "swept_config_keys", []) or [])
            if swept_config_keys:
                logged_config = {
                    key: value
                    for key, value in logged_config.items()
                    if key not in swept_config_keys
                }
            wandb_logger.experiment.config.update(logged_config, allow_val_change=True)
            if bool(getattr(arg, "watch_model", False)):
                wandb_logger.watch(model, log="all", log_freq=steps_per_epoch, log_graph=False)
        else:
            wandb_logger = None
            print("Wandb logging disabled, only using CSVLogger.")

        csv_logger = CSVLogger(save_dir=saved_path, name="logs", version=0)
        prime_csv_metric_header(csv_logger, model)

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=[csv_logger, wandb_logger] if use_wandb else [csv_logger],
            log_every_n_steps=steps_per_epoch,
            gradient_clip_val=GRADIENT_CLIP_VAL,
            limit_train_batches=TRAINER_CFG.get("limit_train_batches", 1.0),
            limit_val_batches=TRAINER_CFG.get("limit_val_batches", 1.0),
            limit_test_batches=TRAINER_CFG.get("limit_test_batches", 1.0),
        )
        trainer.fit(model, datamodule=dm)
        if dm.test_ds is not None and len(dm.test_ds) > 0:
            print("Running test evaluation with best checkpoint...")
            test_trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                limit_test_batches=TRAINER_CFG.get("limit_test_batches", 1.0),
            )
            test_trainer.test(model=model, datamodule=dm, ckpt_path=ckpt.best_model_path, weights_only=False)
        else:
            print("No test split available, skipping test evaluation.")
        if use_wandb:
            wandb_logger.experiment.finish()
    else:
        print("Loading model from checkpoint for evaluation... return datamodule")
        return dm


if __name__ == "__main__":
    from time import time
    t0 = time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG), help="Path to YAML config file")
    argparser.add_argument("--wandb", "-w", action="store_true", help="Enable wandb logging and training mode")
    argparser.add_argument("--run-name", default=None, help="Optional run name for loggers")
    argparser.add_argument("--wandb-project", default="PCRES-regressor", help="Weights & Biases project name")
    argparser.add_argument("--watch-model", action="store_true", help="Log model gradients/parameters to wandb")
    args = argparser.parse_args()
    main(train=True, arg=args)
    t1 = time()
    print(f"Total time: {(t1 - t0):.2f} seconds.")
