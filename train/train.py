import argparse
import os
import shutil
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "configs/config.yaml"
DEFAULT_EXCLUDED_CPUS = (10, 11)


from data import compute_neural_input_stats
from data import load_data as data
from data.data_module import WBosonDataModule
from model import LightningWBoson


def resolve_repo_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def clean_training_output(saved_path):
    if saved_path is None or not str(saved_path).strip():
        raise ValueError("paths.saved_path must be a non-empty directory path")
    saved_path = resolve_repo_path(saved_path)
    if saved_path.is_symlink():
        raise ValueError(f"Refusing to delete symbolic link as paths.saved_path: {saved_path}")
    resolved_path = saved_path.resolve()
    if resolved_path == Path(resolved_path.anchor):
        raise ValueError("Refusing to delete filesystem root as paths.saved_path")
    if saved_path.exists():
        if not saved_path.is_dir():
            raise ValueError(f"paths.saved_path exists but is not a directory: {saved_path}")
        print(f"Found existing checkpoint at {saved_path}, deleting entire folder...")
        shutil.rmtree(saved_path)
    else:
        print("No existing checkpoint found, starting fresh...")
    return resolved_path


def load_config(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
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
    for prefix in ("", "val_", "test_"):
        metric_keys.add(f"{prefix}loss")
        metric_keys.update(f"{prefix}{name}_loss" for name in loss_names)
    if model.log_loss_gradient_cosines:
        for name in model.adaptive_loss_names:
            metric_keys.update(
                (f"grad_cos/{name}__total", f"grad_cos/{name}__rest", f"grad_norm/{name}")
            )
    metric_keys.update(f"loss_weight/{name}" for name in loss_names)

    writer = csv_logger.experiment
    existing_keys = set(getattr(writer, "metrics_keys", []))
    writer.metrics_keys = sorted(existing_keys | metric_keys)


def datamodule_from_splits(cfg, splits):
    """Wrap six train/val/test arrays, returning the datamodule, input width, and input stats."""
    params = cfg["parameters"]
    X_train, Y_train, X_val, Y_val, X_test, Y_test = (split.astype(np.float32) for split in splits)

    dm = WBosonDataModule(
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
    return dm, X_train.shape[1], compute_neural_input_stats(X_train)


def build_datamodule(cfg, data_path):
    splits = data.load_presplit_data(data_path, data_cfg=cfg.get("data", {}))
    return datamodule_from_splits(cfg, splits)


def create_loggers(cfg, model, saved_path, use_wandb):
    csv_logger = CSVLogger(save_dir=saved_path, name="logs", version=0)
    prime_csv_metric_header(csv_logger, model)

    if not use_wandb:
        print("Wandb logging disabled, only using CSVLogger.")
        return [csv_logger], None

    wandb_logger = WandbLogger(
        project="PCRES-regressor",
        name=Path(saved_path).name,
        save_dir=saved_path,
        log_model=True,
    )
    wandb_logger.experiment.config.update(flatten_config(cfg), allow_val_change=True)

    return [csv_logger, wandb_logger], wandb_logger


def build_training_callbacks(params):
    return [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=int(params.get("checkpoint_save_top_k", 16)),
            save_last=True,
            every_n_epochs=1,
            filename="reg-{epoch:02d}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=params.get("early_stopping_patience", 32),
            min_delta=params.get("early_stopping_min_delta", 0.0),
            mode="min",
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def evaluate_best_checkpoint(model, dm, checkpoint, accelerator):
    """Score the best checkpoint on the test split, when the datamodule has one."""
    if dm.test_ds is None or len(dm.test_ds) == 0:
        print("No test split available, skipping test evaluation.")
        return

    print("Running test evaluation with best checkpoint...")
    Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    ).test(
        model=model,
        datamodule=dm,
        ckpt_path=checkpoint.best_model_path,
        weights_only=False,
    )


def run_training(
    cfg,
    dm,
    input_dim,
    standardization,
    saved_path,
    use_wandb,
):
    params = cfg["parameters"]
    std_mean_train, std_scale_train = standardization
    print("Starting training...")
    print(f"Input dimension: {input_dim}")
    model = LightningWBoson(
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

    callbacks = build_training_callbacks(params)
    steps_per_epoch = max(1, len(dm.train_dataloader()))
    saved_path = clean_training_output(saved_path)
    loggers, wandb_logger = create_loggers(cfg, model, saved_path, use_wandb)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        max_epochs=params["epochs"],
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=steps_per_epoch,
        gradient_clip_val=params.get("gradient_clip_val", 1.0),
    )
    trainer.fit(model, datamodule=dm)

    checkpoint = callbacks[0]
    evaluate_best_checkpoint(model, dm, checkpoint, accelerator)

    if wandb_logger is not None:
        wandb_logger.experiment.finish()


def add_common_arguments(parser):
    """Flags shared by train.py and k_fold_train.py."""
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config file",
    )
    parser.add_argument("--wandb", "-w", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        help="Which physical GPU to use (sets CUDA_VISIBLE_DEVICES)",
    )
    return parser


def parse_args():
    return add_common_arguments(argparse.ArgumentParser()).parse_args()


def apply_cpu_affinity(params):
    """Keep the process off this host's unstable CPUs; [] disables the guard."""
    excluded = params.get("excluded_cpus", DEFAULT_EXCLUDED_CPUS)
    if not excluded or not hasattr(os, "sched_setaffinity"):
        return
    available = os.sched_getaffinity(0)
    allowed = available - set(excluded)
    if not allowed:
        print(
            f"Refusing to exclude every available CPU {sorted(available)}; leaving affinity as is"
        )
        return
    if allowed != available:
        os.sched_setaffinity(0, allowed)
        print(
            f"CPU affinity: excluded {sorted(set(excluded) & available)}, using {sorted(allowed)}"
        )


def configure_runtime(params, gpu=None):
    """Pin CPUs, select the GPU, cap thread pools, and seed the process."""
    apply_cpu_affinity(params)

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Using GPU {gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

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


def main(train=True, arg=None, config_path=DEFAULT_CONFIG):
    cfg = load_config(arg.config if arg is not None else config_path)
    configure_runtime(cfg["parameters"], arg.gpu if arg is not None else None)

    data_path = resolve_repo_path(cfg["paths"]["data_path"])
    dm, input_dim, standardization = build_datamodule(cfg, data_path)
    if not train:
        print("Evaluation mode, returning datamodule...")
        return dm

    run_training(
        cfg,
        dm,
        input_dim,
        standardization,
        cfg["paths"]["saved_path"],
        arg.wandb if arg is not None else False,
    )


if __name__ == "__main__":
    started_at = time()
    main(arg=parse_args())
    print(f"Total time: {time() - started_at:.2f} seconds.")
