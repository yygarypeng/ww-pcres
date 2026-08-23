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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "configs/config.yaml"


from data import compute_neural_input_stats
from data import load_data as data
from data.data_module import WBosonDataModule
from model import LightningWBoson
from model.losses import W_MASS_SCALE


def resolve_repo_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def clean_training_output(saved_path):
    if saved_path is None or not str(saved_path).strip():
        raise ValueError("paths.saved_path must be a non-empty directory path")
    saved_path = resolve_repo_path(saved_path)
    if saved_path.resolve() == Path(saved_path.anchor).resolve():
        raise ValueError("Refusing to delete filesystem root as paths.saved_path")
    if saved_path.exists():
        if not saved_path.is_dir():
            raise ValueError(f"paths.saved_path exists but is not a directory: {saved_path}")
        print(f"Found existing checkpoint at {saved_path}, deleting entire folder...")
        shutil.rmtree(saved_path)
    else:
        print("No existing checkpoint found, starting fresh...")


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
        metric_keys.update(f"grad_cos/{name}__total" for name in model.adaptive_loss_names)
        metric_keys.update(f"grad_cos/{name}__rest" for name in model.adaptive_loss_names)
    metric_keys.update(f"loss_weight/{name}" for name in loss_names)

    writer = csv_logger.experiment
    existing_keys = set(getattr(writer, "metrics_keys", []))
    writer.metrics_keys = sorted(existing_keys | metric_keys)


def compute_w_fourvec_scales(targets):
    w_fourvecs = targets[:, :8].reshape(-1, 4)
    scales = np.std(w_fourvecs, axis=0)
    return np.maximum(scales, np.finfo(np.float32).eps).astype(np.float32)


def compute_dmet_scales(features, targets):
    true_nu0_t = targets[:, :2] - features[:, :2]
    true_nu1_t = targets[:, 4:6] - features[:, 4:6]
    true_dmet = features[:, 16:18] - true_nu0_t - true_nu1_t
    scales = np.std(true_dmet, axis=0)
    return np.maximum(scales, np.finfo(np.float32).eps).astype(np.float32)


def compute_mass_mmd_standardization(targets):
    """Fit one robust scale shared by both charge-ordered W-mass slots."""
    masses = np.asarray(targets[:, 8:10], dtype=np.float64)
    transformed = np.arcsinh((masses / W_MASS_SCALE) ** 2).reshape(-1)
    transformed = transformed[np.isfinite(transformed)]
    if transformed.size == 0:
        raise ValueError("cannot fit mass MMD standardization without finite training masses")

    center = np.median(transformed)
    q25, q75 = np.percentile(transformed, [25.0, 75.0])
    scale = (q75 - q25) / 1.349
    scale = max(scale, np.finfo(np.float32).eps)
    return np.float32(center), np.float32(scale)


def build_datamodule(cfg, data_path):
    params = cfg["parameters"]
    splits = data.load_presplit_data(
        data_path,
        data_cfg=cfg.get("data", {}),
    )
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
    dm.setup()
    standardization = compute_neural_input_stats(X_train)
    w_fourvec_scales = compute_w_fourvec_scales(Y_train)
    mass_mmd_standardization = compute_mass_mmd_standardization(Y_train)
    dmet_scales = compute_dmet_scales(X_train, Y_train)
    return (
        dm,
        X_train.shape[1],
        standardization,
        w_fourvec_scales,
        mass_mmd_standardization,
        dmet_scales,
    )


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


class DeferredEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, **kwargs):
        if start_epoch < 0:
            raise ValueError("start_epoch must be a non-negative integer")
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch)

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self.start_epoch:
            return
        super()._run_early_stopping_check(trainer)


def build_training_callbacks(params):
    return [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=16,
            save_last=True,
            every_n_epochs=1,
            filename="reg-{epoch:02d}-{val_loss:.2f}",
        ),
        DeferredEarlyStopping(
            start_epoch=params.get("angular_mmd_ramp_epochs", 0),
            monitor="val_loss",
            patience=params.get("early_stopping_patience", 32),
            min_delta=params.get("early_stopping_min_delta", 0.0),
            mode="min",
            verbose=False,
        ),
    ]


def run_training(
    cfg,
    dm,
    input_dim,
    standardization,
    w_fourvec_scales,
    mass_mmd_standardization,
    dmet_scales,
    saved_path,
    use_wandb,
):
    params = cfg["parameters"]
    std_mean_train, std_scale_train = standardization
    mass_mmd_center, mass_mmd_scale = mass_mmd_standardization
    print("Starting training...")
    print(f"Input dimension: {input_dim}")
    model = LightningWBoson(
        input_dim=input_dim,
        std_mean_train=std_mean_train,
        std_scale_train=std_scale_train,
        w_fourvec_scales=w_fourvec_scales,
        mass_mmd_center=mass_mmd_center,
        mass_mmd_scale=mass_mmd_scale,
        dmet_scales=dmet_scales,
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-4),
        loss_weights=params["loss_weights"],
        mmd_config=cfg.get("mmd", {}),
        angular_mmd_ramp_epochs=params.get("angular_mmd_ramp_epochs", 0),
        adaptive_loss_weights=params.get("adaptive_loss_weights", False),
        log_loss_gradient_cosines=params.get("log_loss_gradient_cosines", False),
        higgs_mass_target=params.get("higgs_mass_target", 125.0),
        higgs_mass_scale=params.get("higgs_mass_scale", 10.0),
        higgs_mass_delta=params.get("higgs_mass_delta", 2.0),
        d_model=params["d_model"],
        num_heads=params["n_heads"],
        attention_blocks=params.get("attention_blocks", 4),
        attention_dropout=params.get("attention_dropout", 0.1),
        decoder_dropout=params.get("decoder_dropout", 0.1),
    )

    callbacks = build_training_callbacks(params)
    ckpt = callbacks[0]
    steps_per_epoch = max(1, len(dm.train_dataloader()))
    saved_path = resolve_repo_path(saved_path).resolve()
    clean_training_output(saved_path)
    loggers, wandb_logger = create_loggers(cfg, model, saved_path, use_wandb)

    trainer = Trainer(
        max_epochs=params["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=steps_per_epoch,
        gradient_clip_val=params.get("gradient_clip_val", 1.0),
    )
    trainer.fit(model, datamodule=dm)

    if dm.test_ds is not None and len(dm.test_ds) > 0:
        print("Running test evaluation with best checkpoint...")
        test_trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        test_trainer.test(
            model=model,
            datamodule=dm,
            ckpt_path=ckpt.best_model_path,
            weights_only=False,
        )
    else:
        print("No test split available, skipping test evaluation.")

    if wandb_logger is not None:
        wandb_logger.experiment.finish()


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def main(train=True, arg=None, config_path=DEFAULT_CONFIG):
    if arg is not None and hasattr(arg, "config"):
        config_path = arg.config
    cfg = load_config(config_path)
    params = cfg["parameters"]

    if arg is not None and getattr(arg, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu)
        print(f"Using GPU {arg.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    num_threads = str(params.get("num_workers", 0))
    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for variable in thread_variables:
        os.environ[variable] = num_threads

    seed_everything(params.get("seed", 114), workers=True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")

    data_path = resolve_repo_path(cfg["paths"]["data_path"])
    (
        dm,
        input_dim,
        standardization,
        w_fourvec_scales,
        mass_mmd_standardization,
        dmet_scales,
    ) = build_datamodule(cfg, data_path)
    if not train:
        print("Evaluation mode, returning datamodule...")
        return dm

    saved_path = resolve_repo_path(cfg["paths"]["saved_path"])
    run_training(
        cfg,
        dm,
        input_dim,
        standardization,
        w_fourvec_scales,
        mass_mmd_standardization,
        dmet_scales,
        saved_path,
        arg.wandb if arg is not None else False,
    )


if __name__ == "__main__":
    started_at = time()
    main(arg=parse_args())
    print(f"Total time: {time() - started_at:.2f} seconds.")
