import os
import argparse
from pathlib import Path
import sys

import numpy as np

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
from train import clean_training_output, flatten_config, load_config, prime_csv_metric_header


def resolve_repo_path(raw_path):
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


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
            d_model=D_MODEL,
            num_heads=N_HEADS
        )

        ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="reg-{epoch:02d}-{val_loss:.2f}")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=64,
            mode="min",
            verbose=False
        )

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
            callbacks=[ckpt, early_stopping],
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
