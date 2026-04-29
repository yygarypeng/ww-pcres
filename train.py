import os
import shutil
import yaml
import argparse
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger


from model import LightningWBoson
from data_module import WBosonDataModule
import load_data as data


def clean_training_output(saved_path):
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

# ====== Load config ======
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def prime_csv_metric_header(csv_logger, model):
    """Pre-register CSV columns to avoid Lightning header rewrite failures."""
    metric_keys = {"epoch", "step"}
    loss_names = sorted(model.loss_weights.keys())
    for prefix in ("", "val_", "test_"):
        metric_keys.add(f"{prefix}loss")
        metric_keys.update(f"{prefix}{name}_loss" for name in loss_names)

    writer = csv_logger.experiment
    existing_keys = set(getattr(writer, "metrics_keys", []))
    writer.metrics_keys = sorted(existing_keys | metric_keys)


def main(train=True, arg=None, config_path="config.yaml"):
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

    # some stable settings for dataloader and numpy
    os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS + 2)

    saved_path = _cfg["paths"]["saved_path"]
    data_path = _cfg["paths"]["data_path"]
    
    if train != True:
        print("Evaluation mode, loading checkpoints...")

    seed_everything(SEED, workers=True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium") # "high" is more accurate but slower
    llvv, ww, _, _ = data.load_data(
        data_path,
        categories=DATA_CFG.get("categories"),
    ) # llvv, WW
    X = llvv.astype(np.float32)
    Y = ww.astype(np.float32)
    dm = WBosonDataModule(
        X, Y,
        batch_size=BATCH_SIZE,
        val_frac=DATA_CFG.get("val_frac", 0.05),
        test_frac=DATA_CFG.get("test_frac", 0.01),
        seed=SEED,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
    )
    dm.setup()
    (std_mean_train, std_scale_train), _ = data.compute_standardization_stats(
        X,
        Y,
        train_indices=dm.train_indices_array(),
    )

    if train == True:
        print("Starting training...")
        input_dim = X.shape[1]
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
            wandb_logger = WandbLogger(
                project="PCRES-regressor",
                name=f"wandb-logs",
                save_dir=saved_path,
                log_model=True,
            )
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
        )
        trainer.fit(model, datamodule=dm)
    else:
        print("Loading model from checkpoint for evaluation... return datamodule")
        return dm

if __name__ == "__main__":
    from time import time
    t0 = time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config file")
    argparser.add_argument("--wandb", "-w", action="store_true", help="Enable wandb logging and training mode")
    args = argparser.parse_args()
    main(train=True, arg=args)
    t1 = time()
    print(f"Total time: {(t1 - t0):.2f} seconds.")
