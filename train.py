import os
import yaml
import argparse
import numpy as np

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger


from model import LightningWBoson
from data_module import WBosonDataModule
import load_data as data

# ====== Load config ======
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(train=True, arg=None):
    # ---------- load config ----------
    _cfg = load_config()
    _param = _cfg["parameters"]
    BATCH_SIZE = _param["batch_size"]
    EPOCHS = _param["epochs"]
    LEARNING_RATE = _param["learning_rate"]
    LOSS_WEIGHTS = _param["loss_weights"]
    D_MODEL = _param["d_model"]
    N_HEADS = _param["n_heads"]
    NUM_BLOCKS = _param["num_blocks"]
    NUM_WORKERS = _param.get("num_workers", 0)
    PERSISTENT_WORKERS = _param.get("persistent_workers", False)
    PIN_MEMORY = _param.get("pin_memory", torch.cuda.is_available())
    PREFETCH_FACTOR = _param.get("prefetch_factor", 2)

    # some stable settings for dataloader and numpy
    os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_WORKERS + 2)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS + 2)

    saved_path = _cfg["paths"]["saved_path"]
    data_path = _cfg["paths"]["data_path"]
    
    if train == True:
        if len(saved_path) > 0:
            print(f"Found existing checkpoint at {saved_path}, deleting entire folder...")
            os.system(f"rm -rf {saved_path}")
        else:
            print("No existing checkpoint found, starting fresh...")
    else:
        print("Evaluation mode, loading checkpoints...")
    
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium") # "high" is more accurate but slower
    llvv, ww, (std_mean_train, std_scale_train), _ = data.load_data(data_path) # llvv, WW
    X = llvv.astype(np.float32)
    Y = ww.astype(np.float32)
    dm = WBosonDataModule(
        X, Y,
        batch_size=BATCH_SIZE,
        val_frac=0.05,
        test_frac=0.01,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
    )
    dm.setup()

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
            num_heads=N_HEADS,
            num_blocks=NUM_BLOCKS
        )

        ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="reg-{epoch:02d}-{val_loss:.2f}")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=128,
            mode="min",
            verbose=False
        )

        steps_per_epoch = max(1, len(dm.train_dataloader()))
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
                
        csv_logger = CSVLogger(save_dir=saved_path, name="logs")
        
        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[ckpt, early_stopping],
            logger=[csv_logger, wandb_logger] if use_wandb else [csv_logger],
            log_every_n_steps=steps_per_epoch,
        )
        trainer.fit(model, datamodule=dm)
    else:
        print("Loading model from checkpoint for evaluation... return datamodule")
        return dm

if __name__ == "__main__":
    from time import time
    t0 = time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wandb', '-w', action='store_true', help='Enable wandb logging and training mode')
    args = argparser.parse_args()
    main(train=True, arg=args)
    t1 = time()
    print(f"Total time: {(t1 - t0):.2f} seconds.")
