import numpy as np
import torch
import wandb
import glob
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from model import LightningWBoson
from data_module import WBosonDataModule
import load_data as data

# ====== Hyperparameters constants ======
BATCH_SIZE = 1024
EPOCHS = 1024
LEARNING_RATE = 1e-5
LOSS_WEIGHTS = {"mae": 1.0, "w_mass_mmd0": 10.0, "w_mass_mmd1": 10.0}

# ====== main parameters ======
project_name = "hww_pcres_regressor"
saved_path = f"/root/work/hww_pcres_regressor/{project_name}"
ckpt_path = glob.glob(saved_path)
data_path = "/root/data/mc20_truth_v4_SM.h5"

def main(train=True):
    if train == True:
        if len(ckpt_path) > 0:
            print(f"Found existing checkpoint at {ckpt_path[0]}, deleting entire folder...")
            os.system(f"rm -rf {ckpt_path[0]}")
        else:
            print("No existing checkpoint found, starting fresh...")
    else:
        print("Evaluation mode, loading checkpoints...")
        
    torch.set_float32_matmul_precision("high")
    train_obj, target_obj = data.load_data(data_path)
    X = train_obj.astype(np.float32)
    Y = target_obj.astype(np.float32)
    input_dim = X.shape[1]

    dm = WBosonDataModule(
        X, Y,
        batch_size=BATCH_SIZE,
        val_frac=0.1,
        test_frac=0.05
    )
    
    if train == True:
        print("Starting training...")
        model = LightningWBoson(
            input_dim=input_dim,
            lr=LEARNING_RATE,
            loss_weights=LOSS_WEIGHTS
        )

        ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="reg-{epoch:02d}-{val_loss:.2f}")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=32,
            mode="min",
            verbose=False
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=1,
            callbacks=[ckpt, early_stopping],
            logger=CSVLogger(save_dir=saved_path, name="logs")
        )
        trainer.fit(model, datamodule=dm)
    else:
        print("Loading model from checkpoint for evaluation... return datamodule")
        return dm

if __name__ == "__main__":
    from time import time
    t0 = time()
    main()
    t1 = time()
    print(f"Total time: {(t1 - t0):.2f} seconds.")
