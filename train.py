import os
import glob

import numpy as np

import torch
from torch.utils.data import TensorDataset
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from model import LightningWBoson
from data_module import WBosonDataModule
import load_data as data

# ====== Hyperparameters constants ======
BATCH_SIZE = 512
EPOCHS = 2048
LEARNING_RATE = 1e-5
LOSS_WEIGHTS = {"mae": 1.0, "w_mass_mmd0": 12.0, "w_mass_mmd1": 12.0}

# ====== main parameters ======
project_name = "hww_pcres_regressor"
saved_path = f"/root/work/hww_pcres_regressor/{project_name}"
ckpt_path = glob.glob(saved_path)
data_path = [
    "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_ggF_train.h5",
    "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_ggF_validate.h5",
    "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_ggF_test.h5"
]

def main(train=True):
    if train == True:
        if len(ckpt_path) > 0:
            print(f"Found existing checkpoint at {ckpt_path[0]}, deleting entire folder...")
            os.system(f"rm -rf {ckpt_path[0]}")
        else:
            print("No existing checkpoint found, starting fresh...")
    else:
        print("Evaluation mode, loading checkpoints...")
        
    torch.set_float32_matmul_precision("medium")
    train_inputs, train_labels = data.load_data(data_path[0])
    val_inputs, val_labels = data.load_data(data_path[1])
    test_inputs, test_labels = data.load_data(data_path[2])
    train_ds = TensorDataset(torch.from_numpy(train_inputs).float(), torch.from_numpy(train_labels).float())
    val_ds = TensorDataset(torch.from_numpy(val_inputs).float(), torch.from_numpy(val_labels).float())
    test_ds = TensorDataset(torch.from_numpy(test_inputs).float(), torch.from_numpy(test_labels).float())

    dm = WBosonDataModule(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, batch_size=BATCH_SIZE)
    
    if train == True:
        print("Starting training...")
        input_dim = train_inputs.shape[1]
        model = LightningWBoson(
            input_dim=input_dim,
            lr=LEARNING_RATE,
            loss_weights=LOSS_WEIGHTS
        )

        ckpt = ModelCheckpoint(monitor="val_mae_loss", mode="min", save_top_k=1, filename="reg-{epoch:02d}-{val_mae_loss:.2f}")
        early_stopping = EarlyStopping(
            monitor="val_mae_loss",
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
