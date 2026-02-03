import os
import glob
import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from model import LightningWBoson
from data_module import WBosonDataModule
import load_data as data

# ====== Hyperparameters constants ======
BATCH_SIZE = 256
EPOCHS = 1024
LEARNING_RATE = 1e-5
LOSS_WEIGHTS = {
    "huber": 1.0,
    "w_mass_mmd0": 5.0,
    "w_mass_mmd1": 5.0,
    "higgs_mass": 0.5,
}

# ====== main parameters ======
project_name = "hww_pcres_regressor_kfold"
saved_path = f"/root/work/hww_pcres_regressor/{project_name}"
data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5"
SEED = 114

def main(train=True):

    # ---------- housekeeping ----------
    if train == True:
        if os.path.exists(saved_path):
            print(f"Found existing checkpoint at {saved_path}, deleting...")
            os.system(f"rm -rf {saved_path}")
        else:
            print("No existing checkpoint found, starting fresh...")
    else:
        print("Evaluation mode, loading checkpoints...")

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")

    # ---------- load data ----------
    llvv, ww, (std_mean_train, std_scale_train), _ = data.load_data(data_path)
    X = llvv.astype(np.float32)
    Y = ww.astype(np.float32)

    # ---------- WORKFLOW ----------
    # prepare train/test splits
    input_idx, testing_idx = train_test_split(np.arange(X.shape[0]), test_size=0.01, random_state=SEED)

    # ---------- EVAL ----------
    if not train:
        print("Loading model from checkpoint for evaluation.")
        _train_idx, _val_idx = train_test_split(input_idx, test_size=0.5, random_state=SEED)
        dm = WBosonDataModule(
			X, Y,
			train_idx=_train_idx,
			val_idx=_val_idx,
			test_idx=testing_idx,
			batch_size=BATCH_SIZE,
		)
        print("Setting up testing data module...")
        return dm

	# ---------- TRAIN ----------
    if train:
        print("Starting 2-fold training...")

        # Split only the training portion
        X = X[input_idx]
        Y = Y[input_idx]

        all_idx = np.arange(X.shape[0])
        even_idx = all_idx[all_idx % 2 == 0]
        odd_idx = all_idx[all_idx % 2 == 1]

        # Fold 0: train on even, val on odd
        # Fold 1: train on odd, val on even
        folds = [(even_idx, odd_idx), (odd_idx, even_idx)]

        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"\n========== Fold {fold} ==========")
            print(f"Training data size: {len(train_idx)} samples.")
            print(f"Validation data size: {len(val_idx)} samples.")

            dm = WBosonDataModule(
                X, Y,
                train_idx=train_idx,
                val_idx=val_idx,
                batch_size=BATCH_SIZE,
            )
            dm.setup()

            input_dim = X.shape[1]
            print(f"Input dimension: {input_dim}")
            model = LightningWBoson(
				input_dim=input_dim,
				std_mean_train=std_mean_train,
				std_scale_train=std_scale_train,
				lr=LEARNING_RATE,
				loss_weights=LOSS_WEIGHTS,
			)

            ckpt = ModelCheckpoint(
                monitor="val_huber_loss",
                mode="min",
                save_top_k=1,
                filename=f"{{epoch:03d}}-{{val_huber_loss:.4f}}-fold{fold}",
            )

            early_stopping = EarlyStopping(
                monitor="val_huber_loss",
                patience=32,
                mode="min",
            )

            trainer = Trainer(
                max_epochs=EPOCHS,
                accelerator="auto",
                devices="auto",
                callbacks=[ckpt, early_stopping],
                logger=CSVLogger(
                    save_dir=saved_path,
                    name=f"fold{fold}",
                ),
                log_every_n_steps=1,
            )

            trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    from time import time
    t0 = time()
    main(train=True)
    print(f"Total time: {time() - t0:.1f} seconds.")