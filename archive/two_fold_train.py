import argparse

import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split

from model import LightningWBoson
from data import load_data as data
from data.data_module import WBosonDataModule
from train import clean_training_output, prime_csv_metric_header

# ====== Hyperparameters constants ======
BATCH_SIZE = 256
EPOCHS = 1024
LEARNING_RATE = 1e-5
D_MODEL = 256
N_HEADS = 16
LOSS_WEIGHTS = {
    "huber": 1.0,
    "w_mass_mmd0": 5.0,
    "w_mass_mmd1": 5.0,
    "higgs_mass": 0.5,
    "aux_mom_mmd0": 0.0,
    "aux_mom_mmd1": 0.0,
}

# ====== main parameters ======
project_name = "hww_pcres_regressor_kfold"
saved_path = f"/root/work/hww_pcres_regressor/{project_name}"
data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5"
SEED = 114

def main(train=True, arg=None):
    run_saved_path = arg.saved_path if arg is not None and arg.saved_path else saved_path
    run_data_path = arg.data_path if arg is not None and arg.data_path else data_path
    use_wandb = bool(arg is not None and arg.wandb)

    # ---------- housekeeping ----------
    if train != True:
        print("Evaluation mode, loading checkpoints...")

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")

    # ---------- load data ----------
    llvv, ww, _, _ = data.load_data(run_data_path)
    X = llvv.astype(np.float32)
    Y = ww.astype(np.float32)

    # ---------- WORKFLOW ----------
    # prepare train/test splits
    input_idx, testing_idx = train_test_split(np.arange(X.shape[0]), test_size=0.01, random_state=SEED)

    # ---------- eval ----------
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
        dm.setup()
        return dm

	# ---------- train ----------
    if train:
        print("Starting 2-fold training...")
        clean_training_output(run_saved_path)
        
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
            (std_mean_train, std_scale_train), _ = data.compute_standardization_stats(
                X,
                Y,
                train_indices=train_idx,
            )
            model = LightningWBoson(
				input_dim=input_dim,
				d_model=D_MODEL,
				num_heads=N_HEADS,
				std_mean_train=std_mean_train,
				std_scale_train=std_scale_train,
				lr=LEARNING_RATE,
				loss_weights=LOSS_WEIGHTS,
			)

            ckpt = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename=f"{{epoch:03d}}-{{val_loss:.4f}}-fold{fold}",
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=32,
                mode="min",
            )

            csv_logger = CSVLogger(
                save_dir=run_saved_path,
                name=f"fold{fold}",
                version=0,
            )
            prime_csv_metric_header(csv_logger, model)
            step_per_epoch = max(1, len(dm.train_dataloader()))
            
            if use_wandb:
                wandb_logger = WandbLogger(
                    project=project_name,
                    name=f"fold{fold}",
                    save_dir=run_saved_path,
                    log_model=True,
                )
                
                wandb_logger.watch(model, log="all", log_freq=step_per_epoch, log_graph=False)

            trainer = Trainer(
                max_epochs=EPOCHS,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[ckpt, early_stopping],
                logger=[csv_logger, wandb_logger] if use_wandb else [csv_logger],
                log_every_n_steps=step_per_epoch,
            )

            trainer.fit(model, datamodule=dm)
            if use_wandb:
                wandb_logger.experiment.finish()


if __name__ == "__main__":
    from time import time
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wandb', '-w', action='store_true', help='Enable wandb logging and training mode')
    argparser.add_argument('--data-path', default=None, help='Override input HDF5 path')
    argparser.add_argument('--saved-path', default=None, help='Override output directory')
    args = argparser.parse_args()
    
    t0 = time()
    main(train=True, arg=args)
    print(f"Total time: {time() - t0:.1f} seconds.")
