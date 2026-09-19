import os
import random

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32).contiguous()


def _check_rows(X, Y, suffix=""):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X{suffix} and Y{suffix} must have the same number of samples, got "
            f"{X.shape[0]} and {Y.shape[0]}"
        )


def _as_split(X, Y, split_x, split_y, split):
    """Check one extra split against the training arrays and wrap it as a dataset."""
    split_x, split_y = _as_tensor(split_x), _as_tensor(split_y)
    _check_rows(split_x, split_y, f"_{split}")
    if X.shape[1:] != split_x.shape[1:]:
        raise ValueError(
            f"X and X_{split} must have matching feature dimensions, got "
            f"{tuple(X.shape[1:])} and {tuple(split_x.shape[1:])}"
        )
    if Y.shape[1:] != split_y.shape[1:]:
        raise ValueError(
            f"Y and Y_{split} must have matching target dimensions, got "
            f"{tuple(Y.shape[1:])} and {tuple(split_y.shape[1:])}"
        )
    return TensorDataset(split_x, split_y)


class WBosonDataModule(L.LightningDataModule):
    """Serve pre-split training, validation, and optional test arrays."""

    def __init__(
        self,
        X,
        Y,
        X_val,
        Y_val,
        X_test=None,
        Y_test=None,
        seed=2330,
        batch_size=512,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
    ):
        super().__init__()

        X, Y = _as_tensor(X), _as_tensor(Y)
        _check_rows(X, Y)
        if (X_test is None) != (Y_test is None):
            raise ValueError("X_test and Y_test must be provided together")

        self.seed = int(seed)
        self.train_ds = TensorDataset(X, Y)
        self.val_ds = _as_split(X, Y, X_val, Y_val, "val")
        self.test_ds = None if X_test is None else _as_split(X, Y, X_test, Y_test, "test")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._train_generator = torch.Generator().manual_seed(self.seed)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())

        if num_workers is None:
            num_workers = max(1, int((os.cpu_count() or 1) * 0.8))
        self.num_workers = max(0, int(num_workers))
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        # DataLoader only accepts a prefetch factor when it uses worker processes.
        self.prefetch_factor = (
            max(1, int(prefetch_factor))
            if self.num_workers > 0 and prefetch_factor is not None
            else None
        )

        print(f"Using {self.num_workers} workers in data loading.")
        print(f"Train split: {len(self.train_ds)} samples")
        print(f"Validation split: {len(self.val_ds)} samples")
        if self.test_ds is None:
            print("Test split: not provided")
        else:
            print(f"Test split: {len(self.test_ds)} samples")
        print(f"Feature dims: {tuple(X.shape[1:])}, target dims: {tuple(Y.shape[1:])}")

    def state_dict(self):
        return {"train_generator_state": self._train_generator.get_state()}

    def load_state_dict(self, state_dict):
        self._train_generator.set_state(state_dict["train_generator_state"])

    def _worker_init_fn(self, worker_id):
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _dataloader(self, dataset, shuffle):
        # Shuffling advances one long-lived generator, so a resumed run keeps its epoch order.
        generator = self._train_generator if shuffle else torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=self._worker_init_fn,
            generator=generator,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return self._dataloader(self.test_ds, shuffle=False)
