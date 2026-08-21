import os
import random

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def _validate_split(reference_x, reference_y, split_x, split_y, split_name) -> None:
    if split_x.shape[0] != split_y.shape[0]:
        raise ValueError(
            f"X_{split_name} and Y_{split_name} must have the same number of samples, got "
            f"{split_x.shape[0]} and {split_y.shape[0]}"
        )
    if reference_x.shape[1:] != split_x.shape[1:]:
        raise ValueError(
            f"X and X_{split_name} must have matching feature dimensions, got "
            f"{tuple(reference_x.shape[1:])} and {tuple(split_x.shape[1:])}"
        )
    if reference_y.shape[1:] != split_y.shape[1:]:
        raise ValueError(
            f"Y and Y_{split_name} must have matching target dimensions, got "
            f"{tuple(reference_y.shape[1:])} and {tuple(split_y.shape[1:])}"
        )


class ArrayDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float32).contiguous()
        self.Y = torch.as_tensor(Y, dtype=torch.float32).contiguous()

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, got "
                f"{self.X.shape[0]} and {self.Y.shape[0]}"
            )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class WBosonDataModule(L.LightningDataModule):
    """
    Supports:
        1) Pre-split arrays via X_val / Y_val / X_test / Y_test
        2) Random split via val_frac / test_frac
        3) Explicit KFold splits via train_idx / val_idx
    """

    def __init__(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        X_test=None,
        Y_test=None,
        seed=114,
        batch_size=512,
        val_frac=0.1,
        test_frac=0.1,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
    ):
        super().__init__()

        self.X = torch.as_tensor(X, dtype=torch.float32).contiguous()
        self.Y = torch.as_tensor(Y, dtype=torch.float32).contiguous()
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, got "
                f"{self.X.shape[0]} and {self.Y.shape[0]}"
            )

        if (X_val is None) != (Y_val is None):
            raise ValueError("X_val and Y_val must be provided together")
        if (X_test is None) != (Y_test is None):
            raise ValueError("X_test and Y_test must be provided together")

        self.X_val = (
            None if X_val is None else torch.as_tensor(X_val, dtype=torch.float32).contiguous()
        )
        self.Y_val = (
            None if Y_val is None else torch.as_tensor(Y_val, dtype=torch.float32).contiguous()
        )
        self.X_test = (
            None if X_test is None else torch.as_tensor(X_test, dtype=torch.float32).contiguous()
        )
        self.Y_test = (
            None if Y_test is None else torch.as_tensor(Y_test, dtype=torch.float32).contiguous()
        )

        self.use_presplit = self.X_val is not None or self.X_test is not None
        if self.use_presplit:
            if self.X_val is None:
                raise ValueError("Pre-split mode requires X_val and Y_val")
            if any(idx is not None for idx in (train_idx, val_idx, test_idx)):
                raise ValueError("Pre-split arrays cannot be used together with split indices")
            _validate_split(self.X, self.Y, self.X_val, self.Y_val, "val")
            if self.X_test is not None:
                _validate_split(self.X, self.Y, self.X_test, self.Y_test, "test")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.val_frac = float(val_frac)
        self.test_frac = float(test_frac)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.prefetch_factor = prefetch_factor
        self.seed = int(seed)
        self._train_generator = torch.Generator().manual_seed(self.seed)

        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            self.num_workers = max(1, int(cpu_count * 0.8))
            print(f"Automatically setting num_workers to {self.num_workers}")
        else:
            self.num_workers = max(0, int(num_workers))
            print(f"Using {self.num_workers} workers in data loading.")

        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        if self.num_workers == 0:
            self.prefetch_factor = None
        elif self.prefetch_factor is not None:
            self.prefetch_factor = max(1, int(self.prefetch_factor))

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        ds = ArrayDataset(self.X, self.Y)

        # -------- Pre-split dataset --------
        if self.use_presplit:
            print("Using pre-split dataset (train / val / test arrays)")

            self.train_ds = ds
            self.val_ds = ArrayDataset(self.X_val, self.Y_val)
            self.test_ds = None
            if self.X_test is not None:
                self.test_ds = ArrayDataset(self.X_test, self.Y_test)

            print(f"Train split: {len(self.train_ds)} samples")
            print(f"Validation split: {len(self.val_ds)} samples")
            if self.test_ds is not None:
                print(f"Test split: {len(self.test_ds)} samples")
                print(
                    f"Feature dims: train={tuple(self.X.shape[1:])}, "
                    f"val={tuple(self.X_val.shape[1:])}, "
                    f"test={tuple(self.X_test.shape[1:])}"
                )
                print(
                    f"Target dims: train={tuple(self.Y.shape[1:])}, "
                    f"val={tuple(self.Y_val.shape[1:])}, "
                    f"test={tuple(self.Y_test.shape[1:])}"
                )
            else:
                print("Test split: not provided")
                print(
                    f"Feature dims: train={tuple(self.X.shape[1:])}, "
                    f"val={tuple(self.X_val.shape[1:])}"
                )
                print(
                    f"Target dims: train={tuple(self.Y.shape[1:])}, "
                    f"val={tuple(self.Y_val.shape[1:])}"
                )
            return

        # -------- KFold / explicit split --------
        if self.train_idx is not None and self.val_idx is not None:
            print("Using explicit (KFold) dataset split")

            self.train_ds = Subset(ds, self.train_idx)
            self.val_ds = Subset(ds, self.val_idx)

            if self.test_idx is not None:
                self.test_ds = Subset(ds, self.test_idx)
            else:
                self.test_ds = None

        # -------- Random split (original behavior) --------
        else:
            print("Using random split (val_frac / test_frac)")

            n = len(ds)
            if not 0 <= self.val_frac < 1 or not 0 <= self.test_frac < 1:
                raise ValueError("val_frac and test_frac must be in [0, 1)")
            if self.val_frac + self.test_frac >= 1:
                raise ValueError("val_frac + test_frac must be less than 1")

            n_val = int(self.val_frac * n)
            n_test = int(self.test_frac * n)
            n_train = n - n_val - n_test
            if n_train <= 0:
                raise ValueError(
                    f"Random split leaves no training samples: "
                    f"n={n}, val_frac={self.val_frac}, test_frac={self.test_frac}"
                )

            self.train_ds, self.val_ds, self.test_ds = random_split(
                ds,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

    def _worker_init_fn(self, worker_id):
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _make_generator(self):
        return torch.Generator().manual_seed(self.seed)

    def state_dict(self):
        return {"train_generator_state": self._train_generator.get_state()}

    def load_state_dict(self, state_dict):
        self._train_generator.set_state(state_dict["train_generator_state"])

    def train_indices_array(self):
        if self.train_ds is None:
            raise RuntimeError("DataModule.setup() must be called before requesting split indices")
        if isinstance(self.train_ds, Subset):
            return np.asarray(self.train_ds.indices, dtype=np.int64)
        return np.arange(len(self.train_ds), dtype=np.int64)

    def _dataloader(self, dataset, *, shuffle):
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders")

        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "worker_init_fn": self._worker_init_fn,
            "generator": self._train_generator if shuffle else self._make_generator(),
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        print("Use test_dataloader() with test_ds of length", len(self.test_ds))
        return self._dataloader(self.test_ds, shuffle=False)
