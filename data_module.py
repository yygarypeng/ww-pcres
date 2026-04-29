import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pytorch_lightning as L


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
    Supports BOTH:
        1) Random split via val_frac / test_frac
        2) Explicit KFold splits via train_idx / val_idx
    """

    def __init__(
        self,
        X,
        Y,
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
        self.std_ds = None

    def setup(self, stage=None):
        ds = ArrayDataset(self.X, self.Y)
        self.std_ds = ds  # keep for (inverse) transform.

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
            "generator": self._make_generator(),
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

        return self._dataloader(self.test_ds, shuffle=False)
