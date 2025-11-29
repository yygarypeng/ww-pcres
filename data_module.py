import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L

class WBosonDataModule(L.LightningDataModule):
    """
    DataModule that accepts pre-split Dataset objects.
    Provide train_ds (required) and optional val_ds/test_ds.
    """
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, batch_size=512, num_workers=4):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Nothing to create if user supplied datasets.
        if self.train_ds is not None:
            return
        raise ValueError("WBosonDataModule requires pre-split datasets: provide train_ds at minimum")

    def train_dataloader(self):
        if self.train_ds is None:
            raise ValueError("train_ds is not provided")
        print("Creating train dataloader...")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        print("Creating val dataloader...")
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        print("Creating test dataloader...")
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)