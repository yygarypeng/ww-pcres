import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as L

class ArrayDataset(Dataset):
    """
    Generic (X, Y) dataset.
    Expect shapes:
    X: (..., input_dim)   with [lep0(4), lep1(4), MET_px, MET_py, ...]
    Y: (..., >=10)        first 8 entries are target W four-vectors, Y[..., 8], Y[..., 9] = true W masses
    """
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class WBosonDataModule(L.LightningDataModule):
    def __init__(self, X, Y, batch_size=512, num_workers=4, val_frac=0.1, test_frac=0.1):
        super().__init__()
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.test_frac = test_frac

    def setup(self, stage=None):
        ds = ArrayDataset(self.X, self.Y)
        n = len(ds)
        n_val = int(self.val_frac * n)
        n_test = int(self.test_frac * n)
        n_train = n - n_val - n_test
        self.train_ds, self.val_ds, self.test_ds = random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(114)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=True)
