import torch
import torch.nn as nn
import pytorch_lightning as L

from layers import DenseDropoutBlock, ResidualBlock, NeutrinosLayer
from losses import (
    mae_loss, w_mass_mmd_losses,
)


class WBosonRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        blocks = []
        dim = input_dim
        for _ in range(3):
            blocks.append(ResidualBlock(dim, 256, dropout=0.3))
            dim = 256
            blocks.append(ResidualBlock(dim, 512, dropout=0.3))
            dim = 512
        for _ in range(3):
            blocks.append(ResidualBlock(dim, 256, dropout=0.3))
            dim = 256
            blocks.append(ResidualBlock(dim, 128, dropout=0.3))
            dim = 128
        self.trunk = nn.Sequential(*blocks)
        self.to_64 = DenseDropoutBlock(dim, 64, dropout=0.0)
        self.to_32 = DenseDropoutBlock(64, 32, dropout=0.0)
        self.leplep_out = nn.Linear(32, 8)
        self.nunu_layer = NeutrinosLayer()

    def forward(self, x):
        h = self.trunk(x)
        h = self.to_64(h)
        h = self.to_32(h)
        leplep_4vec = self.leplep_out(h)
        return self.nunu_layer(leplep_4vec[..., :4], leplep_4vec[..., 4:8])


class LightningWBoson(L.LightningModule):
    def __init__(self, input_dim, lr=1e-4, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = WBosonRegressor(input_dim) # give a base model structure for forward() 
        defaults = {
            "mae": 1.0,
            "w_mass_mmd0": 10.0,
            "w_mass_mmd1": 10.0,
        }
        self.loss_weights = {**defaults, **(loss_weights or {})}
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, x, y, y_pred):
        losses = {
            "mae": mae_loss(y, y_pred),
            # "nu_mass": nu_mass_loss(x, y_pred),
            # "higgs_mass": higgs_mass_loss(y_pred),
            # "w0_mass_mae": w_mass_mae_losses(y, y_pred)[0],
            # "w1_mass_mae": w_mass_mae_losses(y, y_pred)[1],
            "w_mass_mmd0": w_mass_mmd_losses(y, y_pred)[0],
            "w_mass_mmd1": w_mass_mmd_losses(y, y_pred)[1],
            # "dinu_pt": dinu_pt_loss(x, y_pred),
            # "neg_r2": neg_r2_loss(y, y_pred),
        }
        total = sum(self.loss_weights[k] * v for k, v in losses.items())
        return total.mean(), losses

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total, prog_bar=False, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v, prog_bar=False, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # move to GPU asynchronously
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        total, losses = self._compute_losses(x, y, self(x)) # self(x) is equiv to self.forward(x) as it defined in __call__() internally in nn.Module
        self._log_losses("", losses, total)
        return total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # move to GPU asynchronously
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        total, losses = self._compute_losses(x, y, self(x))
        self._log_losses("val_", losses, total)
        return total

    def test_step(self, batch, batch_idx):
        x, y = batch
        # move to GPU asynchronously
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        total, losses = self._compute_losses(x, y, self(x))
        self._log_losses("test_", losses, total)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
