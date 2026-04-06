import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

from layers import Standardization, SelfAttentionBlock, ResidualBlock, WBosonFourVectorLayer
from losses import (
    huber_loss, w_mass_mmd_losses, higgs_mass_loss, 
    neg_r2_loss, nu_mass_loss, dinu_pt_loss, w_mass_mae_losses, aux_mom_mmd_loss
)


class WBosonRegressor(nn.Module):
    def __init__(
            self, 
            input_dim,
            d_model, num_heads, num_blocks,
            std_mean_train, std_scale_train
        ):
        super().__init__()
        
        # do the normalization (need to use large batch size for stable stats)
        self.norm = Standardization(std_mean_train, std_scale_train)
        
        # self-attention to capture global feature interactions
        self.lep_embed = nn.Linear(4, d_model)
        self.jet_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        self.dilep_embed = nn.Linear(4, d_model)
        hl_input_dim = input_dim - (4 + 4 + 4*3 + 2 + 4)
        self.hl_embed = nn.Linear(hl_input_dim, d_model)
        # addtional tokens for global context
        # self.px_embed = nn.Linear(7, d_model)
        # self.py_embed = nn.Linear(7, d_model)
        # self.pz_embed = nn.Linear(6, d_model)
        # self.energy_embed = nn.Linear(6, d_model)
        # self.dphi_embed = nn.Linear(4, d_model)
        # self.dlong_ll_embed = nn.Linear(2, d_model) # dilep longitundal features: dilep_eta and dr_ll
        
        # number of tokens = 8 objects + 6 additional tokens
        # self.num_tokens = 8 + 6
        self.num_tokens = 8
        # # type embedding
        # self.type_embed = nn.Embedding(self.num_tokens, d_model) # [num_tokens, d_model]
        # token normalization
        # self.tokens_norm = nn.LayerNorm(d_model)
        # self-attention blocks for global context refinement
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, dropout=0.5) for _ in range(4)
        ])
        print(f"Using {len(self.sa_blocks)} SA blocks.")
        
        # residual decoder blocks
        _dim = 512 if d_model * self.num_tokens >= 512 else d_model * self.num_tokens
        blocks = [nn.Linear(d_model * self.num_tokens, _dim)] # reduce dimension after flattening
        blocks.append(ResidualBlock(_dim, 128, dropout=0.5))
        blocks.extend([ResidualBlock(128, 128, dropout=0.5) for _ in range(num_blocks)])
        blocks.append(ResidualBlock(128, 256, dropout=0.5))
        self.trunk = nn.Sequential(*blocks)

        # nu momentum regression head
        self.nu_mom_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )
        
        # W bosons decoder
        self.w_layer = WBosonFourVectorLayer()

    def global_feature_aggregation(self, x):
        # standardize input features
        x_std = self.norm(x)
        
        # Key mask for empty jets
        batch_size = x.shape[0]
        key_mask = torch.zeros((batch_size, self.num_tokens), dtype=torch.bool, device=x.device)
        key_mask[:, 2] = (x[:, 8:12].abs().sum(dim=1) == 0)
        key_mask[:, 3] = (x[:, 12:16].abs().sum(dim=1) == 0)
        key_mask[:, 4] = (x[:, 16:20].abs().sum(dim=1) == 0)

        # embedding to get initial context
        l0 = self.lep_embed(x_std[:, 0:4])
        l1 = self.lep_embed(x_std[:, 4:8])
        j0 = self.jet_embed(x_std[:, 8:12])
        j1 = self.jet_embed(x_std[:, 12:16])
        j2 = self.jet_embed(x_std[:, 16:20])
        met = self.met_embed(x_std[:, 20:22])
        dilep = self.dilep_embed(x_std[:, 22:26])
        hl = self.hl_embed(x_std[:, 26:])
        # additional tokens
        # px = self.px_embed(x_std[:, [0, 4, 8, 12, 16, 20, 22]])
        # py = self.py_embed(x_std[:, [1, 5, 9, 13, 17, 21, 23]])
        # pz = self.pz_embed(x_std[:, [2, 6, 10, 14, 18, 24]])
        # energy = self.energy_embed(x_std[:, [3, 7, 11, 15, 19, 25]])
        # dphi = self.dphi_embed(x_std[:, [27, 28, 29, 30]])
        # dlong_ll = self.dlong_ll_embed(x_std[:, [26, 31]])

        # token type embedding
        # type_ids = torch.arange(self.num_tokens, device=x.device) # [num_tokens]
        # type_embed = self.type_embed(type_ids) # [num_tokens, d_model]
        # Combine into Context: [Batch, 8 + 6, d_model]
        context = torch.stack([
            l0, l1, j0, j1, j2, met, dilep, hl, 
            # px, py, pz, energy, dphi, dlong_ll
        ], dim=1)
        
        for refiner in self.sa_blocks:
            context = refiner(context, key_padding_mask=key_mask) # [B, num_tokens, d_model]
            context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)

        return context.reshape(batch_size, -1) # flatten to [B, num_tokens * d_model]

    def predict_nu_mom(self, x):
        h = self.global_feature_aggregation(x)
        h = self.trunk(h)
        return self.nu_mom_head(h)

    def forward(self, x):
        lep0, lep1 = x[..., :4], x[..., 4:8]
        nu_3mom = self.predict_nu_mom(x)
        org  = self.w_layer(lep0, lep1, nu_3mom)
        # swap = self.w_layer(lep1, lep0, nu_3mom)
        return org


class LightningWBoson(L.LightningModule):
    def __init__(
            self, 
            input_dim,
            d_model, num_heads, num_blocks,
            std_mean_train, std_scale_train, 
            lr=1e-4, loss_weights=None
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = WBosonRegressor(
            input_dim, 
            d_model, num_heads, num_blocks,
            std_mean_train, std_scale_train,
        ) # give a base model structure for forward() 
        defaults = {
            "huber": 1.0, 
            "higgs_mass": 0.0,
            "w_mass_mmd0": 0.0, "w_mass_mmd1": 0.0,
            "nu_mass": 0.0, 
            "dinu_pt": 0.0, 
            # monitoring losses (set weights to 0.0)
            "neg_r2": 0.0, 
            "w0_mass_mae": 0.0, "w1_mass_mae": 0.0,
            "aux_mom_mmd0": 0.0, "aux_mom_mmd1": 0.0,
        }
        self.loss_weights = {**defaults, **(loss_weights or {})}
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, x, y, y_pred):
        mmd_w0, mmd_w1 = aux_mom_mmd_loss(y, y_pred, self.current_epoch)
        w0_mass_mae, w1_mass_mae = w_mass_mae_losses(y, y_pred)
        w_mass_mmd0, w_mass_mmd1 = w_mass_mmd_losses(y, y_pred)
        losses = {
            "huber": huber_loss(y, y_pred),
            "w_mass_mmd0": w_mass_mmd0, "w_mass_mmd1": w_mass_mmd1,
            "higgs_mass": higgs_mass_loss(y_pred),
            "nu_mass": nu_mass_loss(x, y_pred),
            "dinu_pt": dinu_pt_loss(x, y_pred),
            # monitoring losses 
            "neg_r2": neg_r2_loss(y, y_pred),
            "w0_mass_mae": w0_mass_mae, "w1_mass_mae": w1_mass_mae,
            "aux_mom_mmd0": mmd_w0, "aux_mom_mmd1": mmd_w1,
        }

        total = sum(self.loss_weights[k] * v for k, v in losses.items())
        return total, losses

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)
            
    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        total, losses = self._compute_losses(x, y, self(x))
        self._log_losses(f"{stage}", losses, total)
        return total
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="")

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="val_")

    def test_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="test_")

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5.0e-4)
    
    def configure_optimizers(self):
            # AdamW optimizer with weight decay
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.lr, 
                weight_decay=0.001
            )
            
            # # Cosine annealing scheduler
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, 
            #     T_max=400, 
            #     eta_min=1e-6
            # )
            
            return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "interval": "epoch",
                # },
            }
