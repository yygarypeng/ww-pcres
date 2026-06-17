import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import pytorch_lightning as L

from model.layers import Standardization, SelfAttentionBlock, ResidualBlock, WBosonFourVectorLayer
from model.losses import (
    huber_loss, w_mass_mmd_losses, higgs_mass_loss, 
    neg_r2_loss, nu_mass_loss, dinu_pt_loss, w_mass_mae_losses, aux_mom_mmd_loss,
    angular_loss_mmd, angular_loss_mae
)

class WBosonRegressor(nn.Module):
    def __init__(
            self, 
            input_dim,
            d_model, num_heads,
            std_mean_train, std_scale_train
        ):
        super().__init__()
        
        # do the normalization (need to use large batch size for stable stats)
        self.norm = Standardization(std_mean_train, std_scale_train)
        self.base_input_dim = 18 # w/o high-level features
        self.hl_input_dim = input_dim - self.base_input_dim
        if self.hl_input_dim < 0:
            raise ValueError(f"input_dim must be at least {self.base_input_dim}, got {input_dim}")
        
        # Object-specific embeddings avoid forcing charge/order symmetry too early.
        self.lep0_embed = nn.Linear(4, d_model)
        self.lep1_embed = nn.Linear(4, d_model)
        self.jet0_embed = nn.Linear(4, d_model)
        self.jet1_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        self.hl_embed = nn.Linear(self.hl_input_dim, d_model) if self.hl_input_dim > 0 else None
        self.num_tokens = 5 + int(self.hl_embed is not None)
        # self.role_embedding = nn.Parameter(torch.zeros(self.num_tokens, d_model))
        # nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, dropout=0.5) for _ in range(8)
        ])
        print(f"Using {len(self.sa_blocks)} SA blocks.")
        
        # residual decoder blocks
        _dim = 256 if d_model * self.num_tokens >= 256 else d_model * self.num_tokens
        blocks = [nn.Linear(d_model * self.num_tokens, _dim)] # reduce dimension after flattening
        blocks.append(ResidualBlock(_dim, 256, dropout=0.5))
        blocks.append(ResidualBlock(256, 256, dropout=0.5))
        blocks.append(ResidualBlock(256, 128, dropout=0.5))
        blocks.append(ResidualBlock(128, 128, dropout=0.5))
        # blocks.append(ResidualBlock(128, 64, dropout=0.5))
        
        self.trunk = nn.Sequential(*blocks)
        self.pre_trunk_bn = nn.LayerNorm(d_model * self.num_tokens)

        # nu momentum regression head
        self.nu_mom_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 6)
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

        # embedding to get initial context
        l0 = self.lep0_embed(x_std[:, 0:4])
        l1 = self.lep1_embed(x_std[:, 4:8])
        j0 = self.jet0_embed(x_std[:, 8:12])
        j1 = self.jet1_embed(x_std[:, 12:16])
        met = self.met_embed(x_std[:, 16:18])
        tokens = [l0, l1, j0, j1, met]
        if self.hl_embed is not None:
            tokens.append(self.hl_embed(x_std[:, self.base_input_dim:]))
        
        context = torch.stack(tokens, dim=1)
        # context = context + self.role_embedding.unsqueeze(0)
        context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
        
        for refiner in self.sa_blocks:
            context = refiner(context, key_padding_mask=key_mask) # [B, num_tokens, d_model]
            context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)

        return context.reshape(batch_size, -1) # flatten to [B, num_tokens * d_model]

    def predict_nu_mom(self, x):
        h = self.global_feature_aggregation(x)
        h = self.pre_trunk_bn(h) # change sttn domain to res domain
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
            d_model, num_heads,
            std_mean_train, std_scale_train, 
            lr=1e-4, loss_weights=None,
            adaptive_loss_weights=False,
            log_loss_gradient_cosines=False,
            **legacy_hparams,
        ):
        super().__init__()
        legacy_hparams.pop("adaptive_loss_update_interval", None)
        if legacy_hparams:
            raise TypeError(f"Unexpected hyperparameters: {sorted(legacy_hparams)}")
        self.save_hyperparameters(ignore=["legacy_hparams"])
        self.model = WBosonRegressor(
            input_dim, 
            d_model, num_heads,
            std_mean_train, std_scale_train,
        ) # give a base model structure for forward() 
        defaults = {
            # main loss
            "huber": 1.0, 
            # mass losses
            "higgs_mass": 0.0,
            "w_mass_mmd0": 0.0, "w_mass_mmd1": 0.0,
            "w0_mass_mae": 0.0, "w1_mass_mae": 0.0,
            # targets correction losses
            "angular_loss_mmd": 0.0, 
            "angular_loss_mae": 0.0,
            # monitoring losses (set weights to 0.0)
            "neg_r2": 0.0, 
            "dinu_pt": 0.0, 
            # "nu_mass": 0.0, 
            # "aux_mom_mmd0": 0.0, "aux_mom_mmd1": 0.0,
        }
        self.loss_weights = {
            name: float(weight)
            for name, weight in {**defaults, **(loss_weights or {})}.items()
        }
        self.adaptive_loss_weights = bool(adaptive_loss_weights)
        self.adaptive_loss_names = [
            name for name, weight in self.loss_weights.items()
            if weight != 0.0
        ]
        self.adaptive_loss_budget = sum(
            self.loss_weights[name] for name in self.adaptive_loss_names
        )
        self.log_loss_gradient_cosines = bool(log_loss_gradient_cosines)
        self._adaptive_batch = None
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _loss_enabled(self, name):
        return self.loss_weights.get(name, 0.0) != 0.0 or name in self.adaptive_loss_names

    def _compute_losses(self, x, y, y_pred):
        losses = {}
        if self._loss_enabled("huber"):
            losses["huber"] = huber_loss(y, y_pred)
        if self._loss_enabled("higgs_mass"):
            losses["higgs_mass"] = higgs_mass_loss(y_pred)
        if self._loss_enabled("w_mass_mmd0") or self._loss_enabled("w_mass_mmd1"):
            w_mass_mmd0, w_mass_mmd1 = w_mass_mmd_losses(y, y_pred)
            losses["w_mass_mmd0"] = w_mass_mmd0
            losses["w_mass_mmd1"] = w_mass_mmd1
        if self._loss_enabled("w0_mass_mae") or self._loss_enabled("w1_mass_mae"):
            w0_mass_mae, w1_mass_mae = w_mass_mae_losses(y, y_pred)
            losses["w0_mass_mae"] = w0_mass_mae
            losses["w1_mass_mae"] = w1_mass_mae
        if self._loss_enabled("angular_loss_mmd"):
            losses["angular_loss_mmd"] = angular_loss_mmd(x, y, y_pred)
        if self._loss_enabled("angular_loss_mae"):
            losses["angular_loss_mae"] = angular_loss_mae(x, y, y_pred)
        if self._loss_enabled("neg_r2"):
            losses["neg_r2"] = neg_r2_loss(y, y_pred)
        if self._loss_enabled("dinu_pt"):
            losses["dinu_pt"] = dinu_pt_loss(x, y_pred)
        if self._loss_enabled("nu_mass"):
            losses["nu_mass"] = nu_mass_loss(x, y_pred)
        if self._loss_enabled("aux_mom_mmd0") or self._loss_enabled("aux_mom_mmd1"):
            aux_mom_mmd0, aux_mom_mmd1 = aux_mom_mmd_loss(y, y_pred)
            losses["aux_mom_mmd0"] = aux_mom_mmd0
            losses["aux_mom_mmd1"] = aux_mom_mmd1

        total = self._weighted_total_loss(losses)
        return total, losses

    def _weighted_total_loss(self, losses, weights=None):
        weights = self.loss_weights if weights is None else weights
        total = None
        for name, loss in losses.items():
            weight = weights.get(name, 0.0)
            if weight == 0.0:
                continue
            term = weight * loss
            total = term if total is None else total + term
        if total is None:
            return next(iter(losses.values())) * 0.0
        return total

    def _loss_grad_vector(self, loss, parameters):
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True, # allow loss only use parts of output (parameters)
        )
        return parameters_to_vector(
            torch.zeros_like(param) if grad is None else grad.detach()
            for param, grad in zip(parameters, grads)
        )

    def _update_adaptive_loss_weights(self, losses, total):
        names = [name for name in self.adaptive_loss_names if name in losses]
        if not names:
            return {}
        if self.adaptive_loss_budget <= 0.0:
            return {}

        # only allow trainble parameters
        parameters = tuple(p for p in self.parameters() if p.requires_grad)
        if not parameters:
            return {}

        total_grad = self._loss_grad_vector(total, parameters)

        raw_weights = []
        cosines = {}
        for name in names:
            grad = self._loss_grad_vector(losses[name], parameters)
            cos = torch.nn.functional.cosine_similarity(grad, total_grad, dim=0, eps=1.0e-6)
            raw = torch.clamp(1.0 - cos, min=0.0)
            if not torch.isfinite(raw).item():
                return {}
            raw_weights.append(raw)
            cosines[name] = cos.detach()

        raw_sum = torch.stack(raw_weights).sum()
        if not torch.isfinite(raw_sum).item() or raw_sum.item() <= 0.0:
            return {}

        for name, raw in zip(names, raw_weights):
            # update/scale weights by normalized cosine similarity
            self.loss_weights[name] = float((raw / raw_sum * self.adaptive_loss_budget).detach().cpu())
        return cosines

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)

    def _log_loss_weights(self):
        for name, weight in self.loss_weights.items():
            self.log(f"loss_weight/{name}", weight, prog_bar=False, on_step=True, on_epoch=False)

    def _log_grad_cosines(self, cosines):
        if not self.log_loss_gradient_cosines:
            return
        for name, cos in cosines.items():
            self.log(f"grad_cos/{name}__total", cos, prog_bar=False, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        self._adaptive_batch = None

    def on_train_epoch_end(self):
        if not self.adaptive_loss_weights or self._adaptive_batch is None:
            return

        x, y = self._adaptive_batch
        with torch.enable_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            total, losses = self._compute_losses(x, y, self(x))
            cosines = self._update_adaptive_loss_weights(losses, total)
        self._log_grad_cosines(cosines)
        self._adaptive_batch = None
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        total, losses = self._compute_losses(x, y, self(x))
        if self.adaptive_loss_weights and self._adaptive_batch is None:
            self._adaptive_batch = (x.detach().cpu(), y.detach().cpu())
        self._log_losses("", losses, total)
        self._log_loss_weights()
        return total

    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        total, losses = self._compute_losses(x, y, self(x))
        self._log_losses(f"{stage}", losses, total)
        return total
    
    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="val_")

    def test_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="test_")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
