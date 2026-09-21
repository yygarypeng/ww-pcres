import math

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from data.preprocessing import BASE_INPUT_DIM, neural_input_features_torch
from model.layers import ResidualBlock, SelfAttentionBlock, Standardization, WConstraintsLayer
from model.losses import (
    H_MASS_SCALE,
    _angular_mmd_with_valid_mask,
    _valid_kinematic_rows,
    alpha_mmd,
    dmet_loss,
    higgs_fourvec_loss,
    higgs_mass_loss,
    w_fourvec_loss,
    w_mass_loss,
    w_mass_mmd,
)

DEFAULT_MMD_CONFIG = {
    "alpha": {
        "kernel": "imq",
        "bandwidths": [0.05, 0.5, 5.0],
    },
    "mass": {
        "kernel": "imq",
        "bandwidths": [0.05, 0.5, 5.0],
    },
    "angular": {
        "kernel": "imq",
        "bandwidths": [0.05, 0.5, 5.0],
    },
}


def resolve_mmd_config(config=None):
    config = {} if config is None else config
    unknown_sections = set(config) - set(DEFAULT_MMD_CONFIG)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unsupported MMD config section(s): {names}")

    resolved = {}
    for section, defaults in DEFAULT_MMD_CONFIG.items():
        supplied = config.get(section, {})
        unknown_keys = set(supplied) - set(defaults)
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unsupported MMD config key(s) in {section}: {names}")
        resolved_section = {**defaults, **supplied}
        if resolved_section["kernel"] not in {"rbf", "imq"}:
            raise ValueError(f"unsupported MMD kernel in {section}: {resolved_section['kernel']}")
        bandwidths = resolved_section["bandwidths"]
        if not bandwidths or not all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in bandwidths
        ):
            raise ValueError(f"MMD bandwidths in {section} must be finite and positive")
        resolved[section] = resolved_section
    return resolved


class WBosonRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        std_mean_train,
        std_scale_train,
        attention_blocks=4,
        attention_dropout=0.1,
        decoder_dropout=0.1,
    ):
        super().__init__()

        if input_dim != BASE_INPUT_DIM:
            raise ValueError(
                f"raw input contract requires input_dim={BASE_INPUT_DIM}, "
                f"got {input_dim}; retraining required for incompatible checkpoints"
            )
        if len(std_mean_train) != input_dim or len(std_scale_train) != input_dim:
            raise ValueError(
                f"normalization statistics must each contain {input_dim} values; "
                "retraining required for incompatible checkpoints"
            )

        self.norm = Standardization(std_mean_train, std_scale_train)

        # Object-specific embeddings avoid forcing charge/order symmetry too early.
        self.lep0_embed = nn.Linear(4, d_model)
        self.lep1_embed = nn.Linear(4, d_model)
        self.jet0_embed = nn.Linear(4, d_model)
        self.jet1_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        self.num_tokens = 5
        self.sa_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d_model, num_heads, dropout=attention_dropout)
                for _ in range(attention_blocks)
            ]
        )
        self.context_norm = nn.LayerNorm(d_model)
        print(
            f"Using {len(self.sa_blocks)} SA blocks; connected context dimension: {d_model * self.num_tokens}"
        )

        self.trunk = nn.Sequential(
            nn.Linear(d_model * self.num_tokens, 512),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            ResidualBlock(512, 256, hidden_dim=256, dropout=decoder_dropout),
            ResidualBlock(256, 256, hidden_dim=256, dropout=decoder_dropout),
            ResidualBlock(256, 128, hidden_dim=128, dropout=decoder_dropout),
            ResidualBlock(128, 128, hidden_dim=128, dropout=decoder_dropout),
        )

        # Latent regression layout: [nu0_px, nu0_py, nu0_pz, nu1_px, nu1_py, nu1_pz]
        self.nu_mom = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 6, bias=False),
        )

        self.w_layer = WConstraintsLayer()

    def global_feature_aggregation(self, x):
        x_std = self.norm(neural_input_features_torch(x))

        l0 = self.lep0_embed(x_std[:, 0:4])
        l1 = self.lep1_embed(x_std[:, 4:8])
        j0 = self.jet0_embed(x_std[:, 8:12])
        j1 = self.jet1_embed(x_std[:, 12:16])
        met = self.met_embed(x_std[:, 16:18])
        context = torch.stack([l0, l1, j0, j1, met], dim=1)

        # Key mask for empty jets
        batch_size = x.shape[0]
        key_mask = torch.zeros((batch_size, self.num_tokens), dtype=torch.bool, device=x.device)
        key_mask[:, 2] = x[:, 8:12].abs().sum(dim=1) == 0
        key_mask[:, 3] = x[:, 12:16].abs().sum(dim=1) == 0

        for refiner in self.sa_blocks:
            context = refiner(context, key_padding_mask=key_mask)  # [B, num_tokens, d_model]

        context = self.context_norm(context)
        context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
        return context.reshape(batch_size, -1)

    def forward(self, x, return_aux=False):
        lep0, lep1 = x[..., :4], x[..., 4:8]
        met = x[..., 16:18]

        h = self.global_feature_aggregation(x)
        h = self.trunk(h)
        nu_params = self.nu_mom(h)

        y_pred = self.w_layer(lep0, lep1, nu_params)

        if return_aux:
            dinu_pt = (y_pred[..., :2] - lep0[..., :2]) + (y_pred[..., 4:6] - lep1[..., :2])
            return y_pred, {
                "dmet": met - dinu_pt,
            }
        return y_pred


class LightningWBoson(L.LightningModule):
    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        std_mean_train,
        std_scale_train,
        lr=1e-4,
        weight_decay=1e-4,
        loss_weights=None,
        mmd_config=None,
        adaptive_loss_weights=False,
        log_loss_gradient_cosines=False,
        higgs_mass_target=H_MASS_SCALE,
        attention_blocks=4,
        attention_dropout=0.1,
        decoder_dropout=0.1,
        lr_plateau_factor=1.0,
        lr_plateau_patience=8,
    ):
        super().__init__()

        higgs_mass_target = float(higgs_mass_target)
        if higgs_mass_target != H_MASS_SCALE:
            raise ValueError(f"higgs_mass_target is fixed at {H_MASS_SCALE:g} GeV")

        mmd_config = resolve_mmd_config(mmd_config)

        self.save_hyperparameters()

        self.model = WBosonRegressor(
            input_dim,
            d_model,
            num_heads,
            std_mean_train,
            std_scale_train,
            attention_blocks=attention_blocks,
            attention_dropout=attention_dropout,
            decoder_dropout=decoder_dropout,
        )
        defaults = {
            # main loss
            "w_fourvec": 1.0,
            "higgs_fourvec": 0.0,
            # mass losses
            "higgs_mass": 0.0,
            "w_mass": 0.0,
            # met loss
            "dmet": 0.0,
            # auxiliary losses
            "alpha_mmd": 0.0,
            "w_mass_mmd": 0.0,
            "angular_mmd": 0.0,
        }
        unsupported = set(loss_weights or {}) - defaults.keys()
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported loss_weights key(s): {names}")
        self.loss_weights = {
            name: float(weight) for name, weight in {**defaults, **(loss_weights or {})}.items()
        }
        self.adaptive_loss_weights = bool(adaptive_loss_weights)
        self.adaptive_loss_names = [
            name for name, weight in self.loss_weights.items() if weight != 0.0
        ]
        self.log_loss_gradient_cosines = bool(log_loss_gradient_cosines)
        self._gradient_analysis_batch = None
        self.mmd_config = mmd_config
        self.higgs_mass_target = higgs_mass_target
        self.lr_plateau_factor = float(lr_plateau_factor)
        self.lr_plateau_patience = int(lr_plateau_patience)

    def forward(self, x, return_aux=False):
        return self.model(x, return_aux=return_aux)

    def _loss_enabled(self, name):
        return self.loss_weights.get(name, 0.0) != 0.0 or (
            self.adaptive_loss_weights and name in self.adaptive_loss_names
        )

    def _mmd_kwargs(self, feature_name):
        feature = self.mmd_config[feature_name]
        return {"kernel": feature["kernel"], "bandwidths": feature["bandwidths"]}

    def _compute_losses(self, x, y, y_pred, aux=None):
        losses = {}
        if self._loss_enabled("w_fourvec"):
            losses["w_fourvec"] = w_fourvec_loss(y, y_pred)
        if self._loss_enabled("higgs_fourvec"):
            losses["higgs_fourvec"] = higgs_fourvec_loss(y, y_pred)
        if self._loss_enabled("higgs_mass"):
            losses["higgs_mass"] = higgs_mass_loss(y_pred)
        if self._loss_enabled("w_mass"):
            losses["w_mass"] = w_mass_loss(y, y_pred)
        kinematic_valid = (
            _valid_kinematic_rows(x, y, y_pred)
            if (
                self._loss_enabled("alpha_mmd")
                or self._loss_enabled("w_mass_mmd")
                or self._loss_enabled("angular_mmd")
            )
            else None
        )
        if self._loss_enabled("alpha_mmd"):
            losses["alpha_mmd"] = alpha_mmd(
                x,
                y,
                y_pred,
                valid_mask=kinematic_valid,
                **self._mmd_kwargs("alpha"),
            )
        if self._loss_enabled("w_mass_mmd"):
            losses["w_mass_mmd"] = w_mass_mmd(
                x,
                y,
                y_pred,
                valid_mask=kinematic_valid,
                **self._mmd_kwargs("mass"),
            )
        if self._loss_enabled("angular_mmd"):
            losses["angular_mmd"] = _angular_mmd_with_valid_mask(
                x,
                y,
                y_pred,
                valid_mask=kinematic_valid,
                **self._mmd_kwargs("angular"),
            )
        if self._loss_enabled("dmet"):
            if aux is None or "dmet" not in aux:
                raise ValueError("dmet loss requires forward(..., return_aux=True) outputs")
            losses["dmet"] = dmet_loss(x, y, aux["dmet"])

        total = self._weighted_total_loss(losses)
        return total, losses

    def _compute_batch_losses(self, x, y):
        y_pred, aux = self(x, return_aux=True)
        return self._compute_losses(x, y, y_pred, aux)

    def _weighted_total_loss(self, losses):
        total = None
        for name, loss in losses.items():
            weight = self.loss_weights.get(name, 0.0)
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
            allow_unused=True,  # allow loss only use parts of output (parameters)
        )
        return parameters_to_vector(
            torch.zeros_like(param) if grad is None else grad.detach()
            for param, grad in zip(parameters, grads)
        )

    def _compute_loss_gradient_stats(self, losses, total):
        names = [name for name in self.adaptive_loss_names if name in losses]
        if not names:
            return {}

        # Only include trainable parameters in the gradient comparison.
        parameters = tuple(p for p in self.parameters() if p.requires_grad)
        if not parameters:
            return {}

        total_grad = self._loss_grad_vector(total, parameters)

        stats = {}
        for name in names:
            grad = self._loss_grad_vector(losses[name], parameters)
            cos_total = torch.nn.functional.cosine_similarity(grad, total_grad, dim=0, eps=1.0e-6)
            weight = self.loss_weights.get(name, 0.0)
            rest_grad = total_grad - weight * grad
            cos_rest = torch.nn.functional.cosine_similarity(grad, rest_grad, dim=0, eps=1.0e-6)
            if not torch.isfinite(cos_total).item() or not torch.isfinite(cos_rest).item():
                return {}
            stats[name] = {
                "total": cos_total.detach(),
                "rest": cos_rest.detach(),
                "norm": (weight * grad).norm().detach(),
            }
        return stats

    def _update_adaptive_loss_weights(self, stats):
        if not stats:
            return

        names = list(stats)
        adaptive_loss_budget = sum(self.loss_weights.get(name, 0.0) for name in names)
        if adaptive_loss_budget <= 0.0:
            return
        raw_weights = [torch.clamp(1.0 - stats[name]["rest"], min=0.0) for name in names]

        raw_sum = torch.stack(raw_weights).sum()
        if not torch.isfinite(raw_sum).item() or raw_sum.item() <= 0.0:
            return

        for name, raw in zip(names, raw_weights):
            target = float((raw / raw_sum * adaptive_loss_budget).detach().cpu())
            old = float(self.loss_weights.get(name, target))

            # todo: smooth update instead of hard assignment.
            rho = 0.001  # 0.01 to 0.10. Smaller = more stable.
            new = (1.0 - rho) * old + rho * target

            self.loss_weights[name] = new

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)

    def _log_loss_weights(self):
        on_step = self.adaptive_loss_weights
        for name, weight in self.loss_weights.items():
            self.log(
                f"loss_weight/{name}",
                weight,
                prog_bar=False,
                on_step=on_step,
                on_epoch=not on_step,
            )

    def _log_grad_stats(self, stats):
        if not self.log_loss_gradient_cosines:
            return
        for name, stat in stats.items():
            for metric, value in (
                (f"grad_cos/{name}__total", stat["total"]),
                (f"grad_cos/{name}__rest", stat["rest"]),
                (f"grad_norm/{name}", stat["norm"]),
            ):
                self.log(metric, value, prog_bar=False, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        self._gradient_analysis_batch = None

    def on_train_epoch_end(self):
        if self._gradient_analysis_batch is None:
            return

        x, y = self._gradient_analysis_batch
        with torch.enable_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            total, losses = self._compute_batch_losses(x, y)
            stats = self._compute_loss_gradient_stats(losses, total)
        if self.adaptive_loss_weights:
            self._update_adaptive_loss_weights(stats)
        self._log_grad_stats(stats)
        self._gradient_analysis_batch = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        total, losses = self._compute_batch_losses(x, y)
        needs_gradient_analysis = self.adaptive_loss_weights or self.log_loss_gradient_cosines
        if needs_gradient_analysis and self._gradient_analysis_batch is None:
            self._gradient_analysis_batch = (x.detach().cpu(), y.detach().cpu())
        self._log_losses("", losses, total)
        if self.adaptive_loss_weights or batch_idx == 0:
            self._log_loss_weights()
        return total

    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        total, losses = self._compute_batch_losses(x, y)
        self._log_losses(f"{stage}", losses, total)
        return total

    def validation_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="val_")

    def test_step(self, batch, batch_idx):
        _ = self._shared_step(batch, batch_idx, stage="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            fused=torch.cuda.is_available(),
        )
        if self.lr_plateau_factor >= 1.0:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.lr_plateau_factor,
                    patience=self.lr_plateau_patience,
                ),
                "monitor": "val_loss",
            },
        }
