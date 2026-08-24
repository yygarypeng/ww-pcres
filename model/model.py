import math
from numbers import Integral

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from data.preprocessing import (
    BASE_INPUT_DIM,
    RAW_INPUT_DIM,
    neural_input_features_torch,
)
from model.layers import ResidualBlock, SelfAttentionBlock, Standardization, WBosonFourVectorLayer
from model.losses import (
    _valid_kinematic_rows,
    alpha_mmd,
    angular_mmd,
    dmet_loss,
    higgs_mass_loss,
    mass_mmd,
    standardized_fourvec_huber_loss,
    w_mass_huber_loss,
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
    config = {} if config is None else dict(config)
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

        if input_dim not in (BASE_INPUT_DIM, RAW_INPUT_DIM):
            raise ValueError(
                f"raw input contract requires input_dim={BASE_INPUT_DIM} or {RAW_INPUT_DIM}, "
                f"got {input_dim}; retraining required for incompatible checkpoints"
            )
        if len(std_mean_train) != input_dim or len(std_scale_train) != input_dim:
            raise ValueError(
                f"normalization statistics must each contain {input_dim} values; "
                "retraining required for incompatible checkpoints"
            )

        # do the normalization (need to use large batch size for stable stats)
        self.input_dim = input_dim
        self.norm = Standardization(std_mean_train, std_scale_train)
        self.base_input_dim = BASE_INPUT_DIM  # w/o high-level features
        self.hl_input_dim = input_dim - BASE_INPUT_DIM

        # Object-specific embeddings avoid forcing charge/order symmetry too early.
        self.lep0_embed = nn.Linear(4, d_model)
        self.lep1_embed = nn.Linear(4, d_model)
        self.jet0_embed = nn.Linear(4, d_model)
        self.jet1_embed = nn.Linear(4, d_model)
        self.met_embed = nn.Linear(2, d_model)
        self.hl_embed = nn.Linear(self.hl_input_dim, d_model) if self.hl_input_dim > 0 else None
        self.num_tokens = 5 + int(self.hl_embed is not None)
        if self.hl_embed is not None:
            print(f"Using {self.hl_input_dim} high-level features in the model.")
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

        # residual decoder blocks
        self.trunk = nn.Sequential(
            nn.Linear(d_model * self.num_tokens, 512),
            ResidualBlock(512, 512, hidden_dim=512, dropout=decoder_dropout),
            ResidualBlock(512, 256, hidden_dim=512, dropout=decoder_dropout),
            ResidualBlock(256, 256, hidden_dim=256, dropout=decoder_dropout),
            ResidualBlock(256, 128, hidden_dim=256, dropout=decoder_dropout),
            ResidualBlock(128, 128, hidden_dim=128, dropout=decoder_dropout),
        )
        # Latent regression head1 layout: [delta_dinu_px, delta_dinu_py, nu0_pz, nu1_pz]
        self.nu_mom_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        # Latent regression head2 layout: [nu0_pz, nu1_pz]
        # self.nu_long_head = nn.Sequential(
        #     nn.LayerNorm(128),
        #     nn.Linear(128, 32),
        #     nn.GELU(),
        #     nn.Linear(32, 2),
        # )
        # Latent regression head3 layout: [dmet_x, dmet_y]
        self.nu_dmet_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        # W bosons decoder
        self.w_layer = WBosonFourVectorLayer()

    def global_feature_aggregation(self, x):
        x_std = self.norm(neural_input_features_torch(x))

        # embedding to get initial context
        l0 = self.lep0_embed(x_std[:, 0:4])
        l1 = self.lep1_embed(x_std[:, 4:8])
        j0 = self.jet0_embed(x_std[:, 8:12])
        j1 = self.jet1_embed(x_std[:, 12:16])
        met = self.met_embed(x_std[:, 16:18])
        tokens = [l0, l1, j0, j1, met]
        if self.hl_embed is not None:
            tokens.append(
                self.hl_embed(
                    x_std[:, self.base_input_dim : self.base_input_dim + self.hl_input_dim]
                )
            )
        context = torch.stack(tokens, dim=1)

        # Key mask for empty jets
        batch_size = x.shape[0]
        key_mask = torch.zeros((batch_size, self.num_tokens), dtype=torch.bool, device=x.device)
        key_mask[:, 2] = x[:, 8:12].abs().sum(dim=1) == 0
        key_mask[:, 3] = x[:, 12:16].abs().sum(dim=1) == 0
        context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)

        for refiner in self.sa_blocks:
            context = refiner(context, key_padding_mask=key_mask)  # [B, num_tokens, d_model]
            context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)

        context = self.context_norm(context)
        context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
        return context.reshape(batch_size, -1)

    def forward(self, x, return_aux=False):
        lep0, lep1 = x[..., :4], x[..., 4:8]
        met = x[..., 16:18]

        h = self.global_feature_aggregation(x)
        h = self.trunk(h)
        nu_mom_params = self.nu_mom_head(h)
        dmet_params = self.nu_dmet_head(h)
        nu_params = torch.cat([nu_mom_params, dmet_params], dim=-1)

        y_pred = self.w_layer(lep0, lep1, nu_params, met)

        if return_aux:
            return y_pred, {
                "dmet": dmet_params,
                "nu_params": nu_params,
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
        w_fourvec_scales=None,
        dmet_scales=None,
        mass_mmd_center=0.0,
        mass_mmd_scale=1.0,
        lr=1e-4,
        weight_decay=1e-4,
        loss_weights=None,
        mmd_config=None,
        angular_mmd_ramp_epochs=0,
        adaptive_loss_weights=False,
        log_loss_gradient_cosines=False,
        higgs_mass_target=125.0,
        higgs_mass_scale=10.0,
        higgs_mass_delta=2.0,
        attention_blocks=4,
        attention_dropout=0.1,
        decoder_dropout=0.1,
    ):
        super().__init__()

        # resolve higgs parameters
        higgs_mass_parameters = {
            "higgs_mass_target": float(higgs_mass_target),
            "higgs_mass_scale": float(higgs_mass_scale),
            "higgs_mass_delta": float(higgs_mass_delta),
        }
        for name, value in higgs_mass_parameters.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        higgs_mass_target = higgs_mass_parameters["higgs_mass_target"]
        higgs_mass_scale = higgs_mass_parameters["higgs_mass_scale"]
        higgs_mass_delta = higgs_mass_parameters["higgs_mass_delta"]

        # resolve angular_mmd_ramp_epochs
        if isinstance(angular_mmd_ramp_epochs, bool):
            raise ValueError("angular_mmd_ramp_epochs must be a non-negative integer")
        if isinstance(angular_mmd_ramp_epochs, Integral):
            angular_mmd_ramp_epochs = int(angular_mmd_ramp_epochs)
            if angular_mmd_ramp_epochs < 0:
                raise ValueError("angular_mmd_ramp_epochs must be a non-negative integer")
        else:
            try:
                ramp_epochs = float(angular_mmd_ramp_epochs)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "angular_mmd_ramp_epochs must be a non-negative integer"
                ) from error
            if not math.isfinite(ramp_epochs) or ramp_epochs < 0.0 or not ramp_epochs.is_integer():
                raise ValueError("angular_mmd_ramp_epochs must be a non-negative integer")
            angular_mmd_ramp_epochs = int(ramp_epochs)
        # resolve mmd_config
        mmd_config = resolve_mmd_config(mmd_config)

        # save model configuration
        self.save_hyperparameters()

        # register buffers for loss scaling (store in stat-dict)
        if w_fourvec_scales is None:
            w_fourvec_scales = torch.ones(4, dtype=torch.float32)
        self.register_buffer(
            "w_fourvec_scales",
            torch.as_tensor(w_fourvec_scales, dtype=torch.float32).clamp_min(
                torch.finfo(torch.float32).eps
            ),
        )
        if dmet_scales is None:
            dmet_scales = torch.ones(2, dtype=torch.float32)
        self.register_buffer(
            "dmet_scales",
            torch.as_tensor(dmet_scales, dtype=torch.float32).clamp_min(
                torch.finfo(torch.float32).eps
            ),
        )
        self.register_buffer(
            "mass_mmd_center",
            torch.as_tensor(mass_mmd_center, dtype=torch.float32),
        )
        self.register_buffer(
            "mass_mmd_scale",
            torch.as_tensor(mass_mmd_scale, dtype=torch.float32).clamp_min(
                torch.finfo(torch.float32).eps
            ),
        )

        # load model
        self.model = WBosonRegressor(
            input_dim,
            d_model,
            num_heads,
            std_mean_train,
            std_scale_train,
            attention_blocks=attention_blocks,
            attention_dropout=attention_dropout,
            decoder_dropout=decoder_dropout,
        )  # give a base model structure for forward()
        defaults = {
            # main loss
            "huber": 1.0,
            # mass losses
            "higgs_mass": 0.0,
            "w_mass_huber": 0.0,
            # met loss
            "dmet": 0.0,
            # auxiliary losses
            "alpha_mmd": 0.0,
            "mass_mmd": 0.0,
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
        # log every loss gradient cosine (if 0 -> do not log)
        self.adaptive_loss_names = [
            name for name, weight in self.loss_weights.items() if weight != 0.0
        ]
        self.log_loss_gradient_cosines = bool(log_loss_gradient_cosines)
        self._gradient_analysis_batch = None
        self.mmd_config = mmd_config
        self.angular_mmd_ramp_epochs = angular_mmd_ramp_epochs
        self.higgs_mass_target = higgs_mass_target
        self.higgs_mass_scale = higgs_mass_scale
        self.higgs_mass_delta = higgs_mass_delta
        self.lr = lr

    def forward(self, x, return_aux=False):
        return self.model(x, return_aux=return_aux)

    def _effective_loss_weights(self):
        if self.angular_mmd_ramp_epochs == 0:
            return self.loss_weights

        # consine ramp for angular_mmd loss weight
        progress = min(max(self.current_epoch / self.angular_mmd_ramp_epochs, 0.0), 1.0)
        multiplier = (1.0 - math.cos(math.pi * progress)) / 2.0
        return {
            **self.loss_weights,
            "angular_mmd": self.loss_weights["angular_mmd"] * multiplier,
        }

    def _loss_enabled(self, name, weights=None):
        weights = self._effective_loss_weights() if weights is None else weights
        return weights.get(name, 0.0) != 0.0 or (
            self.adaptive_loss_weights and name in self.adaptive_loss_names
        )

    def _mmd_kwargs(self, feature_name):
        feature = self.mmd_config[feature_name]
        return {"kernel": feature["kernel"], "bandwidths": feature["bandwidths"]}

    def _compute_losses(self, x, y, y_pred, cond, aux=None):
        losses = {}
        weights = self._effective_loss_weights()
        if self._loss_enabled("huber", weights):
            losses["huber"] = standardized_fourvec_huber_loss(
                y,
                y_pred,
                self.w_fourvec_scales,
            )
        if self._loss_enabled("higgs_mass", weights):
            losses["higgs_mass"] = higgs_mass_loss(
                y_pred,
                target_mass=self.higgs_mass_target,
                scale=self.higgs_mass_scale,
                delta=self.higgs_mass_delta,
            )
        if self._loss_enabled("w_mass_huber", weights):
            losses["w_mass_huber"] = w_mass_huber_loss(y, y_pred)
        kinematic_valid = (
            _valid_kinematic_rows(x, y, y_pred)
            if (self._loss_enabled("alpha_mmd", weights) or self._loss_enabled("mass_mmd", weights))
            else None
        )
        if self._loss_enabled("alpha_mmd", weights):
            losses["alpha_mmd"] = alpha_mmd(
                x,
                y,
                y_pred,
                cond,
                valid_mask=kinematic_valid,
                **self._mmd_kwargs("alpha"),
            )
        if self._loss_enabled("mass_mmd", weights):
            losses["mass_mmd"] = mass_mmd(
                x,
                y,
                y_pred,
                cond,
                self.mass_mmd_center,
                self.mass_mmd_scale,
                valid_mask=kinematic_valid,
                **self._mmd_kwargs("mass"),
            )
        if self._loss_enabled("angular_mmd", weights):
            losses["angular_mmd"] = angular_mmd(
                x,
                y,
                y_pred,
                cond,
                **self._mmd_kwargs("angular"),
            )
        if self._loss_enabled("dmet", weights):
            if aux is None or "dmet" not in aux:
                raise ValueError("dmet loss requires forward(..., return_aux=True) outputs")
            losses["dmet"] = dmet_loss(x, y, aux["dmet"], self.dmet_scales)

        total = self._weighted_total_loss(losses, weights)
        return total, losses

    def _compute_batch_losses(self, x, y):
        y_pred, aux = self(x, return_aux=True)
        return self._compute_losses(x, y, y_pred, None, aux)

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
            allow_unused=True,  # allow loss only use parts of output (parameters)
        )
        return parameters_to_vector(
            torch.zeros_like(param) if grad is None else grad.detach()
            for param, grad in zip(parameters, grads)
        )

    def _compute_loss_gradient_cosines(self, losses, total):
        names = [name for name in self.adaptive_loss_names if name in losses]
        if not names:
            return {}

        # Only include trainable parameters in the gradient comparison.
        parameters = tuple(p for p in self.parameters() if p.requires_grad)
        if not parameters:
            return {}

        total_grad = self._loss_grad_vector(total, parameters)

        effective_weights = self._effective_loss_weights()
        cosines = {}
        for name in names:
            grad = self._loss_grad_vector(losses[name], parameters)
            cos_total = torch.nn.functional.cosine_similarity(grad, total_grad, dim=0, eps=1.0e-6)
            weight = effective_weights.get(name, 0.0)
            rest_grad = total_grad - weight * grad
            cos_rest = torch.nn.functional.cosine_similarity(grad, rest_grad, dim=0, eps=1.0e-6)
            if not torch.isfinite(cos_total).item() or not torch.isfinite(cos_rest).item():
                return {}
            cosines[name] = {
                "total": cos_total.detach(),
                "rest": cos_rest.detach(),
            }
        return cosines

    def _update_adaptive_loss_weights(self, cosines):
        if not cosines:
            return

        names = list(cosines)
        adaptive_loss_budget = sum(self.loss_weights.get(name, 0.0) for name in names)
        if adaptive_loss_budget <= 0.0:
            return
        raw_weights = [torch.clamp(1.0 - cosines[name]["total"], min=0.0) for name in names]

        raw_sum = torch.stack(raw_weights).sum()
        if not torch.isfinite(raw_sum).item() or raw_sum.item() <= 0.0:
            return

        for name, raw in zip(names, raw_weights):
            target = float((raw / raw_sum * adaptive_loss_budget).detach().cpu())
            old = float(self.loss_weights.get(name, target))

            # todo: smooth update instead of hard assignment.
            rho = 0.05  # 0.01 to 0.10. Smaller = more stable.
            new = (1.0 - rho) * old + rho * target

            self.loss_weights[name] = new

    def _log_losses(self, prefix, losses, total):
        self.log(f"{prefix}loss", total.detach(), prog_bar=True, on_step=False, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{prefix}{k}_loss", v.detach(), prog_bar=False, on_step=False, on_epoch=True)

    def _log_loss_weights(self):
        for name, weight in self._effective_loss_weights().items():
            self.log(f"loss_weight/{name}", weight, prog_bar=False, on_step=True, on_epoch=False)

    def _log_grad_cosines(self, cosines):
        if not self.log_loss_gradient_cosines:
            return
        for name, cos in cosines.items():
            self.log(
                f"grad_cos/{name}__total",
                cos["total"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"grad_cos/{name}__rest",
                cos["rest"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

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
            cosines = self._compute_loss_gradient_cosines(losses, total)
        if self.adaptive_loss_weights:
            self._update_adaptive_loss_weights(cosines)
        self._log_grad_cosines(cosines)
        self._gradient_analysis_batch = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        total, losses = self._compute_batch_losses(x, y)
        needs_gradient_analysis = self.adaptive_loss_weights or self.log_loss_gradient_cosines
        if needs_gradient_analysis and self._gradient_analysis_batch is None:
            self._gradient_analysis_batch = (x.detach().cpu(), y.detach().cpu())
        self._log_losses("", losses, total)
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
