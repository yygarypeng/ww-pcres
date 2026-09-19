import torch
import torch.nn as nn

EPS = 1.0e-8

class Standardization(nn.Module):
    def __init__(self, mean, std, eps=EPS):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(eps))

    def forward(self, x):
        return (x - self.mean) / self.std


class _AttnFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.3):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = _AttnFFN(d_model, d_model * 4, dropout)

    def forward(self, x, key_padding_mask=None):
        res = x
        x = self.attn_norm(x)
        x, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = res + self.dropout(x)

        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = res + self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.shortcut = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        y = self.residual(x)
        return x_shortcut + y


class WBosonFourVectorLayer(nn.Module):
    def forward(self, lep0, lep1, nu_params, met):
        # [delta_dinu_px, delta_dinu_py, nu0_pz, nu1_pz, dmet_px, dmet_py].
        delta_dinu_pt = nu_params[..., :2]
        nu0_pz = nu_params[..., 2:3]
        nu1_pz = nu_params[..., 3:4]
        dmet = nu_params[..., 4:6]

        total_dinu_pt = met - dmet
        nu0_pt = 0.5 * (total_dinu_pt + delta_dinu_pt)
        nu1_pt = 0.5 * (total_dinu_pt - delta_dinu_pt)
        nu0_3 = torch.cat([nu0_pt, nu0_pz], dim=-1)
        nu1_3 = torch.cat([nu1_pt, nu1_pz], dim=-1)

        # neutrino energies as |p| for (approx) massless
        nu0_E = torch.linalg.vector_norm(nu0_3, dim=-1, keepdim=True)
        nu1_E = torch.linalg.vector_norm(nu1_3, dim=-1, keepdim=True)
        w0_3 = lep0[..., :3] + nu0_3
        w1_3 = lep1[..., :3] + nu1_3
        w0_E = lep0[..., 3:4] + nu0_E
        w1_E = lep1[..., 3:4] + nu1_E
        return torch.cat([w0_3, w0_E, w1_3, w1_E], dim=-1)


class WConstraintsLayer(nn.Module):

    @staticmethod
    def _dot(a, b):
        return a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(dim=-1, keepdim=True)

    @staticmethod
    def _massless(momentum):
        energy = torch.linalg.vector_norm(momentum, dim=-1, keepdim=True)
        return torch.cat([momentum, energy], dim=-1)

    def higgs_scale(self, dilep, dinu):
        """
        s is scale factor for SM Higgs mass shell constraint. The constraint is:
        (leplep + s * nunu)^2 = leplep_dot + 2 * s * lepnu_dot + s^2 * nunu_dot = mH^2 = 125.0^2
        Therefore, s = (-2 * lepnu_dot +- sqrt(4 * lepnu_dot^2 - 4 * nunu_dot * (leplep_dot - mH^2))) / (2 * nunu_dot)
        
        Drop negative solution since it is unphysical (it'll cause negative Enu). The positive solution is:
        s = (-lepnu_dot + sqrt(lepnu_dot^2 + nunu_dot * (125.0^2 - leplep_dot))) / (nunu_dot)
        
        To stablize the solutions, avoiding cancellations and physical possible 0 nunu_dot ( thanks to Claude :) ):
        s = (-lepnu_dot + sqrt(lepnu_dot^2 + nunu_dot * delta)) / (nunu_dot)
          = delta / (lepnu_dot + sqrt(lepnu_dot^2 + nunu_dot * delta))
        """
        
        lepnu_dot = self._dot(dilep, dinu)
        nunu_dot = self._dot(dinu, dinu).clamp_min(0.0)
        leplep_dot = self._dot(dilep, dilep).clamp_min(0.0)
        
        delta = (125.0**2 - leplep_dot)# MUST ensure that the mll <= mH physically for the inputs (data.py)
        denominator = (lepnu_dot + torch.sqrt(lepnu_dot * lepnu_dot + nunu_dot * delta)).clamp_min(EPS)
        return delta / denominator

    def forward(self, lep0, lep1, nu_params):
        nu0 = self._massless(nu_params[..., 0:3])
        nu1 = self._massless(nu_params[..., 3:6])

        _scale = self.higgs_scale(lep0[..., :4] + lep1[..., :4], nu0 + nu1)
        w_fourvecs = torch.cat([lep0[..., :4] + _scale * nu0, lep1[..., :4] + _scale * nu1], dim=-1)
        return w_fourvecs
