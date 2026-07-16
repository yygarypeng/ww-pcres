import torch
import torch.nn as nn
import torch.nn.functional as F


class Standardization(nn.Module):
    def __init__(self, mean, std, eps=1e-16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(eps))

    def forward(self, x):
        return (x - self.mean) / (self.std)

class _AttnFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.3):
        super().__init__()
        self.ffn = nn.Sequential(
            # nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        x = self.ffn(x)
        return x
    
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
        x, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = res + self.dropout(x)
        
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = res + self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.shortcut = (
            nn.Identity() if in_dim == out_dim
            else nn.Linear(in_dim, out_dim, bias=False)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        y = self.residual(x)
        return x_shortcut + y

class WBosonFourVectorLayer(nn.Module):
    def __init__(self, eps=1.0e-16):
        super().__init__()
        self.eps = eps

    def forward(self, lep0, lep1, nu_params, met):
        # Layout: [nu0_px, nu0_py, nu0_pz, nu1_pz, dmet_px, dmet_py].
        # The residual accounts for reco MET mismatch rather than smearing
        # invisible momenta directly.
        nu0_3 = nu_params[..., :3]
        nu1_pz = nu_params[..., 3:4]
        dmet = nu_params[..., 4:6]

        # measured MET = neutrino transverse momentum + detector/reconstruction residual
        nu_met = met - dmet

        nu1_px = nu_met[..., 0:1] - nu0_3[..., 0:1]
        nu1_py = nu_met[..., 1:2] - nu0_3[..., 1:2]
        nu1_3 = torch.cat([nu1_px, nu1_py, nu1_pz], dim=-1)

        # neutrino energies as |p| for (approx) massless
        nu0_E = torch.sqrt(torch.clamp(torch.sum(nu0_3 ** 2, dim=-1, keepdim=True), min=self.eps))
        nu1_E = torch.sqrt(torch.clamp(torch.sum(nu1_3 ** 2, dim=-1, keepdim=True), min=self.eps))
        w0_3 = lep0[..., :3] + nu0_3
        w1_3 = lep1[..., :3] + nu1_3
        w0_E = torch.clamp(lep0[..., 3:4] + nu0_E, min=self.eps)
        w1_E = torch.clamp(lep1[..., 3:4] + nu1_E, min=self.eps)
        w0_logE = torch.log(w0_E)
        w1_logE = torch.log(w1_E)
        return torch.cat([w0_3, w0_logE, w1_3, w1_logE], dim=-1)
