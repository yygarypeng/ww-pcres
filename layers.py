import torch
import torch.nn as nn
import torch.nn.functional as F


class Standardization(nn.Module):
    def __init__(self, mean, std, eps=1e-16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clamp_min(eps))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(eps))

    def forward(self, x):
        return (x - self.mean) / (self.std)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 4 * d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            # nn.Dropout(dropout),
        )
        

    def forward(self, x, key_padding_mask=None):
        res = x
        x = self.norm1(x)
        x, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = res + x
        
        x = x + self.ffn(self.norm2(x))
        return x

class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        LN(in_dim) -> GELU -> Linear(in_dim -> out_dim) -> Dropout
    P0st-activation block:
        Linear(in_dim -> out_dim) -> LN(out_dim) -> GELU -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, post_act=False):
        super().__init__()
        if not post_act:
            # pre-activation for deeper networks (more stable gradients)
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            )
        else:
            # post-activation for better interpretability
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        # projection only when needed
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        # two pre-activation dense blocks
        self.block1 = DenseDropoutBlock(in_dim, out_dim, dropout)
        self.block2 = DenseDropoutBlock(out_dim, out_dim, dropout=0.0)

    def forward(self, x):
        identity = self.proj(x)
        y = self.block1(x)
        y = self.block2(y)
        return identity + y 

class WBosonFourVectorLayer(nn.Module):
    """
    Compute W four-vectors from leptons and predicted neutrino 3-momenta.
    """
    def forward(self, lep0, lep1, nu_3mom):
        nu0_3, nu1_3 = nu_3mom[..., :3], nu_3mom[..., 3:]
        # neutrino energies as |p| for (approx) massless
        nu0_E = torch.sqrt(torch.clamp(torch.sum(nu0_3 ** 2, dim=-1, keepdim=True), min=1e-16))
        nu1_E = torch.sqrt(torch.clamp(torch.sum(nu1_3 ** 2, dim=-1, keepdim=True), min=1e-16))
        nu0_4 = torch.cat([nu0_3, nu0_E], dim=-1)
        nu1_4 = torch.cat([nu1_3, nu1_E], dim=-1)
        return torch.cat([lep0 + nu0_4, lep1 + nu1_4], dim=-1)

class FeatureImportance(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        # post-activation gating
        hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Tanh(),
        )

    def forward(self, x):
        gate = self.gate(self.norm(x))
        return x * (1.0 + gate)