import torch
import torch.nn as nn


class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        LN(in_dim) -> SiLU -> Linear(in_dim -> out_dim) -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
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
        self.block2 = DenseDropoutBlock(out_dim, out_dim, dropout)

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
    
class Standardization(nn.Module):
    def __init__(self, mean, std, eps=1e-16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

class FeatureAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False), # tuen off bias for signal preserving
            nn.SiLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        # x: (B, D)
        w = self.fc(x)
        return self.norm(x + x * w)