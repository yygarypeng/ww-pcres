import torch
import torch.nn as nn


class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        BN(in_dim) -> SiLU -> Linear(in_dim -> out_dim) -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.act = nn.SiLU()
        self.fc = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.bn(x)
        y = self.act(y)
        y = self.fc(y)
        y = self.drop(y)
        return y

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
        nu0_E = torch.sqrt(torch.clamp(torch.sum(nu0_3 ** 2, dim=-1, keepdim=True), min=1e-10))
        nu1_E = torch.sqrt(torch.clamp(torch.sum(nu1_3 ** 2, dim=-1, keepdim=True), min=1e-10))
        nu0_4 = torch.cat([nu0_3, nu0_E], dim=-1)
        nu1_4 = torch.cat([nu1_3, nu1_E], dim=-1)
        # output concat: [ (lep0 + nu0_4), (lep1 + nu1_4) ] => shape (..., 8)
        return torch.cat([lep0 + nu0_4, lep1 + nu1_4], dim=-1)
