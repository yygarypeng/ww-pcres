import torch
import torch.nn as nn


class DenseDropoutBlock(nn.Module):
    """
    Dense -> BN -> Activation -> (Dropout)
    Activation: SiLU
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.fc(x))))


class ResidualBlock(nn.Module):
    """
    Residual block with two DenseDropoutBlocks.
    If in/out dims differ in the TF, they up-projected via a linear.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False) # It's just a projection, dont use bias

        self.block1 = DenseDropoutBlock(out_dim if self.proj else in_dim, out_dim, dropout)
        self.block2 = DenseDropoutBlock(out_dim, out_dim, dropout)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.proj:
            x = self.proj(x)
        y = self.block1(x)
        y = self.block2(y)
        z = x + y # elementwise add
        return self.act(self.bn(z))


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
