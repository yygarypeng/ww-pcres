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

class NeutrinosLayer(nn.Module):
    """
    Compute W four-vectors from predicted leptons and derived neutrinos 3-momenta.
    """
    def forward(self, lep_pos, lep_neg):
        # in the W rest frame, neutrinos are back-to-back with leptons
        nu_neg_3, nu_pos_3 = -lep_pos[..., :3], -lep_neg[..., :3]
        # neutrino energies as |p| for (approx) massless (cannot be exact 0 to avoid NaNs)
        nu_neg_E = torch.sqrt(torch.clamp(torch.sum(nu_neg_3 ** 2, dim=-1, keepdim=True), min=1e-10))
        nu_pos_E = torch.sqrt(torch.clamp(torch.sum(nu_pos_3 ** 2, dim=-1, keepdim=True), min=1e-10))
        nu_neg = torch.cat([nu_neg_3, nu_neg_E], dim=-1)
        nu_pos = torch.cat([nu_pos_3, nu_pos_E], dim=-1)
        w_pos_4vec = lep_pos + nu_neg
        w_neg_4vec = lep_neg + nu_pos
        w_pos_m_2 = torch.clamp(w_pos_4vec[..., 3:4] ** 2 - torch.sum(w_pos_4vec[..., :3] ** 2, dim=-1, keepdim=True), min=1e-10)
        w_neg_m_2 = torch.clamp(w_neg_4vec[..., 3:4] ** 2 - torch.sum(w_neg_4vec[..., :3] ** 2, dim=-1, keepdim=True), min=1e-10)
        return torch.cat([lep_pos, lep_neg, torch.sqrt(w_pos_m_2), torch.sqrt(w_neg_m_2)], dim=-1)