import torch
import torch.nn as nn


class Standardization(nn.Module):
    def __init__(self, mean, std, eps=1.0e-16):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(eps))

    def forward(self, x):
        return (x - self.mean) / (self.std)


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
    """W four-vectors whose neutrinos are massless and on the Higgs mass shell.

    ``nu_params`` holds the two neutrino three-momenta in GeV, [nu0_p, nu1_p].
    Their energies are |p|, and the pair is then rescaled by the single positive
    factor that puts the dilepton plus dineutrino system at HIGGS_MASS.
    Rescaling leaves E = |p| intact, so the massless construction survives the
    constraint, and it leaves both neutrino directions intact, so only their
    magnitudes pay for it. Writing L for the dilepton and N for the dineutrino
    four-vector,

        m^2(L + s N) = m_ll^2 + 2 s (L.N) + s^2 N^2

    is strictly increasing in s >= 0, because two massless future-pointing
    neutrinos give N^2 >= 0 and L.N > 0. The positive root is therefore unique
    and exists whenever m_ll stays below HIGGS_MASS. The rationalized form used
    below never divides by N^2, which vanishes for collinear neutrinos.

    Only the direction of the predicted pair reaches the output: the W
    four-vectors are invariant under nu_params -> c * nu_params for any c > 0,
    because the constraint sets the overall scale of the neutrino system itself.
    The six parameters therefore carry five effective degrees of freedom, and
    MET reaches the solution through the network inputs alone. The MET
    correction that the solution implies is the measured MET minus the
    transverse sum of the returned neutrinos, which is what ``dmet_loss`` wants.

    Keeping m_ll below HIGGS_MASS is the loader's job, through
    ``data.max_dilepton_mass``: adding massless neutrinos can only raise an
    invariant mass, so a heavier lepton pair can never be brought back to the
    Higgs mass shell. MIN_DELTA_MASS2 floors the mass gap, keeping the root real
    rather than NaN if an unselected event reaches the layer.
    """

    HIGGS_MASS = 125.0
    MIN_DELTA_MASS2 = 1.0
    EPS = 1.0e-6

    @staticmethod
    def _dot(a, b):
        """Mostly-minus inner product of two four-vectors, keeping the last dim."""
        return a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(dim=-1, keepdim=True)

    @staticmethod
    def _massless(momentum):
        """A four-vector with E = |p|."""
        energy = torch.linalg.vector_norm(momentum, dim=-1, keepdim=True)
        return torch.cat([momentum, energy], dim=-1)

    def higgs_scale(self, dilep, dinu):
        """The positive scale putting ``dilep + scale * dinu`` on the mass shell."""
        b = self._dot(dilep, dinu)
        a = self._dot(dinu, dinu).clamp_min(0.0)
        delta = (self.HIGGS_MASS**2 - self._dot(dilep, dilep)).clamp_min(self.MIN_DELTA_MASS2)
        denominator = (b + torch.sqrt(b * b + a * delta)).clamp_min(self.EPS)
        return delta / denominator

    def forward(self, lep0, lep1, nu_params, return_aux=False):
        nu0 = self._massless(nu_params[..., 0:3])
        nu1 = self._massless(nu_params[..., 3:6])

        scale = self.higgs_scale(lep0[..., :4] + lep1[..., :4], nu0 + nu1)
        w_fourvecs = torch.cat([lep0[..., :4] + scale * nu0, lep1[..., :4] + scale * nu1], dim=-1)
        # if return_aux:
        #     return w_fourvecs, {"higgs_scale": scale}
        return w_fourvecs
