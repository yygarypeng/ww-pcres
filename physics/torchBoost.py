import numpy as np
import torch
import torch.nn as nn

from physics import _diff_angle, _sum_angle

#######################
# Auxiliary Functions #
#######################


class _SafeAcos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        x = torch.clamp(x, -1.0, 1.0)
        ctx.save_for_backward(x)
        ctx.eps = eps
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = torch.sqrt((1.0 - x * x).clamp_min(ctx.eps * ctx.eps))
        # return grad_x, grad_eps (return things orderly in foward inputs)
        return -grad_output / denom, None


class _SafeAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, eps):
        ctx.save_for_backward(y, x)
        ctx.eps = eps
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = (x * x + y * y).clamp_min(ctx.eps * ctx.eps)
        # return grad_y, grad_x, grad_eps (return things orderly in foward inputs)
        return grad_output * x / denom, -grad_output * y / denom, None


def safe_acos(x, eps=1e-6):
    return _SafeAcos.apply(x, eps)


def safe_atan2(y, x, eps=1e-6):
    return _SafeAtan2.apply(y, x, eps)


#######################
# Main bossting codes #
#######################


class Booster(nn.Module):
    """
    Torch W-rest-frame booster.
    4-vec needs to be in (px, py, pz, E) format.

    :param lep: Tensor of shape [batch, 8] containing the two lepton 4-vectors (lep0, lep1) in the lab frame.
    :param wboson: Tensor of shape [batch, 8] containing the two W boson 4-vectors (w0, w1) in the lab frame.
    """

    def __init__(
        self,
        lep,
        wboson,
        eps=1e-12,
    ):
        super().__init__()
        self.eps = eps
        w0, w1 = wboson[..., :4], wboson[..., 4:8]
        lep0, lep1 = lep[..., :4], lep[..., 4:8]
        self.particles = torch.cat([w0, lep0, w1, lep1], dim=-1)

    #############
    # Utilities #
    #############

    def _eps(self, x):
        return max(self.eps, torch.finfo(x.dtype).eps)

    def _norm(self, x, keepdim=True):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=keepdim).clamp_min(self._eps(x))

    def _mass2(self, p4):
        return p4[..., 3] ** 2 - torch.sum(p4[..., 0:3] ** 2, dim=-1)

    def _has_rest_frame(self, p4):
        return (
            torch.isfinite(p4).all(dim=-1) & (p4[..., 3] > 0.0) & (self._mass2(p4) > self._eps(p4))
        )

    def valid_rest_frame_mask(self, particles=None):
        if particles is None:
            particles = self.particles
        higgs_ok, w0_ok, w1_ok = self._rest_frame_flags(particles)
        _, _, _, _, w1_h, _ = self._rest_frame_state(particles, higgs_ok=higgs_ok)
        return self._valid_rest_frame_mask(particles, higgs_ok, w0_ok, w1_ok, w1_h)

    def _rest_frame_flags(self, particles):
        w0 = particles[..., 0:4]
        w1 = particles[..., 8:12]
        higgs_ok = self._has_rest_frame(w0 + w1)
        return higgs_ok, self._has_rest_frame(w0), self._has_rest_frame(w1)

    def _rest_frame_state(self, particles=None, higgs_ok=None):
        if particles is None:
            particles = self.particles

        w0 = particles[..., 0:4]
        lep0 = particles[..., 4:8]
        w1 = particles[..., 8:12]
        lep1 = particles[..., 12:16]
        higgs = w0 + w1
        if higgs_ok is None:
            higgs_ok = self._has_rest_frame(higgs)
        w0_h = self._boost_to_rest(w0, higgs, higgs_ok)
        lep0_h = self._boost_to_rest(lep0, higgs, higgs_ok)
        w1_h = self._boost_to_rest(w1, higgs, higgs_ok)
        lep1_h = self._boost_to_rest(lep1, higgs, higgs_ok)
        return particles, higgs, w0_h, lep0_h, w1_h, lep1_h

    def _valid_rest_frame_mask(self, particles, higgs_ok, w0_ok, w1_ok, w1_h):
        lep0 = particles[..., 4:8]
        lep1 = particles[..., 12:16]
        w1_axis = w1_h[..., 0:3]
        eps = self._eps(w1_h)
        axis_norm = torch.linalg.vector_norm(w1_axis, dim=-1)
        transverse_fraction = torch.linalg.vector_norm(
            w1_axis[..., 0:2], dim=-1
        ) / axis_norm.clamp_min(eps)

        return (
            torch.isfinite(lep0).all(dim=-1)
            & torch.isfinite(lep1).all(dim=-1)
            & higgs_ok
            & w0_ok
            & w1_ok
            & torch.isfinite(w1_axis).all(dim=-1)
            & (axis_norm > eps)
            & (transverse_fraction > eps**0.5)
        )

    ###################
    # Boost functions #
    ###################

    def _boost(self, p4, beta):
        """
        Lorentz boost with the same sign convention as ROOT TLorentzVector.Boost.
        https://root.cern.ch/doc/v632/classTLorentzVector.html
        """
        p3 = p4[..., 0:3]
        e = p4[..., 3:4]

        eps = self._eps(p4)
        beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        beta2 = torch.sum(beta * beta, dim=-1, keepdim=True)
        valid_beta = beta2 < 1.0  # cannot exceed the speed of light
        beta = torch.where(valid_beta, beta, torch.zeros_like(beta))
        beta2 = torch.where(valid_beta, beta2, torch.zeros_like(beta2))
        gamma = torch.rsqrt((1.0 - beta2).clamp_min(eps))
        beta_dot_p = torch.sum(beta * p3, dim=-1, keepdim=True)
        # for small beta, gamma ~ 1 + 0.5 * beta2, so (gamma - 1) / beta2 ~ 0.5 := gamma2
        gamma2 = torch.where(beta2 > eps, (gamma - 1.0) / beta2, 0.5 * torch.ones_like(beta2))

        boosted_p3 = p3 + gamma2 * beta_dot_p * beta + gamma * e * beta
        boosted_e = gamma * (e + beta_dot_p)
        return torch.cat([boosted_p3, boosted_e], dim=-1)

    def _boost_to_rest(self, p4, reference, reference_is_valid=None):
        valid = (
            self._has_rest_frame(reference) if reference_is_valid is None else reference_is_valid
        ).unsqueeze(-1)
        energy = torch.where(valid, reference[..., 3:4], torch.ones_like(reference[..., 3:4]))
        beta = torch.where(
            valid, reference[..., 0:3] / energy, torch.zeros_like(reference[..., 0:3])
        )
        return self._boost(p4, -beta)

    ######################
    # Basis construction #
    ######################

    def _basis(self, w_axis):
        k = w_axis[..., 0:3] / self._norm(w_axis[..., 0:3])

        beam = torch.zeros_like(k)
        beam[..., 2] = 1.0

        y = torch.sum(beam * k, dim=-1, keepdim=True)
        transverse = torch.sqrt((1.0 - y * y).clamp_min(0.0) + self._eps(w_axis))

        r = (beam - y * k) / transverse
        n = torch.cross(beam, k, dim=-1) / transverse
        return n, r, k

    ###############
    # Projections #
    ###############

    @staticmethod
    def _project(p4, n, r, k):
        p3 = p4[..., 0:3]
        p3_projected = torch.stack(
            [
                torch.sum(p3 * n, dim=-1),
                torch.sum(p3 * r, dim=-1),
                torch.sum(p3 * k, dim=-1),
            ],
            dim=-1,
        )
        return torch.cat([p3_projected, p4[..., 3:4]], dim=-1)

    ######################
    # Feature extraction #
    ######################

    def _theta(self, p4):
        p = self._norm(p4[..., 0:3], keepdim=False)
        cos_theta = torch.clamp(p4[..., 2] / p, -1.0, 1.0)
        return safe_acos(cos_theta)

    @staticmethod
    def _phi(p4):
        return safe_atan2(p4[..., 1], p4[..., 0])

    ##################
    # Main functions #
    ##################

    def lep_4_in_w_rest(self, particles=None):
        """
        Return:
            lep0_rest, lep1_rest

        Both have shape [batch, 4] and are expressed in the (n, r, k) basis.
        """
        _, _, w0_h, lep0_h, w1_h, lep1_h = self._rest_frame_state(particles)

        return self._lep_4_from_rest_frame_state(w0_h, lep0_h, w1_h, lep1_h)

    def _lep_4_from_rest_frame_state(self, w0_h, lep0_h, w1_h, lep1_h, w0_ok=None, w1_ok=None):

        # k is along W1 in the Higgs rest frame, beam direction is +z.
        n, r, k = self._basis(w1_h)

        lep0_w = self._boost_to_rest(lep0_h, w0_h, w0_ok)
        lep1_w = self._boost_to_rest(lep1_h, w1_h, w1_ok)

        return (
            self._project(lep0_w, n, r, k),
            self._project(lep1_w, n, r, k),
        )

    def lep_theta_phi_with_validity(self, particles=None):
        if particles is None:
            particles = self.particles
        higgs_ok, w0_ok, w1_ok = self._rest_frame_flags(particles)
        state = self._rest_frame_state(particles, higgs_ok=higgs_ok)
        valid = self._valid_rest_frame_mask(state[0], higgs_ok, w0_ok, w1_ok, state[4])
        lep0, lep1 = self._lep_4_from_rest_frame_state(*state[2:], w0_ok=w0_ok, w1_ok=w1_ok)
        angles = torch.stack(
            [self._theta(lep0), self._phi(lep0), self._theta(lep1), self._phi(lep1)],
            dim=-1,
        )
        return valid, angles

    def lep_theta_phi_in_w_rest(self, particles=None):
        lep0, lep1 = self.lep_4_in_w_rest(particles)
        lep0_theta = self._theta(lep0)
        lep0_phi = self._phi(lep0)
        lep1_theta = self._theta(lep1)
        lep1_phi = self._phi(lep1)
        sum_theta = _sum_angle(lep0_theta, lep1_theta)
        diff_theta = _diff_angle(lep0_theta, lep1_theta)
        sum_phi = _sum_angle(lep0_phi, lep1_phi)
        diff_phi = _diff_angle(lep0_phi, lep1_phi)

        return (
            lep0_theta,
            lep0_phi,
            lep1_theta,
            lep1_phi,
            sum_theta,
            diff_theta,
            sum_phi,
            diff_phi,
        )

    def forward(self, particles=None):
        return self.lep_4_in_w_rest(particles)


def _mock_inputs(batch=1024, device="cpu"):
    """Make test lepton and W-boson tensors with the layout used by Booster."""
    dtype = torch.float32

    w0_p3 = torch.randn(batch, 3, dtype=dtype, device=device) * 40.0
    w1_p3 = torch.randn(batch, 3, dtype=dtype, device=device) * 40.0
    lep0_p3 = torch.randn(batch, 3, dtype=dtype, device=device) * 25.0
    lep1_p3 = torch.randn(batch, 3, dtype=dtype, device=device) * 25.0

    w0_e = torch.sqrt(torch.sum(w0_p3 * w0_p3, dim=1, keepdim=True) + 80.379**2)
    w1_e = torch.sqrt(torch.sum(w1_p3 * w1_p3, dim=1, keepdim=True) + 80.379**2)
    lep0_e = torch.sqrt(torch.sum(lep0_p3 * lep0_p3, dim=1, keepdim=True) + 0.105**2)
    lep1_e = torch.sqrt(torch.sum(lep1_p3 * lep1_p3, dim=1, keepdim=True) + 0.105**2)

    w0 = torch.cat([w0_p3, w0_e], dim=-1)
    w1 = torch.cat([w1_p3, w1_e], dim=-1)
    lep0 = torch.cat([lep0_p3, lep0_e], dim=-1)
    lep1 = torch.cat([lep1_p3, lep1_e], dim=-1)

    lep = torch.cat([lep0, lep1], dim=-1)
    wboson = torch.cat([w0, w1], dim=-1)
    return lep, wboson


def _plot_theta_phi(
    torch_angles,
    root_angles=None,
    output="torchboost_theta_phi_compare.png",
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values = [
        (
            "lep0 theta",
            torch_angles[0],
            None if root_angles is None else root_angles[0],
            (0.0, np.pi),
        ),
        (
            "lep0 phi",
            torch_angles[1],
            None if root_angles is None else root_angles[1],
            (-np.pi, np.pi),
        ),
        (
            "lep1 theta",
            torch_angles[2],
            None if root_angles is None else root_angles[2],
            (0.0, np.pi),
        ),
        (
            "lep1 phi",
            torch_angles[3],
            None if root_angles is None else root_angles[3],
            (-np.pi, np.pi),
        ),
        (
            "sum theta",
            torch_angles[4],
            None if root_angles is None else root_angles[4],
            (-np.pi, np.pi),
        ),
        (
            "diff theta",
            torch_angles[5],
            None if root_angles is None else root_angles[5],
            (-np.pi, np.pi),
        ),
        (
            "sum phi",
            torch_angles[6],
            None if root_angles is None else root_angles[6],
            (-np.pi, np.pi),
        ),
        (
            "diff phi",
            torch_angles[7],
            None if root_angles is None else root_angles[7],
            (-np.pi, np.pi),
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, (title, torch_tensor, root_array, xlim) in zip(axes.flat, values):
        torch_data = torch_tensor.detach().cpu().numpy()
        ax.hist(
            torch_data,
            bins=60,
            range=xlim,
            histtype="step",
            linewidth=1.8,
            density=True,
            label="torch",
        )
        if root_array is not None:
            ax.hist(
                root_array,
                bins=60,
                range=xlim,
                histtype="step",
                linewidth=1.3,
                density=True,
                label="ohbboosting",
            )
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def _ohbboosting_angles(particles):
    from physics.ohbboosting import Booster as RootBooster

    root_booster = RootBooster(particles)
    root_booster.setup()
    lep0_angles, lep1_angles = root_booster.lep_theta_phi_in_w_rest()

    def unpack_theta_phi(angles):
        if len(angles) == 2:
            return angles
        # if len(angles) == 3:
        #     theta, sin_phi, cos_phi = angles
        #     return theta, np.arctan2(sin_phi, cos_phi)
        raise ValueError(f"Expected 2 angle arrays from ohbboosting, got {len(angles)}")

    lep0_theta, lep0_phi = unpack_theta_phi(lep0_angles)
    lep1_theta, lep1_phi = unpack_theta_phi(lep1_angles)
    sum_theta = _sum_angle(lep0_theta, lep1_theta)
    diff_theta = _diff_angle(lep0_theta, lep1_theta)
    sum_phi = _sum_angle(lep0_phi, lep1_phi)
    diff_phi = _diff_angle(lep0_phi, lep1_phi)

    return (
        lep0_theta,
        lep0_phi,
        lep1_theta,
        lep1_phi,
        sum_theta,
        diff_theta,
        sum_phi,
        diff_phi,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lep, wboson = _mock_inputs(batch=1024, device=device)
    wboson.requires_grad_(True)

    booster = Booster(lep, wboson).to(device)
    lep0, lep1 = booster()
    torch_angles = booster.lep_theta_phi_in_w_rest()
    (
        lep0_theta,
        lep0_phi,
        lep1_theta,
        lep1_phi,
        sum_theta,
        diff_theta,
        sum_phi,
        diff_phi,
    ) = torch_angles

    loss = lep0.square().mean() + lep1.square().mean()
    loss = loss + sum(angle.mean() for angle in torch_angles)
    loss.backward()

    print("device:", device)
    print("particles:", tuple(booster.particles.shape))
    print("lep0 rest:", tuple(lep0.shape))
    print("lep1 rest:", tuple(lep1.shape))
    print("lep0 theta range:", float(lep0_theta.min()), float(lep0_theta.max()))
    print("lep0 phi range:", float(lep0_phi.min()), float(lep0_phi.max()))
    print("lep1 theta range:", float(lep1_theta.min()), float(lep1_theta.max()))
    print("lep1 phi range:", float(lep1_phi.min()), float(lep1_phi.max()))
    print("sum theta range:", float(sum_theta.min()), float(sum_theta.max()))
    print("diff theta range:", float(diff_theta.min()), float(diff_theta.max()))
    print("sum phi range:", float(sum_phi.min()), float(sum_phi.max()))
    print("diff phi range:", float(diff_phi.min()), float(diff_phi.max()))
    print("gradient finite:", bool(torch.isfinite(wboson.grad).all()))

    particles = booster.particles.detach().cpu().numpy()
    try:
        root_angles = _ohbboosting_angles(particles)
        for name, torch_angle, root_angle in zip(
            [
                "lep0 theta",
                "lep0 phi",
                "lep1 theta",
                "lep1 phi",
                "sum theta",
                "diff theta",
                "sum phi",
                "diff phi",
            ],
            torch_angles,
            root_angles,
        ):
            diff = np.max(np.abs(torch_angle.detach().cpu().numpy() - root_angle))
            print(f"{name} max torch-ohbboosting diff: {diff:.3e}")
    except Exception as err:
        root_angles = None
        print(f"ohbboosting comparison skipped: {err}")

    _plot_theta_phi(torch_angles, root_angles)
