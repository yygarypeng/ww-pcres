"""Loss terms for the physics-constrained W regressor.

Two conventions keep the configured weights meaningful:

* Pointwise terms are a mean absolute error in GeV. Every residual is reduced to
  a GeV-valued quantity first, so equal weights mean equal cost per GeV and the
  logged values read directly as physical errors.
* MMD terms compare distributions through bounded, O(1) feature maps. The kernel
  bandwidths are fixed absolute numbers, so a feature map that leaves its natural
  units in place would put every pair far outside every bandwidth and collapse
  the kernel.
"""

import math

import torch
import torch.nn.functional as F

from physics.torchBoost import Booster

TOR = 1e-16
W_MASS_SCALE = 80.4
H_MASS_SCALE = 125.0

##################
# Shared helpers #
##################


def invariant_mass2(fourvec):
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    return E**2 - (px**2 + py**2 + pz**2)


def mass_residual(fourvec, target_mass2, reference_mass):
    """Mass residual in GeV, linearized as (m^2 - m_target^2) / (2 m_ref).

    This equals m - m_target to first order around the reference mass and takes no
    square root, so it stays finite and differentiable when a prediction goes
    spacelike instead of folding that case onto the timelike side.
    """
    return (invariant_mass2(fourvec) - target_mass2) / (2.0 * reference_mass)


def _valid_kinematic_rows(x_batch, y_true, y_pred):
    return (
        torch.isfinite(x_batch[..., :8]).all(dim=-1)
        & torch.isfinite(y_true[..., :8]).all(dim=-1)
        & torch.isfinite(y_pred).all(dim=-1)
    )


def _sanitized_rows(tensor, keep):
    """Zero invalid rows before differentiable operations to prevent NaN gradients."""
    return torch.where(keep.unsqueeze(-1), tensor, torch.zeros_like(tensor))


def mass_mmd_features(w_fourvecs):
    """Bounded mass features for the fixed-bandwidth kernels.

    asinh is monotonic in m^2 and defined for spacelike values, and dividing by
    m_W^2 first maps the whole off-shell spectrum into an O(1) range that the
    configured bandwidths can resolve.
    """
    mass2 = torch.stack(
        [invariant_mass2(w_fourvecs[..., :4]), invariant_mass2(w_fourvecs[..., 4:8])],
        dim=-1,
    )
    return torch.asinh(mass2 / W_MASS_SCALE**2)


def angular_mmd_features(angles):
    theta_pos = angles[..., 0]
    phi_pos = angles[..., 1]
    theta_neg = angles[..., 2]
    phi_neg = angles[..., 3]

    return torch.stack(
        [
            2.0 * theta_pos / torch.pi - 1.0,
            torch.sin(phi_pos),
            torch.cos(phi_pos),
            2.0 * theta_neg / torch.pi - 1.0,
            torch.sin(phi_neg),
            torch.cos(phi_neg),
        ],
        dim=-1,
    )


####################
# Pointwise losses #
####################


def w_fourvec_loss(y_true, y_pred):
    """Mean L1 loss on raw W components in GeV."""
    return F.l1_loss(y_pred, y_true[..., :8])


def higgs_fourvec_loss(y_true, y_pred):
    """Mean L1 loss on the summed W four-vectors in GeV."""
    higgs_true = y_true[..., :4] + y_true[..., 4:8]
    higgs_pred = y_pred[..., :4] + y_pred[..., 4:8]
    return F.l1_loss(higgs_pred, higgs_true)


def w_mass_loss(y_true, y_pred):
    """Mean L1 loss on the two W mass residuals in GeV.

    The truth W mass reaches far off shell, so the fixed W scale is used as the
    linearization reference instead of the per-event target, whose reciprocal
    would blow up as that target approaches zero.
    """
    residuals = torch.stack(
        [
            mass_residual(y_pred[..., :4], y_true[..., 8] ** 2, W_MASS_SCALE),
            mass_residual(y_pred[..., 4:8], y_true[..., 9] ** 2, W_MASS_SCALE),
        ],
        dim=-1,
    )
    return residuals.abs().mean()


def higgs_mass_loss(y_pred, target_mass=H_MASS_SCALE):
    """Mean L1 loss on the Higgs mass residual in GeV."""
    higgs = y_pred[..., :4] + y_pred[..., 4:8]
    return mass_residual(higgs, target_mass**2, target_mass).abs().mean()


def dmet_loss(x_batch, y_true, dmet):
    """Mean L1 loss on raw MET corrections in GeV."""
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]
    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy

    return F.l1_loss(dmet, dmet_target)


##############
# Global MMD #
##############


def compute_mmd(x, y, *, kernel="imq", bandwidths=(0.1, 1.0, 10.0), valid_mask=None):
    """Compute fixed-bandwidth V-statistic MMD without changing the batch shape."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of paired rows")
    if valid_mask is not None and valid_mask.shape[0] != x.shape[0]:
        raise ValueError("valid_mask must have one entry per paired row")

    finite = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    if valid_mask is not None:
        finite = finite & valid_mask
    row_weight = finite.to(x.dtype).unsqueeze(-1)
    pair_weight = row_weight * row_weight.T
    pair_total = pair_weight.sum().clamp_min(1.0)

    # Weighting alone cannot prevent NaNs from entering pairwise distances.
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * row_weight
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) * row_weight

    x2 = x.square().sum(dim=1)
    y2 = y.square().sum(dim=1)
    dxx = (x2[:, None] + x2[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
    dxy = (x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)).clamp_min(0.0)
    with torch.no_grad():
        dyy = (y2[:, None] + y2[None, :] - 2.0 * (y @ y.T)).clamp_min(0.0)

    def kernel_mean(distance, bandwidth):
        bandwidth2 = bandwidth**2

        if kernel == "imq":
            values = bandwidth2 / (bandwidth2 + distance + TOR)
        else:
            values = torch.exp(-0.5 * distance / (bandwidth2 + TOR))
        return (values * pair_weight).sum() / pair_total

    mmd = x.new_zeros(())
    for bandwidth in bandwidths:
        # Keep the constant y-y value while excluding it from the autograd graph.
        with torch.no_grad():
            dyy_term = kernel_mean(dyy, bandwidth)
        mmd = mmd + kernel_mean(dxx, bandwidth) + dyy_term
        mmd = mmd - 2.0 * kernel_mean(dxy, bandwidth)

    return (mmd / len(bandwidths)).clamp_min(0.0)


def alpha_mmd(x_batch, y_true, y_pred, valid_mask=None, **mmd_kwargs):
    if valid_mask is None:
        valid_mask = _valid_kinematic_rows(x_batch, y_true, y_pred)

    lep = _sanitized_rows(x_batch[..., :8], valid_mask)
    w_true = _sanitized_rows(y_true[..., :8], valid_mask)
    w_pred = _sanitized_rows(y_pred, valid_mask)

    lep_pos = lep[..., :4]
    lep_neg = lep[..., 4:8]

    nu_pos_true = w_true[..., :4] - lep_pos
    nu_neg_true = w_true[..., 4:8] - lep_neg
    nu_pos_pred = w_pred[..., :4] - lep_pos
    nu_neg_pred = w_pred[..., 4:8] - lep_neg

    p_pos_true = torch.linalg.vector_norm(nu_pos_true[..., :3], dim=-1)
    p_neg_true = torch.linalg.vector_norm(nu_neg_true[..., :3], dim=-1)
    p_pos_pred = torch.linalg.vector_norm(nu_pos_pred[..., :3], dim=-1)
    p_neg_pred = torch.linalg.vector_norm(nu_neg_pred[..., :3], dim=-1)

    total_true = p_pos_true + p_neg_true
    total_pred = p_pos_pred + p_neg_pred

    alpha_valid = (
        valid_mask
        & torch.isfinite(total_true)
        & torch.isfinite(total_pred)
        & (total_true > TOR)
        & (total_pred > TOR)
    )
    # Rows that carry no weight still go through the division, so keep their
    # denominators finite instead of selecting the surviving rows out.
    safe_total_true = torch.where(alpha_valid, total_true, torch.ones_like(total_true))
    safe_total_pred = torch.where(alpha_valid, total_pred, torch.ones_like(total_pred))

    true_alpha = (p_pos_true / safe_total_true).unsqueeze(-1)
    pred_alpha = (p_pos_pred / safe_total_pred).unsqueeze(-1)

    return compute_mmd(
        2.0 * pred_alpha - 1.0,
        2.0 * true_alpha - 1.0,
        valid_mask=alpha_valid,
        **mmd_kwargs,
    )


def w_mass_mmd(x_batch, y_true, y_pred, valid_mask=None, **mmd_kwargs):
    if valid_mask is None:
        valid_mask = _valid_kinematic_rows(x_batch, y_true, y_pred)

    true_features = mass_mmd_features(_sanitized_rows(y_true[..., :8], valid_mask))
    pred_features = mass_mmd_features(_sanitized_rows(y_pred, valid_mask))
    return compute_mmd(pred_features, true_features, valid_mask=valid_mask, **mmd_kwargs)


def _angular_mmd_with_valid_mask(x_batch, y_true, y_pred, valid_mask, **mmd_kwargs):
    lep = x_batch[..., :8]
    with torch.no_grad():
        truth_valid, truth_angles = Booster(lep, y_true[..., :8]).lep_theta_phi_with_validity()
        truth_features = angular_mmd_features(truth_angles)

    # The boost chain is only NaN-safe for finite inputs, and its gradient runs
    # before the weighting in compute_mmd, so clean the rows up front.
    finite = valid_mask
    pred_valid, pred_angles = Booster(
        _sanitized_rows(lep, finite),
        _sanitized_rows(y_pred, finite),
    ).lep_theta_phi_with_validity()

    return compute_mmd(
        angular_mmd_features(pred_angles),
        truth_features,
        valid_mask=finite & truth_valid & pred_valid,
        **mmd_kwargs,
    )


def angular_mmd(x_batch, y_true, y_pred, **mmd_kwargs):
    """Compute MMD over lepton angles in the W rest frames."""
    return _angular_mmd_with_valid_mask(
        x_batch,
        y_true,
        y_pred,
        _valid_kinematic_rows(x_batch, y_true, y_pred),
        **mmd_kwargs,
    )


###################
# Local MMD (WIP) #
###################


def _positive_median_pairwise_distance(values):
    if values.shape[0] < 2:
        return values.new_tensor(1.0)

    distances = torch.pdist(values, p=2)
    distances = distances[torch.isfinite(distances) & (distances > 0.0)]

    if distances.numel() == 0:
        return values.new_tensor(1.0)

    return torch.median(distances)


def _validate_bandwidth_multipliers(values, name):
    multipliers = tuple(float(value) for value in values)

    if not multipliers:
        raise ValueError(f"{name} must contain at least one value")
    if not all(math.isfinite(value) and value > 0.0 for value in multipliers):
        raise ValueError(f"{name} values must be finite and positive")

    return multipliers


def compute_local_mmd(
    x,
    y,
    cond,
    *,
    local=True,
    feature_kernel="imq",
    condition_kernel="rbf",
    feature_bandwidth_multipliers=(0.25, 0.5, 1.0, 2.0),
    condition_bandwidth_multipliers=(0.5, 1.0, 2.0),
    feature_bandwidths=None,
    estimator="u",
):
    """Compute optionally conditional MMD with configurable U/V estimators."""
    if not isinstance(local, bool):
        raise ValueError("local must be a boolean")
    if estimator not in ("u", "v"):
        raise ValueError("estimator must be 'u' or 'v'")
    if feature_bandwidths is None:
        feature_bandwidth_multipliers = _validate_bandwidth_multipliers(
            feature_bandwidth_multipliers,
            "feature_bandwidth_multipliers",
        )
    else:
        feature_bandwidths = _validate_bandwidth_multipliers(
            feature_bandwidths,
            "feature_bandwidths",
        )
    if local:
        condition_bandwidth_multipliers = _validate_bandwidth_multipliers(
            condition_bandwidth_multipliers,
            "condition_bandwidth_multipliers",
        )

    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    cond = cond.reshape(cond.shape[0], -1)

    if x.shape[0] != y.shape[0] or x.shape[0] != cond.shape[0]:
        raise ValueError("x, y, and cond must have the same number of paired rows")

    # Filter paired rows before constructing any pairwise kernel matrix.
    finite_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    if local:
        finite_mask = finite_mask & torch.isfinite(cond).all(dim=1)
    x = x[finite_mask]
    y = y[finite_mask]
    cond = cond[finite_mask]

    if x.shape[0] == 0 or (estimator == "u" and x.shape[0] < 2):
        return (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).sum()
        ) * 0.0

    with torch.no_grad():
        if feature_bandwidths is None:
            feature_scale = _positive_median_pairwise_distance(y)
            feature_bandwidths = [value * feature_scale for value in feature_bandwidth_multipliers]
        if local:
            condition_scale = _positive_median_pairwise_distance(cond)
            condition_bandwidths = [
                value * condition_scale for value in condition_bandwidth_multipliers
            ]

    def pairwise_squared_distances(x, y):
        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * xy
        dxx = dxx.clamp_min(0.0)
        dyy = dyy.clamp_min(0.0)
        dxy = dxy.clamp_min(0.0)

        return dxx, dyy, dxy

    dxx, dyy, dxy = pairwise_squared_distances(x, y)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + TOR))

    def imq_kernel(a, d):
        return a**2 / (a**2 + d + TOR)

    kernels = {"rbf": rbf_kernel, "imq": imq_kernel}
    if feature_kernel not in kernels:
        raise ValueError(f"Unsupported feature kernel: {feature_kernel}")
    if local and condition_kernel not in kernels:
        raise ValueError(f"Unsupported condition kernel: {condition_kernel}")

    def mixed_kernel(kind, bandwidths, distances):
        kernel_fn = kernels[kind]
        return torch.stack(
            [kernel_fn(bandwidth, distances) for bandwidth in bandwidths],
            dim=0,
        ).mean(dim=0)

    kernel_xx = mixed_kernel(feature_kernel, feature_bandwidths, dxx)
    kernel_yy = mixed_kernel(feature_kernel, feature_bandwidths, dyy)
    kernel_xy = mixed_kernel(feature_kernel, feature_bandwidths, dxy)
    if local:
        cond_dxx, _, _ = pairwise_squared_distances(cond, cond)
        cond_matrix = mixed_kernel(condition_kernel, condition_bandwidths, cond_dxx)
        kernel_xx = kernel_xx * cond_matrix
        kernel_yy = kernel_yy * cond_matrix
        kernel_xy = kernel_xy * cond_matrix

    h = kernel_xx + kernel_yy - kernel_xy - kernel_xy.T
    if estimator == "v":
        return h.mean().clamp_min(0.0)
    off_diagonal = ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
    return h[off_diagonal].mean()
