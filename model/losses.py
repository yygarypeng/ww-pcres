import math

import torch
import torch.nn.functional as F

from physics.torchBoost import Booster

####################
# Global constants #
####################

TOR = 1e-16
W_MASS_SCALE = 80.4
H_MASS_SCALE = 125.0


#############
# Utilities #
#############


def invariant_mass2(fourvec):
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]

    return E**2 - (px**2 + py**2 + pz**2)


def _valid_kinematic_rows(x_batch, y_true, y_pred):

    return (
        torch.isfinite(x_batch[..., :8]).all(dim=-1)
        & torch.isfinite(y_true[..., :8]).all(dim=-1)
        & torch.isfinite(y_pred).all(dim=-1)
    )


def _differentiable_zero(y_true, y_pred):

    return (
        torch.nan_to_num(y_true[..., :8], nan=0.0, posinf=0.0, neginf=0.0).sum()
        + torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0).sum()
    ) * 0.0


########################
# Feature construction #
########################


def _alpha_features(x_batch, w_fourvecs, slot0_on):
    lep0, lep1 = x_batch[..., :4], x_batch[..., 4:8]
    nu0 = w_fourvecs[..., :4] - lep0
    nu1 = w_fourvecs[..., 4:8] - lep1

    nu_on = torch.where(slot0_on.unsqueeze(-1), nu0, nu1)
    nu_off = torch.where(slot0_on.unsqueeze(-1), nu1, nu0)
    lep_on = torch.where(slot0_on.unsqueeze(-1), lep0, lep1)
    lep_off = torch.where(slot0_on.unsqueeze(-1), lep1, lep0)

    dinu = nu_on + nu_off
    on_mass2 = invariant_mass2(lep_on + dinu)
    off_mass2 = invariant_mass2(lep_off + dinu)
    p_on = torch.linalg.vector_norm(nu_on[..., :3], dim=-1)
    p_off = torch.linalg.vector_norm(nu_off[..., :3], dim=-1)
    total = p_on + p_off
    valid = (
        torch.isfinite(on_mass2)
        & torch.isfinite(off_mass2)
        & (on_mass2 >= 0.0)
        & (off_mass2 >= 0.0)
        & torch.isfinite(total)
        & (total > TOR)
    )
    selected_momentum = torch.where(on_mass2 > off_mass2, p_on, p_off)
    safe_total = torch.where(valid, total, torch.ones_like(total))

    return (2.0 * selected_momentum / safe_total - 1.0).unsqueeze(-1), valid


def _mass_features(w_fourvecs, center, scale):
    w_pos, w_neg = w_fourvecs[..., :4], w_fourvecs[..., 4:8]

    mass2 = torch.stack([invariant_mass2(w_pos), invariant_mass2(w_neg)], dim=-1)
    transformed = torch.asinh(mass2 / W_MASS_SCALE**2)

    return (transformed - center) / scale


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


def compute_mmd(x, y, *, kernel="imq", bandwidths=(0.1, 1.0, 10.0)):
    bandwidths = _validate_bandwidth_multipliers(bandwidths, "bandwidths")
    if kernel not in {"imq", "rbf"}:
        raise ValueError(f"Unsupported kernel: {kernel}")

    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of paired rows")

    finite = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    x = x[finite]
    y = y[finite]

    if x.shape[0] == 0:
        return (x.sum() + y.sum()) * 0.0

    x2 = x.square().sum(dim=1)
    y2 = y.square().sum(dim=1)
    dxx = (x2[:, None] + x2[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
    dyy = (y2[:, None] + y2[None, :] - 2.0 * (y @ y.T)).clamp_min(0.0)
    dxy = (x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)).clamp_min(0.0)

    def kernel_mean(distance, bandwidth):
        bandwidth2 = bandwidth**2

        if kernel == "imq":
            return (bandwidth2 / (bandwidth2 + distance + TOR)).mean()
        return torch.exp(-0.5 * distance / (bandwidth2 + TOR)).mean()

    mmd = x.new_zeros(())
    for bandwidth in bandwidths:
        mmd = mmd + kernel_mean(dxx, bandwidth)
        mmd = mmd + kernel_mean(dyy, bandwidth)
        mmd = mmd - 2.0 * kernel_mean(dxy, bandwidth)

    return (mmd / len(bandwidths)).clamp_min(0.0)


####################
# Loss functions
####################


def standardized_fourvec_huber_loss(y_true, y_pred, component_scales):
    true_fourvecs = y_true[..., :8].reshape(*y_true.shape[:-1], 2, 4)
    pred_fourvecs = y_pred.reshape(*y_pred.shape[:-1], 2, 4)

    residual = (pred_fourvecs - true_fourvecs) / component_scales

    return F.huber_loss(residual, torch.zeros_like(residual))


def w_mass_huber_loss(y_true, y_pred):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)

    w_lst_true = torch.stack([w0_true_mass**2, w1_true_mass**2], dim=-1) / W_MASS_SCALE**2
    w_lst_pred = torch.stack([w0_mass2, w1_mass2], dim=-1) / W_MASS_SCALE**2

    return F.huber_loss(w_lst_pred, w_lst_true)


def higgs_mass_loss(
    y_pred,
    target_mass=H_MASS_SCALE,
    scale=10.0,
    delta=2.0,
):
    w0_4, w1_4 = y_pred[..., :4], y_pred[..., 4:8]

    higgs_mass2 = invariant_mass2(w0_4 + w1_4)
    residual = (higgs_mass2 - target_mass**2) / (2.0 * target_mass * scale)

    return F.l1_loss(residual, torch.zeros_like(residual))


def dmet_loss(x_batch, y_true, dmet, component_scales):
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]
    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy

    residual = (dmet - dmet_target) / component_scales

    return F.huber_loss(residual, torch.zeros_like(residual))


####################
# Local MMD (WIP) #
####################


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
    """
    The default U-statistic excludes paired diagonal terms and can be negative.
    The V-statistic includes all terms and is nonnegative. Feature bandwidths
    may be fixed absolutely or inferred from y using the multiplier defaults.
    """
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
    finit_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    if local:
        finit_mask = finit_mask & torch.isfinite(cond).all(dim=1)
    x = x[finit_mask]
    y = y[finit_mask]
    cond = cond[finit_mask]

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

    def _matrix(x, y):
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

    dxx, dyy, dxy = _matrix(x, y)

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

    XX = mixed_kernel(feature_kernel, feature_bandwidths, dxx)
    YY = mixed_kernel(feature_kernel, feature_bandwidths, dyy)
    XY = mixed_kernel(feature_kernel, feature_bandwidths, dxy)
    if local:
        # The product of these normalized mixtures equals the mean over the
        # Cartesian product of all feature and condition bandwidths.
        cond_dxx, _, _ = _matrix(cond, cond)
        cond_matrix = mixed_kernel(condition_kernel, condition_bandwidths, cond_dxx)
        XX = XX * cond_matrix
        YY = YY * cond_matrix
        XY = XY * cond_matrix

    h = XX + YY - XY - XY.T
    if estimator == "v":
        return h.mean().clamp_min(0.0)
    off_diagonal = ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
    return h[off_diagonal].mean()


def alpha_mmd(x_batch, y_true, y_pred, cond, **mmd_kwargs):
    valid = _valid_kinematic_rows(x_batch, y_true, y_pred) & torch.isfinite(y_true[..., 8:10]).all(
        dim=-1
    )
    if not valid.any():
        return _differentiable_zero(y_true, y_pred)
    x_batch = x_batch[valid]
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    w_mass2 = W_MASS_SCALE**2
    slot0_on = torch.abs(y_true[..., 8] ** 2 - w_mass2) < torch.abs(y_true[..., 9] ** 2 - w_mass2)
    true_features, true_alpha_valid = _alpha_features(
        x_batch,
        y_true[..., :8],
        slot0_on,
    )
    pred_features, pred_alpha_valid = _alpha_features(x_batch, y_pred, slot0_on)
    alpha_valid = true_alpha_valid & pred_alpha_valid
    if not alpha_valid.any():
        return _differentiable_zero(y_true, y_pred)
    return compute_mmd(
        pred_features[alpha_valid],
        true_features[alpha_valid],
        **mmd_kwargs,
    )


def mass_mmd(x_batch, y_true, y_pred, cond, center, scale, **mmd_kwargs):
    valid = _valid_kinematic_rows(x_batch, y_true, y_pred)
    if not valid.any():
        return _differentiable_zero(y_true, y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    true_features = _mass_features(y_true[..., :8], center, scale)
    pred_features = _mass_features(y_pred, center, scale)
    return compute_mmd(pred_features, true_features, **mmd_kwargs)


def angular_mmd(x_batch, y_true, y_pred, cond, **mmd_kwargs):
    lep = x_batch[..., :8]
    true_w0, true_w1 = y_true[..., :4], y_true[..., 4:8]
    pred_w0, pred_w1 = y_pred[..., :4], y_pred[..., 4:8]
    true_w = torch.cat([true_w0, true_w1], dim=-1)
    pred_w = torch.cat([pred_w0, pred_w1], dim=-1)

    with torch.no_grad():
        true_booster = Booster(lep, true_w)
        true_valid, true_ang = true_booster.lep_theta_phi_with_validity()
    pred_booster = Booster(lep, pred_w)
    pred_valid, pred_ang = pred_booster.lep_theta_phi_with_validity()
    valid = true_valid & pred_valid
    true_ang = true_ang[valid]
    pred_ang = pred_ang[valid]
    if true_ang.shape[0] == 0:
        return (torch.nan_to_num(true_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum() + (
            torch.nan_to_num(pred_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        ).sum()

    true_ang = angular_mmd_features(true_ang)
    pred_ang = angular_mmd_features(pred_ang)

    return compute_mmd(pred_ang, true_ang, **mmd_kwargs)
