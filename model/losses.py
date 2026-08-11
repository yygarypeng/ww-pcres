import math

import torch
import torch.nn.functional as F
from physics.torchBoost import Booster

######################
## Global constants ##
######################

TOR = 1e-16
W_MASS_SCALE = 80.4
H_MASS_SCALE = 125.0

###############
## Utilities ##
###############

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
    feature_kernel="imq",
    condition_kernel="rbf",
    feature_bandwidth_multipliers=(0.25, 0.5, 1.0, 2.0),
    condition_bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Biased MMD on output features, localized by a separate condition kernel."""
    feature_bandwidth_multipliers = _validate_bandwidth_multipliers(
        feature_bandwidth_multipliers,
        "feature_bandwidth_multipliers",
    )
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
    finit_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1) & torch.isfinite(cond).all(dim=1)
    x = x[finit_mask]
    y = y[finit_mask]
    cond = cond[finit_mask]

    if x.shape[0] == 0 or y.shape[0] == 0 or cond.shape[0] == 0:
        return (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).sum()
        ) * 0.0

    with torch.no_grad():
        feature_scale = _positive_median_pairwise_distance(y)
        condition_scale = _positive_median_pairwise_distance(cond)
        feature_bandwidths = [value * feature_scale for value in feature_bandwidth_multipliers]
        condition_bandwidths = [value * condition_scale for value in condition_bandwidth_multipliers]

    def _matrix(x, y):
        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * xy
        dxx = dxx.clamp_min(0.0)
        dyy = dyy.clamp_min(0.0)
        dxy = dxy.clamp_min(0.0)

        return dxx, dyy, dxy

    dxx, dyy, dxy = _matrix(x, y)
    cond_dxx, _, _ = _matrix(cond, cond)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + TOR))
    def imq_kernel(a, d):
        return a**2 / (a**2 + d + TOR)

    kernels = {"rbf": rbf_kernel, "imq": imq_kernel}
    if feature_kernel not in kernels:
        raise ValueError(f"Unsupported feature kernel: {feature_kernel}")
    if condition_kernel not in kernels:
        raise ValueError(f"Unsupported condition kernel: {condition_kernel}")

    def mixed_kernel(kind, bandwidths, distances):
        kernel_fn = kernels[kind]
        return torch.stack(
            [kernel_fn(bandwidth, distances) for bandwidth in bandwidths],
            dim=0,
        ).mean(dim=0)

    # The product of these two normalized mixtures equals the mean over the
    # Cartesian product of all feature and condition bandwidths.
    cond_matrix = mixed_kernel(condition_kernel, condition_bandwidths, cond_dxx)
    XX = mixed_kernel(feature_kernel, feature_bandwidths, dxx) * cond_matrix
    YY = mixed_kernel(feature_kernel, feature_bandwidths, dyy) * cond_matrix
    XY = mixed_kernel(feature_kernel, feature_bandwidths, dxy) * cond_matrix
    return torch.mean(XX + YY - XY - XY.T)

def invariant_mass2(fourvec):
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    return E**2 - (px**2 + py**2 + pz**2)


def standardized_fourvec_huber_loss(y_true, y_pred, component_scales):
    true_fourvecs = y_true[..., :8].reshape(*y_true.shape[:-1], 2, 4)
    pred_fourvecs = y_pred.reshape(*y_pred.shape[:-1], 2, 4)
    true_transformed = torch.cat(
        [true_fourvecs[..., :3], torch.log1p(true_fourvecs[..., 3:4])],
        dim=-1,
    )
    pred_transformed = torch.cat(
        [pred_fourvecs[..., :3], torch.log1p(pred_fourvecs[..., 3:4])],
        dim=-1,
    )
    residual = (pred_transformed - true_transformed) / component_scales
    return F.huber_loss(residual, torch.zeros_like(residual))

####################
## Loss functions ##
####################

def w_mass_huber_loss(y_true, y_pred):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)

    w_lst_true = torch.stack([w0_true_mass**2, w1_true_mass**2], dim=-1) / W_MASS_SCALE**2
    w_lst_pred = torch.stack([w0_mass2, w1_mass2], dim=-1) / W_MASS_SCALE**2
    return F.huber_loss(w_lst_pred, w_lst_true)

def higgs_mass_loss(y_pred, delta=2):
    w0_4, w1_4 = y_pred[..., :4], y_pred[..., 4:8]

    higgs_4 = w0_4 + w1_4
    h_mass2 = invariant_mass2(higgs_4)
    h_mass = torch.sqrt(torch.clamp(h_mass2, min=TOR))

    return F.huber_loss(h_mass , torch.full_like(h_mass, H_MASS_SCALE), delta=delta)
    # return F.l1_loss(h_mass , torch.full_like(h_mass, H_MASS_SCALE))

def dmet_loss(x_batch, y_true, dmet, component_scales):
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]

    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy
    residual = (dmet - dmet_target) / component_scales

    return F.huber_loss(residual, torch.zeros_like(residual))


def _require_mmd_condition(cond, name):
    if cond.shape[-1] == 0:
        raise ValueError(f"{name} requires the four high-level conditioning features")


def _valid_kinematic_rows(x_batch, y_true, y_pred, cond):
    return (
        torch.isfinite(x_batch[..., :8]).all(dim=-1)
        & torch.isfinite(y_true[..., :8]).all(dim=-1)
        & torch.isfinite(y_pred).all(dim=-1)
        & torch.isfinite(cond).all(dim=-1)
    )


def _differentiable_zero(y_true, y_pred, cond):
    return (
        torch.nan_to_num(y_true[..., :8], nan=0.0, posinf=0.0, neginf=0.0).sum()
        + torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0).sum()
        + torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).sum()
    ) * 0.0


def _alpha_features(x_batch, w_fourvecs):
    w_pos, w_neg = w_fourvecs[..., :4], w_fourvecs[..., 4:8]
    nu_pos = w_pos - x_batch[..., :4]
    nu_neg = w_neg - x_batch[..., 4:8]
    p_pos = torch.linalg.vector_norm(nu_pos[..., :3], dim=-1)
    p_neg = torch.linalg.vector_norm(nu_neg[..., :3], dim=-1)
    total = p_pos + p_neg
    safe_total = torch.where(total == 0.0, torch.ones_like(total), total)
    alpha_pos = torch.where(total == 0.0, torch.full_like(total, 0.5), p_pos / safe_total)
    return (2.0 * alpha_pos - 1.0).unsqueeze(-1)


def _mass_features(w_fourvecs, center, scale):
    w_pos, w_neg = w_fourvecs[..., :4], w_fourvecs[..., 4:8]
    mass2 = torch.stack([invariant_mass2(w_pos), invariant_mass2(w_neg)], dim=-1)
    transformed = torch.asinh(mass2 / W_MASS_SCALE**2)
    return (transformed - center) / scale


def alpha_mmd(x_batch, y_true, y_pred, cond, **mmd_kwargs):
    _require_mmd_condition(cond, "alpha MMD")

    valid = _valid_kinematic_rows(x_batch, y_true, y_pred, cond)
    if not valid.any():
        return _differentiable_zero(y_true, y_pred, cond)
    x_batch = x_batch[valid]
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    cond = cond[valid]

    true_features = _alpha_features(x_batch, y_true[..., :8])
    pred_features = _alpha_features(x_batch, y_pred)
    return compute_local_mmd(pred_features, true_features, cond, **mmd_kwargs)


def mass_mmd(x_batch, y_true, y_pred, cond, center, scale, **mmd_kwargs):
    _require_mmd_condition(cond, "mass MMD")

    valid = _valid_kinematic_rows(x_batch, y_true, y_pred, cond)
    if not valid.any():
        return _differentiable_zero(y_true, y_pred, cond)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    cond = cond[valid]

    true_features = _mass_features(y_true[..., :8], center, scale)
    pred_features = _mass_features(y_pred, center, scale)
    return compute_local_mmd(pred_features, true_features, cond, **mmd_kwargs)

def angular_mmd(x_batch, y_true, y_pred, cond, **mmd_kwargs):
    _require_mmd_condition(cond, "angular MMD")

    def _features_for_mmd(angles):
        theta0 = angles[..., 0]
        phi0 = angles[..., 1]
        theta1 = angles[..., 2]
        phi1 = angles[..., 3]

        return torch.stack([
            2.0 * theta0 / torch.pi - 1.0,
            torch.sin(phi0), torch.cos(phi0),
            2.0 * theta1 / torch.pi - 1.0,
            torch.sin(phi1), torch.cos(phi1),
        ], dim=-1)

    lep = x_batch[..., :8]
    true_w0, true_w1 = y_true[..., :4], y_true[..., 4:8]
    pred_w0, pred_w1 = y_pred[..., :4], y_pred[..., 4:8]
    true_w = torch.cat([true_w0, true_w1], dim=-1)
    pred_w = torch.cat([pred_w0, pred_w1], dim=-1)

    true_booster = Booster(lep, true_w)
    pred_booster = Booster(lep, pred_w)
    valid = true_booster.valid_rest_frame_mask() & pred_booster.valid_rest_frame_mask()

    true_ang = torch.stack(true_booster.lep_theta_phi_in_w_rest(), dim=-1)[valid]
    pred_ang = torch.stack(pred_booster.lep_theta_phi_in_w_rest(), dim=-1)[valid]
    if true_ang.shape[0] == 0:
        return (
            (torch.nan_to_num(true_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()
            + (torch.nan_to_num(pred_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()
        )

    true_ang = _features_for_mmd(true_ang)
    pred_ang = _features_for_mmd(pred_ang)

    cond = cond[valid]
    return compute_local_mmd(pred_ang, true_ang, cond, **mmd_kwargs)

#############################
## Archived loss functions ##
#############################

def neg_r2_loss(y_true, y_pred):
    y_t = y_true[..., :8]
    y_p = y_pred[..., :8]

    ss_res = torch.sum((y_t - y_p) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2).clamp_min(TOR)
    return ss_res / ss_tot - 1.0

def nu_mass_loss(x_batch, y_pred):
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]

    nu0_mass2 = invariant_mass2(n0_4)
    nu1_mass2 = invariant_mass2(n1_4)
    return F.huber_loss(nu0_mass2, torch.zeros_like(nu0_mass2)) + F.huber_loss(nu1_mass2, torch.zeros_like(nu1_mass2))
