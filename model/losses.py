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


def transform_mmd_loss(mmd2, *, kind=None, epsilon=1.0e-3):
    if kind is None:
        return mmd2
    if kind != "sqrt":
        raise ValueError(f"unsupported MMD loss transform: {kind}")
    epsilon_tensor = mmd2.new_tensor(epsilon)
    return torch.sqrt(mmd2.clamp_min(0.0) + epsilon_tensor.square()) - epsilon_tensor


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


def _require_mmd_condition(cond, name):
    if cond.shape[-1] == 0:
        raise ValueError(f"{name} requires the four high-level conditioning features")


def _valid_kinematic_rows(x_batch, y_true, y_pred, cond, local=True):
    valid = (
        torch.isfinite(x_batch[..., :8]).all(dim=-1)
        & torch.isfinite(y_true[..., :8]).all(dim=-1)
        & torch.isfinite(y_pred).all(dim=-1)
    )
    return valid & torch.isfinite(cond).all(dim=-1) if local else valid


def _differentiable_zero(y_true, y_pred, cond):
    return (
        torch.nan_to_num(y_true[..., :8], nan=0.0, posinf=0.0, neginf=0.0).sum()
        + torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0).sum()
        + torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).sum()
    ) * 0.0


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


def higgs_mass_loss(
    y_pred,
    target_mass=H_MASS_SCALE,
    scale=10.0,
    delta=2.0,
):
    w0_4, w1_4 = y_pred[..., :4], y_pred[..., 4:8]
    higgs_mass2 = invariant_mass2(w0_4 + w1_4)
    residual = (higgs_mass2 - target_mass**2) / (2.0 * target_mass * scale)
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta)


def dmet_loss(x_batch, y_true, dmet, component_scales):
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]

    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy
    residual = (dmet - dmet_target) / component_scales

    return F.huber_loss(residual, torch.zeros_like(residual))


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
):
    """Biased feature MMD, optionally localized by a condition kernel."""
    if not isinstance(local, bool):
        raise ValueError("local must be a boolean")
    feature_bandwidth_multipliers = _validate_bandwidth_multipliers(
        feature_bandwidth_multipliers,
        "feature_bandwidth_multipliers",
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

    if x.shape[0] == 0 or y.shape[0] == 0:
        return (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).sum()
        ) * 0.0

    with torch.no_grad():
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
    return torch.mean(XX + YY - XY - XY.T)


def alpha_mmd(x_batch, y_true, y_pred, cond, **mmd_kwargs):
    _require_mmd_condition(cond, "alpha MMD")

    valid = _valid_kinematic_rows(
        x_batch,
        y_true,
        y_pred,
        cond,
        local=mmd_kwargs.get("local", True),
    ) & torch.isfinite(y_true[..., 8:10]).all(dim=-1)
    if not valid.any():
        return _differentiable_zero(y_true, y_pred, cond)
    x_batch = x_batch[valid]
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    cond = cond[valid]

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
        return _differentiable_zero(y_true, y_pred, cond)
    return compute_local_mmd(
        pred_features[alpha_valid],
        true_features[alpha_valid],
        cond[alpha_valid],
        **mmd_kwargs,
    )


def mass_mmd(x_batch, y_true, y_pred, cond, center, scale, **mmd_kwargs):
    _require_mmd_condition(cond, "mass MMD")

    valid = _valid_kinematic_rows(
        x_batch,
        y_true,
        y_pred,
        cond,
        local=mmd_kwargs.get("local", True),
    )
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

        return torch.stack(
            [
                2.0 * theta0 / torch.pi - 1.0,
                torch.sin(phi0),
                torch.cos(phi0),
                2.0 * theta1 / torch.pi - 1.0,
                torch.sin(phi1),
                torch.cos(phi1),
            ],
            dim=-1,
        )

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
        return (torch.nan_to_num(true_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum() + (
            torch.nan_to_num(pred_w, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        ).sum()

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
    return F.huber_loss(nu0_mass2, torch.zeros_like(nu0_mass2)) + F.huber_loss(
        nu1_mass2, torch.zeros_like(nu1_mass2)
    )
