import torch
import torch.nn.functional as F

from physics.physics import HIGGS_MASS, TOR, invariant_mass2
from physics.torchBoost import Booster

W_MASS_SCALE = 80.4
H_MASS_SCALE = HIGGS_MASS


def mass_residual(fourvec, target_mass2, reference_mass):
    return (invariant_mass2(fourvec) - target_mass2) / (2.0 * reference_mass)


def _valid_kinematic_rows(x_batch, y_true, y_pred):
    return (
        torch.isfinite(x_batch[..., :8]).all(dim=-1)
        & torch.isfinite(y_true[..., :8]).all(dim=-1)
        & torch.isfinite(y_pred).all(dim=-1)
    )


def _sanitized_rows(tensor, keep):
    return torch.where(keep.unsqueeze(-1), tensor, torch.zeros_like(tensor))


def mass_mmd_features(w_fourvecs):
    mass2 = torch.stack(
        [invariant_mass2(w_fourvecs[..., :4]), invariant_mass2(w_fourvecs[..., 4:8])],
        dim=-1,
    )
    return torch.asinh(mass2 / W_MASS_SCALE**2)


def angular_mmd_features(angles):
    return torch.stack(
        [
            2.0 * angles[..., 0] / torch.pi - 1.0,
            torch.sin(angles[..., 1]),
            torch.cos(angles[..., 1]),
            2.0 * angles[..., 2] / torch.pi - 1.0,
            torch.sin(angles[..., 3]),
            torch.cos(angles[..., 3]),
        ],
        dim=-1,
    )


FOURVEC_LOSSES = ("l1", "huber", "rmse")


def w_fourvec_loss(y_true, y_pred, kind="l1", huber_delta=10.0):
    """Distance between the predicted and true W four-vectors, in GeV.

    ``l1`` is median-seeking, so it biases the skewed energies low; ``rmse`` is
    mean-seeking, and ``huber`` matches L1 above ``huber_delta``.
    """
    target = y_true[..., :8]
    if kind == "l1":
        return F.l1_loss(y_pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(y_pred, target, beta=huber_delta)
    if kind == "rmse":
        return torch.sqrt(F.mse_loss(y_pred, target).clamp_min(TOR))
    raise ValueError(f"fourvec_loss must be one of {FOURVEC_LOSSES}, got {kind!r}")


def mean_residual_penalty(y_true, y_pred):
    """Penalize a shared component offset in units of residual spread."""
    residual = y_pred - y_true[..., :8]
    scale = residual.detach().std(dim=0).clamp_min(TOR)
    return torch.mean((residual.mean(dim=0) / scale) ** 2)


def w_mass_loss(y_true, y_pred):
    residuals = torch.stack(
        [
            mass_residual(y_pred[..., :4], y_true[..., 8] ** 2, W_MASS_SCALE),
            mass_residual(y_pred[..., 4:8], y_true[..., 9] ** 2, W_MASS_SCALE),
        ],
        dim=-1,
    )
    return residuals.abs().mean()


def dmet_loss(x_batch, y_true, dmet):
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]
    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy

    return F.l1_loss(dmet, dmet_target)


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
        dyy_term = kernel_mean(dyy, bandwidth)
        mmd = mmd + kernel_mean(dxx, bandwidth) + dyy_term
        mmd = mmd - 2.0 * kernel_mean(dxy, bandwidth)

    return torch.sqrt((mmd / len(bandwidths)).clamp_min(TOR))


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


def higgs_fourvec_loss(y_true, y_pred):
    """Mean L1 loss on the summed W four-vectors in GeV."""
    higgs_true = y_true[..., :4] + y_true[..., 4:8]
    higgs_pred = y_pred[..., :4] + y_pred[..., 4:8]
    return F.l1_loss(higgs_pred, higgs_true)


def higgs_mass_loss(y_pred, target_mass=HIGGS_MASS):
    """Mean L1 loss on the Higgs mass residual in GeV."""
    higgs = y_pred[..., :4] + y_pred[..., 4:8]
    return mass_residual(higgs, target_mass**2, target_mass).abs().mean()
