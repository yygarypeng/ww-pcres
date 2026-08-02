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

def compute_mmd(x, y, kernel="imq", bandwidth_range=None):
    if bandwidth_range is None:
        bandwidth_range = [0.01, 0.1, 1.0]
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    # Ignore samples with non-finite values
    # the full pairwise kernel matrix.
    finit_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    x = x[finit_mask]
    y = y[finit_mask]

    if x.shape[0] == 0 or y.shape[0] == 0:
        return (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).sum()
            + torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).sum()
        ) * 0.0

    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        sel_dist = torch.where(
            median_dist == 0.0,
            torch.ones_like(median_dist),
            median_dist,
        )
        bandwidth_range = [s * sel_dist for s in bandwidth_range]

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy
    dxx = dxx.clamp_min(0.0)
    dyy = dyy.clamp_min(0.0)
    dxy = dxy.clamp_min(0.0)

    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + TOR))
    def imq_kernel(a, d):
        return a**2 / (a**2 + d + TOR)

    if kernel == "rbf":
        _ker = lambda a, d: rbf_kernel(a, d)
    elif kernel == "imq":
        _ker = lambda a, d: imq_kernel(a, d)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    for a in bandwidth_range:
        XX += _ker(a, dxx)
        YY += _ker(a, dyy)
        XY += _ker(a, dxy)
    return torch.mean(XX + YY - 2. * XY)

def compute_local_mmd(x, y, cond, kernel="imq", base_bandwidth_range=None):
    if base_bandwidth_range is None:
        base_bandwidth_range = [0.05, 0.1, 0.5, 1.0]
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    # Ignore samples with non-finite values
    # the full pairwise kernel matrix.
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

    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        sel_dist = torch.where(
            median_dist == 0.0,
            torch.ones_like(median_dist),
            median_dist,
        )
        bandwidth_range = [s * sel_dist for s in base_bandwidth_range]

    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(cond, cond, p=2)
        median_dist = torch.median(dists)
        sel_dist = torch.where(
            median_dist == 0.0,
            torch.ones_like(median_dist),
            median_dist,
        )
        cond_bandwidth_range = [s * sel_dist for s in base_bandwidth_range]

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

        return xx, yy, xy, dxx, dyy, dxy

    xx, yy, xy, dxx, dyy, dxy = _matrix(x, y)
    _, _, _, cond_dxx, cond_dyy, cond_dxy = _matrix(cond, cond)

    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    def rbf_kernel(a, d):
        return torch.exp(-0.5 * d / (a**2 + TOR))
    def imq_kernel(a, d):
        return a**2 / (a**2 + d + TOR)

    if kernel == "rbf":
        _ker = lambda a, d: rbf_kernel(a, d)
    elif kernel == "imq":
        _ker = lambda a, d: imq_kernel(a, d)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    for a, b in zip(bandwidth_range, cond_bandwidth_range):
        XX += _ker(a, dxx) * _ker(b, cond_dxx)
        YY += _ker(a, dyy) * _ker(b, cond_dyy)
        XY += _ker(a, dxy) * _ker(b, cond_dxy)
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

def w_mass_losses(y_true, y_pred, *, include_mmd=True, include_huber=True):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)

    w_lst_true = torch.stack([w0_true_mass**2, w1_true_mass**2], dim=-1) / W_MASS_SCALE**2
    w_lst_pred = torch.stack([w0_mass2, w1_mass2], dim=-1) / W_MASS_SCALE**2
    mmd = compute_mmd(w_lst_pred, w_lst_true) if include_mmd else None
    huber = F.huber_loss(w_lst_pred, w_lst_true) if include_huber else None
    return mmd, huber

def higgs_mass_loss(y_pred):
    w0_4, w1_4 = y_pred[..., :4], y_pred[..., 4:8]

    higgs_4 = w0_4 + w1_4
    h_mass2 = invariant_mass2(higgs_4)
    h_mass = torch.sqrt(torch.clamp(h_mass2, min=TOR))
    # causal_penalty = F.relu(-h_mass2) / H_MASS_SCALE

    # return F.huber_loss(h_mass , torch.full_like(h_mass, H_MASS_SCALE)) + causal_penalty.mean()
    return F.huber_loss(h_mass , torch.full_like(h_mass, H_MASS_SCALE), delta=1.0)

def dmet_loss(x_batch, y_true, dmet, component_scales):
    true_w0 = y_true[..., :4]
    true_w1 = y_true[..., 4:8]

    true_nu0 = true_w0 - x_batch[..., :4]
    true_nu1 = true_w1 - x_batch[..., 4:8]
    true_dinu_pxpy = true_nu0[..., :2] + true_nu1[..., :2]
    dmet_target = x_batch[..., 16:18] - true_dinu_pxpy
    residual = (dmet - dmet_target) / component_scales

    return F.huber_loss(residual, torch.zeros_like(residual))

def angular_loss_mmd(x_batch, y_true, y_pred, cond):
    if cond.shape[-1] == 0:
        raise ValueError("angular local MMD requires the four high-level conditioning features")

    def _features_for_mmd(angles):
        theta0 = angles[..., 0]
        phi0 = angles[..., 1]
        theta1 = angles[..., 2]
        phi1 = angles[..., 3]

        return torch.stack([
            theta0 / torch.pi,
            torch.sin(phi0), torch.cos(phi0),
            theta1 / torch.pi,
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
    _sigma_lst = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]
    return compute_local_mmd(pred_ang, true_ang, cond=cond, base_bandwidth_range=_sigma_lst)

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
