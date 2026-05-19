import torch
import torch.nn.functional as F
from physics.torchBoost import Booster

TOR = 1e-16
W_MASS_SCALE = 80.4
H_MASS_SCALE = 125.0
LOG_CUT = 20.0

def compute_mmd(x, y, kernel="imq", bandwidth_range=None):
    if bandwidth_range is None:
        bandwidth_range = [0.05, 0.1, 0.5, 1.0]
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    # Ignore samples with non-finite values
    # the full pairwise kernel matrix.
    finit_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    x = x[finit_mask]
    y = y[finit_mask]

    if x.shape[0] == 0 or y.shape[0] == 0:
        # TODO: check the stability 
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
        # print("Median distance: ", median_dist.item()) # debug
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
        # print("bandwidth: ", a)
        # print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    return torch.mean(XX + YY - 2. * XY)

def invariant_mass2(fourvec):
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    mass2 = E**2 - (px**2 + py**2 + pz**2)
    return torch.clamp(mass2, min=TOR)

def mae_loss(y_true, y_pred):
    # do not consider mass targets in y_true
    return F.l1_loss(y_pred[..., :8], y_true[..., :8])

def huber_loss(y_true, y_pred):
    # do not consider mass targets in y_true
    return F.huber_loss(y_pred[..., :8], y_true[..., :8])

def neg_r2_loss(y_true, y_pred):
    y_t = y_true[..., :8]
    y_p = y_pred[..., :8]
    ss_res = torch.sum((y_t - y_p) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2).clamp_min(TOR)
    return ss_res / ss_tot - 1.0


def w_4vec_construct(w_4vec_loge):
    w_3 = w_4vec_loge[..., :3]
    w_logE = torch.exp(torch.clamp(w_4vec_loge[..., 3], min=-LOG_CUT, max=LOG_CUT))
    return torch.cat([w_3, w_logE.reshape(-1, 1)], dim=-1)

def w_mass_mae_losses(y_true, y_pred):
    w0_pred, w1_pred = w_4vec_construct(y_pred[..., :4]), w_4vec_construct(y_pred[..., 4:8])
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)
    mass2_scale = W_MASS_SCALE**2
    return (
        F.l1_loss((w0_mass2 - w0_true_mass**2) / mass2_scale, torch.zeros_like(w0_mass2)),
        F.l1_loss((w1_mass2 - w1_true_mass**2) / mass2_scale, torch.zeros_like(w1_mass2))
    )

def w_mass_mmd_losses(y_true, y_pred):
    w0_pred, w1_pred = w_4vec_construct(y_pred[..., :4]), w_4vec_construct(y_pred[..., 4:8])
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)
    return compute_mmd(w0_mass2, w0_true_mass**2), compute_mmd(w1_mass2, w1_true_mass**2)

def higgs_mass_loss(y_pred):
    w0, w1 = w_4vec_construct(y_pred[..., :4]), w_4vec_construct(y_pred[..., 4:8])
    higgs_4 = w0 + w1
    h_mass2 = invariant_mass2(higgs_4).clamp_min(TOR)
    h_mass = torch.sqrt(h_mass2)
    return F.huber_loss(h_mass, torch.full_like(h_mass, H_MASS_SCALE))



def nu_mass_loss(x_batch, y_pred):
    """This is just a monitor loss"""
    n0_4 = w_4vec_construct(y_pred[..., :4]) - x_batch[..., :4]
    n1_4 = w_4vec_construct(y_pred[..., 4:8]) - x_batch[..., 4:8]

    nu0_mass2 = invariant_mass2(n0_4)
    nu1_mass2 = invariant_mass2(n1_4)
    return F.huber_loss(nu0_mass2, torch.zeros_like(nu0_mass2)) + F.huber_loss(nu1_mass2, torch.zeros_like(nu1_mass2))

def aux_mom_mmd_loss(y_true, y_pred):
    """This is just a monitor loss"""
    w0_pred, w1_pred = w_4vec_construct(y_pred[..., :4]), w_4vec_construct(y_pred[..., 4:8])
    w0_true, w1_true = w_4vec_construct(y_true[..., :4]), w_4vec_construct(y_true[..., 4:8])
    _sigma_lst = [0.1, 0.5, 1.0, 5.0]
    return compute_mmd(w0_pred, w0_true, bandwidth_range=_sigma_lst), compute_mmd(w1_pred, w1_true, bandwidth_range=_sigma_lst)

def dinu_pt_loss(x_batch, y_pred):
    n0_4 = w_4vec_construct(y_pred[..., :4]) - x_batch[..., :4]
    n1_4 = w_4vec_construct(y_pred[..., 4:8]) - x_batch[..., 4:8]
    nn_4 = n0_4 + n1_4
    dinu_pxpy = nn_4[..., :2]
    met_pxpy = x_batch[..., 16:18]
    return F.huber_loss(dinu_pxpy, met_pxpy)

def angular_loss_mmd(x_batch, y_true, y_pred):
    lep = x_batch[..., :8]
    true_w0, true_w1 = w_4vec_construct(y_true[..., :4]), w_4vec_construct(y_true[..., 4:8])
    pred_w0, pred_w1 = w_4vec_construct(y_pred[..., :4]), w_4vec_construct(y_pred[..., 4:8])
    true_w = torch.cat([true_w0, true_w1], dim=-1)
    pred_w = torch.cat([pred_w0, pred_w1], dim=-1)

    true_booster = Booster(lep, true_w)
    pred_booster = Booster(lep, pred_w)
    valid = true_booster.valid_rest_frame_mask() & pred_booster.valid_rest_frame_mask()

    true_ang = torch.stack(true_booster.lep_theta_phi_in_w_rest(), dim=-1)[valid]
    pred_ang = torch.stack(pred_booster.lep_theta_phi_in_w_rest(), dim=-1)[valid]
    if true_ang.shape[0] == 0:
        return (true_w.sum() + pred_w.sum()) * 0.0

    # theta_idx = [0, 3]
    # true_ang = true_ang.clone()
    # pred_ang = pred_ang.clone()
    # true_ang[..., theta_idx] = true_ang[..., theta_idx] / torch.pi
    # pred_ang[..., theta_idx] = pred_ang[..., theta_idx] / torch.pi

    _sigma_lst = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    # DEBUG
    # print("Shape of pred_ang: ", pred_ang.shape, "Shape of true_ang: ", true_ang.shape)
    return compute_mmd(pred_ang, true_ang, bandwidth_range=_sigma_lst)
