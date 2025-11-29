import torch
import torch.nn.functional as F

# RBF kernel widths
SIGMA_LST = [0.05, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 50.0]


def compute_mmd(x, y, bandwidth_range=SIGMA_LST):
    """
    Maximum Mean Discrepancy (Gaussian kernel) between tensors x and y.
    x, y: (...,) or (N,) shaped tensors.
    """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy
    
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a**2)
        YY += torch.exp(-0.5 * dyy / a**2)
        XY += torch.exp(-0.5 * dxy / a**2)
        
    #     print("bandwidth: ", a)
    #     print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    
    return torch.mean(XX + YY - 2. * XY)


def invariant_mass(fourvec):
    """
    fourvec: (..., 4) where order is (px, py, pz, E)
    returns: (...,) positive mass = sqrt(|E^2 - |p|^2|)
    """
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    mass2 = E * E - (px * px + py * py + pz * pz)
    return torch.sqrt(torch.clamp(mass2.abs(), min=1e-10))


def mae_loss(y_true, y_pred):
    # do not consider mass targets in y_true
    return F.l1_loss(y_pred[..., :8], y_true[..., :8])


def neg_r2_loss(y_true, y_pred):
    y_t = y_true[..., :8]
    y_p = y_pred[..., :8]
    ss_res = torch.sum((y_t - y_p) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2)
    return ss_res / ss_tot - 1.0


def w_mass_mae_losses(y_true, y_pred):
    """
    Returns: (w0_mae, w1_mae) using true W masses at y_true[..., 8], y_true[..., 9]
    """
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = invariant_mass(w0_pred)
    w1_mass = invariant_mass(w1_pred)

    return (
        torch.mean(torch.abs(w0_mass - w0_true_mass)),
        torch.mean(torch.abs(w1_mass - w1_true_mass)),
    )


def w_mass_mmd_losses(y_true, y_pred):
    """
    Returns: (mmd_w0, mmd_w1) comparing predicted mass distributions to truth.
    """
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = invariant_mass(w0_pred)
    w1_mass = invariant_mass(w1_pred)

    return compute_mmd(w0_mass, w0_true_mass), compute_mmd(w1_mass, w1_true_mass)


def higgs_mass_loss(y_pred):
    """
    Higgs mass from sum of two W four-vectors; target 125.0 (GeV)
    """
    w0, w1 = y_pred[..., :4], y_pred[..., 4:8]
    higgs_4 = w0 + w1
    h_mass = invariant_mass(higgs_4)
    return torch.clamp(torch.mean(h_mass - 125.0).abs(), min=1e-10)


def nu_mass_loss(x_batch, y_pred):
    """
    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]
    penalize neutrino invariant mass (prefer ~0)
    """
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]

    nu0_mass = invariant_mass(n0_4)
    nu1_mass = invariant_mass(n1_4)
    return torch.mean(nu0_mass + nu1_mass)


def dinu_pt_loss(x_batch, y_pred):
    """
    Di-neutrino pT consistency with MET in inputs:
    x[..., 8] = MET px, x[..., 9] = MET py  (as in your TF code)
    """
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]
    nn_4 = n0_4 + n1_4
    nn_px_diff = torch.clamp((nn_4[..., 0] - x_batch[..., 8]).abs(), min=1e-10)
    nn_py_diff = torch.clamp((nn_4[..., 1] - x_batch[..., 9]).abs(), min=1e-10)
    return torch.mean(nn_px_diff + nn_py_diff)
