import torch
import torch.nn.functional as F

# RBF kernel widths
SIGMA_LST = [0.05, 0.1, 0.5, 1.0, 5.0]
TOR = 1e-16

def compute_mmd(x, y, bandwidth_range=SIGMA_LST):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        sel_dist = 1.0 if median_dist.item() == 0.0 else median_dist
        # print("Median distance: ", median_dist.item()) # debug
        # bandwidth_range = [0.03*median_dist, 0.07*median_dist, 0.3*median_dist, 0.7*median_dist, 3*median_dist, 7*median_dist]
        bandwidth_range = [s * sel_dist for s in bandwidth_range]
    
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
        XX += torch.exp(-0.5 * dxx / (a + TOR)**2)
        YY += torch.exp(-0.5 * dyy / (a + TOR)**2)
        XY += torch.exp(-0.5 * dxy / (a + TOR)**2)
        
    #     print("bandwidth: ", a)
    #     print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    
    return torch.mean(XX + YY - 2. * XY)


def invariant_mass2(fourvec):
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    mass2 = E**2 - (px**2 + py**2 + pz**2)
    return mass2


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
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2)
    return ss_res / ss_tot - 1.0


def w_mass_mae_losses(y_true, y_pred):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)
    return (
        F.huber_loss(w0_mass2, w0_true_mass**2),
        F.huber_loss(w1_mass2, w1_true_mass**2)
    )


def w_mass_mmd_losses(y_true, y_pred):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass2 = invariant_mass2(w0_pred)
    w1_mass2 = invariant_mass2(w1_pred)
    return compute_mmd(w0_mass2, w0_true_mass**2), compute_mmd(w1_mass2, w1_true_mass**2)


def higgs_mass_loss(y_pred):
    w0, w1 = y_pred[..., :4], y_pred[..., 4:8]
    higgs_4 = w0 + w1
    h_mass = torch.sqrt(invariant_mass2(higgs_4).abs()) # less likely < 0, so take abs() not square
    return F.huber_loss(h_mass, torch.full_like(h_mass, 125.0))


def nu_mass_loss(x_batch, y_pred):
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]

    nu0_mass2 = invariant_mass2(n0_4)
    nu1_mass2 = invariant_mass2(n1_4)
    # return torch.mean(nu0_mass2 + nu1_mass2)
    return F.huber_loss(nu0_mass2, torch.zeros_like(nu0_mass2)) + F.huber_loss(nu1_mass2, torch.zeros_like(nu1_mass2))

def aux_mom_mmd_loss(y_true, y_pred, epoch):
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true, w1_true = y_true[..., :4], y_true[..., 4:8]
    return compute_mmd(w0_pred, w0_true, [0.1, 0.5, 1.0, 5.0]), compute_mmd(w1_pred, w1_true, [0.1, 0.5, 1.0, 5.0])

def dinu_pt_loss(x_batch, y_pred):
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]
    nn_4 = n0_4 + n1_4
    # Penalize mismatch in the 2D MET vector magnitude (rotation-invariant in x-y plane).
    dpx = nn_4[..., 0] - x_batch[..., 20]
    dpy = nn_4[..., 1] - x_batch[..., 21]
    dpt = torch.sqrt(dpx**2 + dpy**2 + TOR)
    return F.huber_loss(dpt, torch.zeros_like(dpt))