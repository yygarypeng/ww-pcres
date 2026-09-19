import numpy as np

TOR = 1e-16
HIGGS_MASS = 125.0


def pt(px, py):
    return np.sqrt(px**2 + py**2)


def eta(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    return 0.5 * np.log((p + pz) / (p - pz + TOR))


def phi(px, py):
    return np.arctan2(py, px)


def invariant_mass2(fourvec):
    """Squared invariant mass of (..., 4) four-vectors ordered (px, py, pz, E)."""
    px, py, pz, energy = (fourvec[..., i] for i in range(4))
    return energy**2 - (px**2 + py**2 + pz**2)


def deta(eta1, eta2):
    return eta1 - eta2


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _sum_angle(angle0, angle1):
    return _wrap_angle(angle0 + angle1)


def _diff_angle(angle0, angle1):
    return _wrap_angle(angle0 - angle1)


def dphi(phi1, phi2):
    return _diff_angle(phi1, phi2)


def sphi(phi1, phi2):
    return _sum_angle(phi1, phi2)
