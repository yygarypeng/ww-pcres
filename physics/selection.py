"""Analysis selection on the raw 18-column PCRes inputs.

The cuts live here rather than in the notebook so they can be unit tested and reused
by the evaluation scripts. They are applied after inference: one pass over the test
set feeds every selection, and a cut can be retuned without re-running the model.
"""

import numpy as np

from physics.physics import dphi, phi, pt

# Raw input layout: lep+ 0:4, lep- 4:8, jet0 8:12, jet1 12:16, MET 16:18, with every
# four-vector ordered (px, py, pz, E) in GeV. Padding jets are all-zero rows.
LEPTON_COLUMNS = slice(0, 8)
JET_ENERGY_COLUMNS = [11, 15]
MET_COLUMNS = slice(16, 18)

LEAD_LEPTON_PT_MIN = 22.0  # GeV
SUBLEAD_LEPTON_PT_MIN = 15.0  # GeV
DILEPTON_MASS_MIN = 10.0  # GeV
DILEPTON_MASS_MAX = 55.0  # GeV
MET_MIN = 20.0  # GeV
DILEPTON_DPHI_MAX = 2.0  # rad
DILEPTON_MET_DPHI_MIN = 1.57  # rad

def _leptons(features):
    """The lepton block as (events, {+, -}, {px, py, pz, E})."""
    return np.asarray(features)[:, LEPTON_COLUMNS].reshape(-1, 2, 4)


def lepton_pts(features):
    """Per-event (leading, subleading) lepton pT, ordered by pT rather than by charge."""
    leptons = _leptons(features)
    pts = pt(leptons[..., 0], leptons[..., 1])
    return pts.max(axis=1), pts.min(axis=1)


def dilepton_mass(features):
    """Invariant mass of the lepton pair; the absolute value keeps rounding real."""
    dilepton = _leptons(features).sum(axis=1)
    mass2 = dilepton[:, 3] ** 2 - np.sum(dilepton[:, :3] ** 2, axis=-1)
    return np.sqrt(np.abs(mass2))


def dilepton_dphi(features):
    """Unsigned azimuthal opening angle between the two leptons, in [0, pi]."""
    leptons = _leptons(features)
    lepton_phis = phi(leptons[..., 0], leptons[..., 1])
    return np.abs(dphi(lepton_phis[:, 0], lepton_phis[:, 1]))


def missing_et(features):
    """Missing transverse energy from the MET two-vector."""
    met = np.asarray(features)[:, MET_COLUMNS]
    return pt(met[:, 0], met[:, 1])


def dilepton_met_dphi(features):
    """Unsigned azimuthal angle between the dilepton system and MET, in [0, pi]."""
    dilepton = _leptons(features).sum(axis=1)
    met = np.asarray(features)[:, MET_COLUMNS]
    return np.abs(dphi(phi(dilepton[:, 0], dilepton[:, 1]), phi(met[:, 0], met[:, 1])))


def jet_multiplicity(features):
    """Number of real jets; a zero jet energy tags a padded slot."""
    return (np.asarray(features)[:, JET_ENERGY_COLUMNS] > 0.0).sum(axis=1)


def selection_masks(features, njets=None):
    """Per-cut boolean masks in cutflow order.

    `njets` of 0 / 1 / 2 keeps only that jet multiplicity; None keeps every event.
    """
    features = np.asarray(features)
    lead_pt, sublead_pt = lepton_pts(features)
    mll = dilepton_mass(features)
    n_jets_mask = (
        np.ones(len(features), dtype=bool) if njets is None else jet_multiplicity(features) == njets
    )
    return {
        "l0_pt": lead_pt > LEAD_LEPTON_PT_MIN,
        "l1_pt": sublead_pt > SUBLEAD_LEPTON_PT_MIN,
        "mll": (mll > DILEPTON_MASS_MIN) & (mll < DILEPTON_MASS_MAX),
        "met": missing_et(features) > MET_MIN,
        "dphi_ll": dilepton_dphi(features) < DILEPTON_DPHI_MAX,
        "dphi_ll_met": dilepton_met_dphi(features) > DILEPTON_MET_DPHI_MIN,
        "n_jets": n_jets_mask,
    }
