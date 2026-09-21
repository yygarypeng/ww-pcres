import numpy as np
import pytest

from physics.selection import (
    DILEPTON_MASS_MAX,
    DILEPTON_MASS_MIN,
    LEAD_LEPTON_PT_MIN,
    MET_MIN,
    SUBLEAD_LEPTON_PT_MIN,
    dilepton_dphi,
    dilepton_mass,
    dilepton_met_dphi,
    jet_multiplicity,
    missing_et,
    selection_masks,
)


def massless(pt_value, phi_value=0.0):
    """A massless four-vector (px, py, pz, E) with pz = 0."""
    px = pt_value * np.cos(phi_value)
    py = pt_value * np.sin(phi_value)
    return [px, py, 0.0, pt_value]


def make_event(pos_lep, neg_lep, jets=(), met=(0.0, 0.0)):
    """One raw 18-column row; unfilled jet slots stay zero-padded."""
    row = list(pos_lep) + list(neg_lep)
    for slot in range(2):
        row += list(jets[slot]) if slot < len(jets) else [0.0] * 4
    return row + list(met)


def make_features(*events):
    return np.array(events, dtype=np.float64)


def back_to_back(pt_pos, pt_neg, **kwargs):
    """Opposite-hemisphere massless leptons of given pT, so m_ll = 2 sqrt(pt+ pt-)."""
    return make_event(massless(pt_pos, 0.0), massless(pt_neg, np.pi), **kwargs)


def passing_event(**overrides):
    """An event comfortably inside every cut, for overriding one quantity at a time."""
    defaults = {
        "pos_lep": massless(30.0, 0.0),
        "neg_lep": massless(20.0, 1.0),
        "met": (0.0, -40.0),
    }
    return make_event(**{**defaults, **overrides})


def test_lepton_pt_cuts_order_by_pt_not_by_charge():
    features = make_features(
        back_to_back(30.0, 18.0),  # leading is the positive lepton
        back_to_back(18.0, 30.0),  # leading is the negative lepton
        back_to_back(20.0, 18.0),  # no lepton above the leading threshold
        back_to_back(30.0, 12.0),  # subleading too soft
    )
    masks = selection_masks(features)

    assert masks["l0_pt"].tolist() == [True, True, False, True]
    assert masks["l1_pt"].tolist() == [True, True, True, False]


def test_lepton_pt_thresholds_are_strict():
    features = make_features(
        back_to_back(LEAD_LEPTON_PT_MIN, 18.0),
        back_to_back(30.0, SUBLEAD_LEPTON_PT_MIN),
    )
    masks = selection_masks(features)

    assert masks["l0_pt"].tolist() == [False, True]
    assert masks["l1_pt"].tolist() == [True, False]


def test_dilepton_mass_cut_is_a_window():
    collinear = make_event(massless(30.0, 0.0), massless(30.0, 0.0))
    features = make_features(
        collinear,  # hard leptons, but m_ll = 0
        back_to_back(4.0, 4.0),  # m_ll = 8 GeV, below the window
        back_to_back(6.0, 6.0),  # m_ll = 12 GeV, inside
        back_to_back(20.0, 20.0),  # m_ll = 40 GeV, inside
        back_to_back(40.0, 40.0),  # m_ll = 80 GeV, above the window
    )

    assert dilepton_mass(features) == pytest.approx([0.0, 8.0, 12.0, 40.0, 80.0])
    assert selection_masks(features)["mll"].tolist() == [False, False, True, True, False]
    assert (DILEPTON_MASS_MIN, DILEPTON_MASS_MAX) == (10.0, 55.0)


def test_pre_dilepton_mass_cut_is_the_lower_edge_only():
    features = make_features(
        make_event(massless(30.0, 0.0), massless(30.0, 0.0)),  # m_ll = 0
        back_to_back(4.0, 4.0),  # m_ll = 8 GeV, below the window
        back_to_back(20.0, 20.0),  # m_ll = 40 GeV, inside
        back_to_back(40.0, 40.0),  # m_ll = 80 GeV, above the window
    )

    # pre_mll drops the low-mass resonances but keeps everything above them.
    assert selection_masks(features)["pre_mll"].tolist() == [False, False, True, True]


def test_missing_et_cut_uses_the_met_two_vector():
    features = make_features(
        passing_event(met=(30.0, 40.0)),  # 50 GeV
        passing_event(met=(MET_MIN, 0.0)),  # exactly at the threshold, strict cut
        passing_event(met=(0.0, 12.0)),
    )

    assert missing_et(features) == pytest.approx([50.0, 20.0, 12.0])
    assert selection_masks(features)["met"].tolist() == [True, False, False]


def test_dilepton_dphi_is_unsigned_and_cut_from_above():
    features = make_features(
        make_event(massless(30.0, 0.0), massless(20.0, 1.0)),  # 1 rad apart
        make_event(massless(30.0, 1.0), massless(20.0, 0.0)),  # same, opposite sign
        make_event(massless(30.0, 0.0), massless(20.0, 2.5)),  # 2.5 rad apart
        make_event(massless(30.0, -3.0), massless(20.0, 3.0)),  # wraps to 2pi - 6
    )

    expected = [1.0, 1.0, 2.5, 2.0 * np.pi - 6.0]
    assert dilepton_dphi(features) == pytest.approx(expected)
    assert selection_masks(features)["dphi_ll"].tolist() == [True, True, False, True]


def test_dilepton_met_dphi_is_unsigned_and_cut_from_below():
    # Both leptons along +x, so the dilepton system points at phi = 0. The 1.57 rad
    # threshold sits just below pi/2, so a back-to-back-in-phi MET still passes.
    def event(met_phi):
        met = (40.0 * np.cos(met_phi), 40.0 * np.sin(met_phi))
        return make_event(massless(30.0, 0.0), massless(20.0, 0.0), met=met)

    features = make_features(
        event(np.pi),
        event(np.pi / 2),
        event(-np.pi / 2),
        event(1.0),
        event(0.0),
    )

    expected = [np.pi, np.pi / 2, np.pi / 2, 1.0, 0.0]
    assert dilepton_met_dphi(features) == pytest.approx(expected)
    assert selection_masks(features)["dphi_ll_met"].tolist() == [True, True, True, False, False]


def test_jet_multiplicity_counts_unpadded_slots():
    features = make_features(
        passing_event(),
        passing_event(jets=[massless(40.0)]),
        passing_event(jets=[massless(40.0), massless(35.0, np.pi / 2)]),
    )

    assert jet_multiplicity(features).tolist() == [0, 1, 2]


def test_njets_selection_filters_and_none_keeps_everything():
    features = make_features(passing_event(), passing_event(jets=[massless(40.0)]))

    assert selection_masks(features, njets=None)["n_jets"].tolist() == [True, True]
    assert selection_masks(features, njets=0)["n_jets"].tolist() == [True, False]
    assert selection_masks(features, njets=1)["n_jets"].tolist() == [False, True]


def test_a_signal_like_event_passes_every_cut():
    masks = selection_masks(make_features(passing_event()))

    assert all(mask.tolist() == [True] for mask in masks.values())


def test_masks_are_returned_in_cutflow_order():
    features = make_features(passing_event())

    assert list(selection_masks(features)) == [
        "l0_pt",
        "l1_pt",
        "mll",
        "pre_mll",
        "met",
        "dphi_ll",
        "dphi_ll_met",
        "n_jets",
    ]
