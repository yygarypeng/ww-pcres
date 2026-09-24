from unittest.mock import patch

import numpy as np
import pytest

from sweep import metrics, selection


def _observables(rows=200):
    zeros = np.zeros(rows)
    return {
        "theta0": np.concatenate([np.zeros(rows // 2), np.full(rows // 2, np.pi / 2)]),
        "phi0": zeros.copy(),
        "theta1": zeros.copy(),
        "phi1": np.concatenate([np.full(rows // 2, -2.0), np.full(rows // 2, 2.0)]),
        "sum_theta": np.full(rows, 2.0),
        "diff_theta": zeros.copy(),
        "sum_phi": zeros.copy(),
        "diff_phi": zeros.copy(),
    }


def test_distribution_objectives_group_only_the_requested_1d_observables():
    prediction = _observables()
    truth = {name: np.zeros_like(values) for name, values in prediction.items()}

    with patch("sweep.metrics.total_variation_2d", return_value=0.91):
        _, angular_1d, interangular_1d, joint_2d = metrics.distribution_report(prediction, truth)

    assert angular_1d == pytest.approx(0.375)
    assert interangular_1d == pytest.approx(0.25)
    assert joint_2d == pytest.approx(0.91)


def _report(angular, interangular, bias_values, bias_score=0.1):
    return {
        "objectives": {
            "bias": bias_score,
            "angular_1d": angular,
            "interangular_1d": interangular,
        },
        "bias_directions": {
            name: {"bias_gev": value, "resolution_gev": 20.0}
            for name, value in zip(metrics.SCORED_DIRECTIONS, bias_values)
        },
    }


def test_thresholds_use_two_sample_standard_deviations():
    reports = [
        _report(0.1, 0.2, [-1.0] * 8),
        _report(0.2, 0.3, [0.0] * 8),
        _report(0.3, 0.4, [1.0] * 8),
    ]

    thresholds = selection.derive_thresholds(reports)

    assert thresholds["angular_1d"] == pytest.approx(0.4)
    assert thresholds["interangular_1d"] == pytest.approx(0.5)
    assert set(thresholds["bias_gev"]) == set(metrics.ZERO_BIAS_DIRECTIONS)
    assert all(limit == pytest.approx(2.0) for limit in thresholds["bias_gev"].values())


def test_thresholds_leave_the_w_energies_unconstrained():
    """Only the momentum directions have to sit at zero; an energy offset is allowed."""
    reports = [
        _report(0.1, 0.2, [-1.0] * 8),
        _report(0.2, 0.3, [0.0] * 8),
        _report(0.3, 0.4, [1.0] * 8),
    ]

    thresholds = selection.derive_thresholds(reports)

    assert not set(metrics.ENERGY_DIRECTIONS) & set(thresholds["bias_gev"])
    assert set(metrics.ZERO_BIAS_DIRECTIONS) | set(metrics.ENERGY_DIRECTIONS) == set(
        metrics.SCORED_DIRECTIONS
    )


def test_feasibility_rejects_one_violated_direction():
    thresholds = {
        "angular_1d": 0.2,
        "interangular_1d": 0.3,
        "bias_gev": {name: 1.0 for name in metrics.SCORED_DIRECTIONS},
    }
    report = _report(0.19, 0.29, [0.0] * 7 + [1.1])

    violations = selection.constraint_violations(report, thresholds)

    assert violations[metrics.SCORED_DIRECTIONS[-1]] > 0.0
    assert not selection.is_feasible(report, thresholds)


@pytest.mark.parametrize("missing", [False, True])
def test_non_finite_or_missing_direction_is_infeasible(missing):
    thresholds = {
        "angular_1d": 0.2,
        "interangular_1d": 0.3,
        "bias_gev": {name: 1.0 for name in metrics.SCORED_DIRECTIONS},
    }
    report = _report(float("nan"), 0.2, [0.0] * 8)
    if missing:
        report["bias_directions"].pop(metrics.SCORED_DIRECTIONS[0])

    violations = selection.constraint_violations(report, thresholds)

    assert any(not np.isfinite(value) for value in violations.values())
    assert not selection.is_feasible(report, thresholds)


def test_zero_bias_spread_accepts_only_zero_bias():
    direction = metrics.SCORED_DIRECTIONS[0]
    thresholds = {
        "angular_1d": 0.2,
        "interangular_1d": 0.3,
        "bias_gev": {direction: 0.0},
    }
    zero = _report(0.1, 0.2, [0.0] * 8)
    biased = _report(0.1, 0.2, [0.1] + [0.0] * 7)

    assert selection.constraint_violations(zero, thresholds)[direction] <= 0.0
    assert np.isinf(selection.constraint_violations(biased, thresholds)[direction])


def test_selection_score_is_the_equal_weight_objective_sum():
    report = _report(0.2, 0.3, [0.0] * 8, bias_score=0.1)
    assert selection.selection_score(report) == pytest.approx(0.6)


def test_threshold_derivation_rejects_incomplete_baselines():
    with pytest.raises(ValueError, match="at least two"):
        selection.derive_thresholds([_report(0.1, 0.2, [0.0] * 8)])


def _report_with(resolution_gev, bias_gev=0.0):
    from sweep import metrics

    return {
        "objectives": {"bias": 0.1, "angular_1d": 0.01, "interangular_1d": 0.01},
        "bias_directions": {
            name: {"bias_gev": bias_gev, "resolution_gev": resolution_gev}
            for name in metrics.ZERO_BIAS_DIRECTIONS
        },
    }


def test_resolution_limits_come_from_the_baseline_seed_scatter():
    reports = [_report_with(20.0), _report_with(21.0), _report_with(22.0)]
    thresholds = selection.derive_thresholds(reports, sigma=2.0)
    # mean 21.0, sample std 1.0, so the limit sits two sigmas above the mean.
    for limit in thresholds["resolution_gev"].values():
        assert limit == pytest.approx(23.0)


def test_a_wider_spread_fails_even_with_no_bias():
    reports = [_report_with(20.0), _report_with(21.0), _report_with(22.0)]
    thresholds = selection.derive_thresholds(reports, sigma=2.0)

    sharp = _report_with(21.0, bias_gev=0.0)
    blunt = _report_with(30.0, bias_gev=0.0)
    assert selection.is_feasible(sharp, thresholds)
    assert not selection.is_feasible(blunt, thresholds)

    violations = selection.constraint_violations(blunt, thresholds)
    assert violations["resolution.px_sum"] == pytest.approx(30.0 / 23.0 - 1.0)


def test_thresholds_without_resolution_limits_still_score():
    """A study recorded before resolution was constrained keeps its own meaning."""
    reports = [_report_with(20.0), _report_with(22.0)]
    thresholds = selection.derive_thresholds(reports, sigma=2.0)
    thresholds.pop("resolution_gev")
    violations = selection.constraint_violations(_report_with(99.0), thresholds)
    assert not any(name.startswith("resolution.") for name in violations)
