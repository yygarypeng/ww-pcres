"""Baseline thresholds and constrained sweep selection."""

import math

import numpy as np

from sweep import metrics

DISTRIBUTION_OBJECTIVES = ("angular_1d", "interangular_1d")


def derive_thresholds(reports, sigma=2.0):
    """Non-inferiority and zero-bias limits from the baseline seed scatter."""
    if len(reports) < 2:
        raise ValueError("at least two complete baseline reports are required")
    sigma = float(sigma)
    thresholds = {}
    for name in DISTRIBUTION_OBJECTIVES:
        values = np.asarray([report["objectives"][name] for report in reports], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"baseline objective {name} contains a non-finite value")
        thresholds[name] = float(values.mean() + sigma * values.std(ddof=1))

    # Resolution is not scored, so constrain it to stop bias being bought with spread.
    thresholds["resolution_gev"] = {}
    for direction in metrics.ZERO_BIAS_DIRECTIONS:
        values = _direction_values(reports, direction, "resolution_gev")
        thresholds["resolution_gev"][direction] = float(values.mean() + sigma * values.std(ddof=1))

    thresholds["bias_gev"] = {}
    for direction in metrics.ZERO_BIAS_DIRECTIONS:
        values = _direction_values(reports, direction, "bias_gev")
        thresholds["bias_gev"][direction] = float(sigma * values.std(ddof=1))
    return thresholds


def _direction_values(reports, direction, field):
    """One per-direction quantity across the baseline reports, checked for use."""
    try:
        values = np.asarray(
            [report["bias_directions"][direction][field] for report in reports], dtype=float
        )
    except KeyError as error:
        raise ValueError(
            f"baseline report is missing {field} for bias direction {direction}"
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(f"baseline direction {direction} has a non-finite {field}")
    return values


def _relative_violation(value, limit):
    if not math.isfinite(value) or not math.isfinite(limit) or limit < 0.0:
        return float("inf")
    if limit == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return value / limit - 1.0


def constraint_violations(report, thresholds):
    """Optuna-compatible constraint values: non-positive passes, positive fails."""
    violations = {}
    objectives = report.get("objectives", {})
    for name in DISTRIBUTION_OBJECTIVES:
        value = float(objectives.get(name, float("nan")))
        violations[name] = _relative_violation(value, float(thresholds[name]))

    directions = report.get("bias_directions", {})
    for name, limit in thresholds["bias_gev"].items():
        entry = directions.get(name, {})
        value = abs(float(entry.get("bias_gev", float("nan"))))
        violations[name] = _relative_violation(value, float(limit))

    # Optional, so thresholds recorded before resolution was constrained still apply.
    for name, limit in thresholds.get("resolution_gev", {}).items():
        entry = directions.get(name, {})
        value = float(entry.get("resolution_gev", float("nan")))
        violations[f"resolution.{name}"] = _relative_violation(value, float(limit))
    return violations


def is_feasible(report, thresholds):
    return all(value <= 0.0 for value in constraint_violations(report, thresholds).values())


def selection_score(report):
    """Equal-weight score used to rank feasible configurations."""
    return float(sum(report["objectives"][name] for name in metrics.OBJECTIVE_NAMES))
