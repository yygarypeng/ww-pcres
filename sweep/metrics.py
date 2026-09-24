"""Fixed-ruler sweep metrics; nothing here may depend on a swept hyper-parameter.

Objectives: ``bias`` (RMS bias over resolution of the six W momentum directions),
``angular_1d`` (TV distance of the four decay angles), and ``interangular_1d``
(TV distance of their sums and differences).
"""

import numpy as np
import torch

from physics import _diff_angle, _sum_angle
from physics.diagnostics import component_calibration
from physics.torchBoost import Booster

W_COMPONENT_NAMES = (
    "w0_px",
    "w0_py",
    "w0_pz",
    "w0_energy",
    "w1_px",
    "w1_py",
    "w1_pz",
    "w1_energy",
)

ANGLE_BINS = 40
# Coarse enough that each 2D bin keeps ~380 fold-0 validation events.
JOINT_BINS = 20

# theta lies in [0, pi]; phi and the angle sums/differences wrap into (-pi, pi].
THETA_RANGE = (0.0, float(np.pi))
WRAPPED_RANGE = (-float(np.pi), float(np.pi))

SINGLE_ANGLES = {
    "theta0": THETA_RANGE,
    "phi0": WRAPPED_RANGE,
    "theta1": THETA_RANGE,
    "phi1": WRAPPED_RANGE,
}
PAIR_ANGLES = {
    "sum_theta": WRAPPED_RANGE,
    "diff_theta": WRAPPED_RANGE,
    "sum_phi": WRAPPED_RANGE,
    "diff_phi": WRAPPED_RANGE,
}
JOINT_ANGLES = (
    ("theta0", "theta1"),
    ("phi0", "phi1"),
    ("theta0", "phi0"),
    ("theta1", "phi1"),
    ("theta0", "phi1"),
    ("theta1", "phi0"),
)
OBJECTIVE_NAMES = ("bias", "angular_1d", "interangular_1d")


def edges(value_range, bins):
    return np.linspace(value_range[0], value_range[1], bins + 1)


def _normalized(counts):
    total = counts.sum()
    return counts / total if total > 0 else counts


def total_variation(prediction, truth, bin_edges):
    """Fraction of events that would have to change bin to reconcile the two histograms."""
    predicted = _normalized(np.histogram(prediction, bins=bin_edges)[0].astype(np.float64))
    expected = _normalized(np.histogram(truth, bins=bin_edges)[0].astype(np.float64))
    return 0.5 * float(np.abs(predicted - expected).sum())


def total_variation_2d(prediction, truth, x_edges, y_edges):
    """Two-dimensional counterpart of :func:`total_variation`, for a pair of observables."""
    predicted = _normalized(
        np.histogram2d(prediction[0], prediction[1], bins=(x_edges, y_edges))[0].astype(np.float64)
    )
    expected = _normalized(
        np.histogram2d(truth[0], truth[1], bins=(x_edges, y_edges))[0].astype(np.float64)
    )
    return 0.5 * float(np.abs(predicted - expected).sum())


def chi2_per_ndf(prediction, truth, bin_edges):
    """Two-sample Poisson chi-square; both histograms hold the same number of events."""
    predicted = np.histogram(prediction, bins=bin_edges)[0].astype(np.float64)
    expected = np.histogram(truth, bins=bin_edges)[0].astype(np.float64)
    denominator = predicted + expected
    used = denominator > 0
    if not used.any():
        return float("nan")
    contributions = (predicted - expected) ** 2 / np.where(used, denominator, 1.0)
    return float(contributions[used].sum() / used.sum())


def _observables(angles):
    """The eight angular observables built from a (rows, 4) theta/phi block."""
    theta0, phi0, theta1, phi1 = (angles[..., index] for index in range(4))
    return {
        "theta0": theta0,
        "phi0": phi0,
        "theta1": theta1,
        "phi1": phi1,
        "sum_theta": _sum_angle(theta0, theta1),
        "diff_theta": _diff_angle(theta0, theta1),
        "sum_phi": _sum_angle(phi0, phi1),
        "diff_phi": _diff_angle(phi0, phi1),
    }


def angular_observables(x, y_pred, y_true):
    """Decay angles on the rows where both the truth and predicted rest frames are valid."""
    lep = x[..., :8]
    with torch.no_grad():
        truth_valid, truth_angles = Booster(lep, y_true[..., :8]).lep_theta_phi_with_validity()
        pred_valid, pred_angles = Booster(lep, y_pred[..., :8]).lep_theta_phi_with_validity()
        keep = (
            truth_valid
            & pred_valid
            & torch.isfinite(truth_angles).all(dim=-1)
            & torch.isfinite(pred_angles).all(dim=-1)
        )
    kept = keep.detach().cpu().numpy()
    truth = _observables(truth_angles.detach().cpu().numpy()[kept])
    prediction = _observables(pred_angles.detach().cpu().numpy()[kept])
    return prediction, truth, kept


# dmet pins the px/py sums but not their split between the W bosons, which only the
# four-vector loss and the fourvec_bias penalty hold.
CONSTRAINED_DIRECTIONS = ("px_sum", "py_sum", "w0_pz", "w0_energy", "w1_pz", "w1_energy")
UNCONSTRAINED_DIRECTIONS = ("px_diff", "py_diff")
SCORED_DIRECTIONS = CONSTRAINED_DIRECTIONS + UNCONSTRAINED_DIRECTIONS

# Only the momentum directions are ranked and constrained to zero bias; the energy
# offset of the median-seeking L1 loss is reported as ``bias_energy``.
ENERGY_DIRECTIONS = ("w0_energy", "w1_energy")
ZERO_BIAS_DIRECTIONS = tuple(name for name in SCORED_DIRECTIONS if name not in ENERGY_DIRECTIONS)


def _direction_residuals(residual):
    """Residuals along the directions that are scored, from the per-component ones."""
    px0, py0, pz0, e0, px1, py1, pz1, e1 = (residual[:, i] for i in range(8))
    return {
        "px_sum": px0 + px1,
        "py_sum": py0 + py1,
        "px_diff": px0 - px1,
        "py_diff": py0 - py1,
        "w0_pz": pz0,
        "w0_energy": e0,
        "w1_pz": pz1,
        "w1_energy": e1,
    }


def _scaled_bias(values):
    """Mean offset over its own spread; NaN when the residual has no spread."""
    bias = float(np.mean(values))
    width = float(np.std(values))
    scale = max(abs(bias), float(np.max(np.abs(values))) if len(values) else 0.0)
    if width <= 1e-9 * max(scale, 1.0):
        return {
            "bias_gev": bias,
            "resolution_gev": 0.0,
            "bias_over_resolution": float("nan"),
            "bias_significance": float("nan"),
        }
    return {
        "bias_gev": bias,
        "resolution_gev": width,
        "bias_over_resolution": abs(bias) / width,
        "bias_significance": abs(bias) / (width / np.sqrt(len(values))),
    }


def _rms(directions, names):
    scaled = [directions[name]["bias_over_resolution"] for name in names]
    usable = [value for value in scaled if np.isfinite(value)]
    return float(np.sqrt(np.mean(np.square(usable)))) if usable else float("nan")


def bias_report(y_pred, y_true):
    """Per-component calibration of the eight W four-vector outputs."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    target = np.asarray(y_true, dtype=np.float64)[:, :8]
    calibration = component_calibration(prediction, target)

    bias = calibration["bias"]
    # rmse^2 = bias^2 + var(residual), so the residual width needs no second pass.
    width = np.sqrt(np.clip(calibration["rmse"] ** 2 - bias**2, 0.0, None))
    safe_width = np.where(width > 0.0, width, np.nan)
    scaled = np.abs(bias) / safe_width
    significance = np.abs(bias) / (safe_width / np.sqrt(max(len(prediction), 1)))

    components = {
        name: {
            "bias_gev": float(bias[index]),
            "resolution_gev": float(width[index]),
            "bias_over_resolution": float(scaled[index]),
            "bias_significance": float(significance[index]),
            "width_ratio": float(calibration["width_ratio"][index]),
            "slope": float(calibration["slope"][index]),
        }
        for index, name in enumerate(W_COMPONENT_NAMES)
    }
    residual = prediction - target
    directions = {
        name: _scaled_bias(values) for name, values in _direction_residuals(residual).items()
    }

    usable = scaled[np.isfinite(scaled)]
    scores = {
        "bias": _rms(directions, ZERO_BIAS_DIRECTIONS),
        "bias_energy": _rms(directions, ENERGY_DIRECTIONS),
        "bias_constrained": _rms(directions, CONSTRAINED_DIRECTIONS),
        "bias_split": _rms(directions, UNCONSTRAINED_DIRECTIONS),
        # The old all-component score, kept for comparison with earlier runs.
        "bias_all": float(np.sqrt(np.mean(usable**2))) if usable.size else float("nan"),
    }
    return components, directions, scores


def distribution_report(prediction, truth):
    """Total-variation and chi-square distances for every pinned angular observable."""
    single, pair, joint = {}, {}, {}

    for name, value_range in SINGLE_ANGLES.items():
        bin_edges = edges(value_range, ANGLE_BINS)
        single[name] = {
            "tv": total_variation(prediction[name], truth[name], bin_edges),
            "chi2_per_ndf": chi2_per_ndf(prediction[name], truth[name], bin_edges),
        }

    for name, value_range in PAIR_ANGLES.items():
        bin_edges = edges(value_range, ANGLE_BINS)
        pair[name] = {
            "tv": total_variation(prediction[name], truth[name], bin_edges),
            "chi2_per_ndf": chi2_per_ndf(prediction[name], truth[name], bin_edges),
        }

    for first, second in JOINT_ANGLES:
        x_edges = edges(SINGLE_ANGLES[first], JOINT_BINS)
        y_edges = edges(SINGLE_ANGLES[second], JOINT_BINS)
        joint[f"{first}__{second}"] = {
            "tv": total_variation_2d(
                (prediction[first], prediction[second]),
                (truth[first], truth[second]),
                x_edges,
                y_edges,
            )
        }

    angular_1d = float(np.mean([entry["tv"] for entry in single.values()]))
    interangular_1d = float(np.mean([entry["tv"] for entry in pair.values()]))
    joint_2d = float(np.mean([entry["tv"] for entry in joint.values()]))
    return (
        {"single": single, "pair": pair, "joint": joint},
        angular_1d,
        interangular_1d,
        joint_2d,
    )


def sampling_floor(truth, seed=2330, repeats=3):
    """Distance between two bootstrap resamples of the truth: an upper bound on noise."""
    rng = np.random.default_rng(seed)
    rows = len(next(iter(truth.values())))
    scores = []
    for _ in range(repeats):
        left = {name: values[rng.integers(0, rows, rows)] for name, values in truth.items()}
        right = {name: values[rng.integers(0, rows, rows)] for name, values in truth.items()}
        _, angular_1d, interangular_1d, joint_2d = distribution_report(left, right)
        scores.append((angular_1d, interangular_1d, joint_2d))
    angular_1d, interangular_1d, joint_2d = (float(np.mean(values)) for values in zip(*scores))
    return {
        "angular_1d": angular_1d,
        "interangular_1d": interangular_1d,
        "joint_2d": joint_2d,
    }


def evaluate(x, y_pred, y_true, *, with_floor=False):
    """Every fixed-ruler number for one model on one validation split."""
    components, directions, bias_scores = bias_report(
        y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred,
        y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true,
    )
    prediction, truth, kept = angular_observables(x, y_pred, y_true)
    distributions, angular_1d, interangular_1d, joint_2d = distribution_report(prediction, truth)

    report = {
        "objectives": {
            "bias": bias_scores["bias"],
            "angular_1d": angular_1d,
            "interangular_1d": interangular_1d,
        },
        "joint_2d": joint_2d,
        # Reported only; never ranked or constrained.
        "bias_energy": bias_scores["bias_energy"],
        "bias_constrained": bias_scores["bias_constrained"],
        "bias_split": bias_scores["bias_split"],
        "bias_all": bias_scores["bias_all"],
        "bias_components": components,
        "bias_directions": directions,
        "distributions": distributions,
        "rows_total": int(len(kept)),
        "rows_with_rest_frame": int(kept.sum()),
    }
    if with_floor:
        report["sampling_floor"] = sampling_floor(truth)
    return report


def composite(objectives, weights):
    """Weighted objective sum used for early stopping and checkpoint selection."""
    return float(sum(weights[name] * objectives[name] for name in OBJECTIVE_NAMES))
