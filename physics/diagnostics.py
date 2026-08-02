import numpy as np


def component_calibration(prediction, truth):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ValueError(f"prediction and truth shapes differ: {prediction.shape} != {truth.shape}")
    if prediction.ndim == 1:
        prediction = prediction[:, None]
        truth = truth[:, None]
    if prediction.ndim != 2:
        raise ValueError(f"expected one- or two-dimensional arrays, got {prediction.ndim} dimensions")

    metrics = {
        name: np.full(prediction.shape[1], np.nan, dtype=np.float64)
        for name in ("bias", "width_ratio", "correlation", "slope", "intercept", "rmse")
    }
    for index in range(prediction.shape[1]):
        pred = prediction[:, index]
        target = truth[:, index]
        valid = np.isfinite(pred) & np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        if pred.size == 0:
            continue

        pred_std = pred.std()
        target_std = target.std()
        metrics["bias"][index] = np.mean(pred - target)
        metrics["rmse"][index] = np.sqrt(np.mean((pred - target) ** 2))
        if target_std > 0.0:
            metrics["width_ratio"][index] = pred_std / target_std
            covariance = np.mean((pred - pred.mean()) * (target - target.mean()))
            metrics["slope"][index] = covariance / target_std**2
            metrics["intercept"][index] = pred.mean() - metrics["slope"][index] * target.mean()
        if pred_std > 0.0 and target_std > 0.0:
            metrics["correlation"][index] = np.corrcoef(pred, target)[0, 1]

    return metrics


def transverse_sharing_fraction(nu0, nu1, eps=1.0e-12):
    nu0 = np.asarray(nu0, dtype=np.float64)
    nu1 = np.asarray(nu1, dtype=np.float64)
    if nu0.shape != nu1.shape or nu0.ndim != 2 or nu0.shape[1] != 2:
        raise ValueError("nu0 and nu1 must have matching shape (events, 2)")

    total = nu0 + nu1
    denominator = np.sum(total**2, axis=1)
    alpha = np.full(len(total), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(nu0).all(axis=1)
        & np.isfinite(nu1).all(axis=1)
        & (denominator > eps)
    )
    alpha[valid] = np.sum(nu0[valid] * total[valid], axis=1) / denominator[valid]
    return alpha


def histogram_total_variation(prediction, truth, bins):
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    pred_counts, _ = np.histogram(prediction[np.isfinite(prediction)], bins=bins)
    truth_counts, _ = np.histogram(truth[np.isfinite(truth)], bins=bins)
    if pred_counts.sum() == 0 or truth_counts.sum() == 0:
        return np.nan

    pred_density = pred_counts / pred_counts.sum()
    truth_density = truth_counts / truth_counts.sum()
    return 0.5 * np.abs(pred_density - truth_density).sum()
