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
        raise ValueError(
            f"expected one- or two-dimensional arrays, got {prediction.ndim} dimensions"
        )

    metrics = {
        name: np.full(prediction.shape[1], np.nan)
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
