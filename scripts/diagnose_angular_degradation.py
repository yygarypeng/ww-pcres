#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.load_data import load_data
from model import LightningWBoson
from model.losses import angular_mmd_features, compute_local_mmd
from notebooks.plottingtool import plot_angular_1d_grid, plot_angular_2d_grid
from physics.torchBoost import Booster
from scripts.evaluate_mmd_bandwidths import discover_unique_checkpoints, tensor_batch

VALIDATION_ROWS = 4096
PARTITION_ROWS = 512
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
LEGACY_RAW_BANDWIDTH_MULTIPLIERS = (0.05, 0.5, 5.0)
CSV_COLUMNS = (
    "epoch",
    "val_loss",
    "raw_angular_mmd",
    "weighted_angular_mmd",
    "mmd_joint_fixed",
    "mmd_wplus_fixed",
    "mmd_wminus_fixed",
    "huber_wplus",
    "huber_wminus",
    "pred_rest_frame_valid_fraction",
    "negative_raw_mmd_batch_fraction",
    "learning_rate",
)
GRADIENT_CSV_COLUMNS = (
    "epoch",
    "loss",
    "raw_loss",
    "effective_weight",
    "weighted_gradient_l2",
    "cosine_huber_wplus",
    "cosine_huber_wminus",
)
GRADIENT_LOSS_ORDER = (
    "huber",
    "higgs_mass",
    "w_mass_huber",
    "alpha_mmd",
    "mass_mmd",
    "angular_mmd",
    "dmet",
    "huber_wplus",
    "huber_wminus",
)
PLOT_FAMILIES = (
    "angular_1d",
    "angular_2d",
    "mixed_sum_1d",
    "mixed_sum_2d",
    "mixed_diff_1d",
    "mixed_diff_2d",
)


def _fit_bandwidths(features):
    distances = torch.pdist(features.reshape(features.shape[0], -1))
    positive = distances[distances > 0]
    scale = positive.median() if positive.numel() else features.new_tensor(1.0)
    return [float(scale * multiplier) for multiplier in BANDWIDTH_MULTIPLIERS]


def validate_manifest(manifest, *, validation_size):
    selected = manifest.get("validation_indices")
    if not isinstance(selected, list) or len(selected) != VALIDATION_ROWS:
        raise ValueError("manifest must contain exactly 4,096 validation indices")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
        raise ValueError("manifest validation indices must be integers")
    if len(set(selected)) != VALIDATION_ROWS:
        raise ValueError("manifest validation indices must be unique")
    if any(index < 0 or index >= validation_size for index in selected):
        raise ValueError("manifest validation indices must be in range")

    if manifest.get("gradient_indices") != selected[:PARTITION_ROWS]:
        raise ValueError("manifest gradient indices must equal the first 512 validation indices")
    partitions = manifest.get("partitions")
    if (
        not isinstance(partitions, list)
        or len(partitions) != VALIDATION_ROWS // PARTITION_ROWS
        or any(
            not isinstance(partition, list) or len(partition) != PARTITION_ROWS
            for partition in partitions
        )
        or [index for partition in partitions for index in partition] != selected
    ):
        raise ValueError("manifest partitions must be eight ordered 512-row panel partitions")

    plot_bins = manifest.get("plot_bins")
    if not isinstance(plot_bins, dict) or set(plot_bins) != {"theta", "phi"}:
        raise ValueError("manifest plot bins must contain theta and phi")
    for values in plot_bins.values():
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("manifest plot bins must be numeric") from None
        if (
            values.ndim != 1
            or len(values) < 2
            or not np.isfinite(values).all()
            or not np.all(np.diff(values) > 0)
        ):
            raise ValueError("manifest plot bins must be finite and strictly increasing")

    bandwidths = manifest.get("feature_bandwidths")
    if not isinstance(bandwidths, dict) or set(bandwidths) != {"joint", "wplus", "wminus"}:
        raise ValueError("manifest bandwidths must contain joint, wplus, and wminus")
    for values in bandwidths.values():
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("manifest bandwidth values must be numeric") from None
        if (
            values.ndim != 1
            or len(values) == 0
            or not np.isfinite(values).all()
            or not np.all(values > 0)
        ):
            raise ValueError("manifest bandwidth values must be finite and positive")
    return manifest


def manifest_partition_positions(manifest):
    position_by_index = {
        selected_index: position
        for position, selected_index in enumerate(manifest["validation_indices"])
    }
    return [
        [position_by_index[selected_index] for selected_index in partition]
        for partition in manifest["partitions"]
    ]


def build_manifest(filtered_indices, truth_angles, truth_valid, *, seed):
    filtered_indices = np.asarray(filtered_indices)
    truth_angles = torch.as_tensor(truth_angles, dtype=torch.float64)
    truth_valid = torch.as_tensor(truth_valid, dtype=torch.bool)
    if (
        filtered_indices.ndim != 1
        or len(filtered_indices) != len(truth_angles)
        or len(filtered_indices) != len(truth_valid)
    ):
        raise ValueError(
            "filtered indices, truth angles, and truth validity must contain the same rows"
        )
    if len(filtered_indices) < VALIDATION_ROWS:
        raise ValueError(f"at least {VALIDATION_ROWS} filtered validation rows are required")

    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(len(filtered_indices), size=VALIDATION_ROWS, replace=False)
    selected_indices = filtered_indices[selected_positions].astype(np.int64).tolist()
    selected_truth_valid = truth_valid[selected_positions]
    features = angular_mmd_features(truth_angles[selected_positions][selected_truth_valid])
    feature_bandwidths = {
        "joint": _fit_bandwidths(features),
        "wplus": _fit_bandwidths(features[:, :3]),
        "wminus": _fit_bandwidths(features[:, 3:]),
    }
    partitions = [
        selected_indices[start : start + PARTITION_ROWS]
        for start in range(0, VALIDATION_ROWS, PARTITION_ROWS)
    ]
    manifest = {
        "seed": int(seed),
        "validation_indices": selected_indices,
        "gradient_indices": selected_indices[:PARTITION_ROWS],
        "partitions": partitions,
        "plot_bins": {
            "theta": np.linspace(0.0, np.pi, 41).tolist(),
            "phi": np.linspace(-np.pi, np.pi, 41).tolist(),
        },
        "feature_bandwidths": feature_bandwidths,
    }
    return validate_manifest(manifest, validation_size=len(filtered_indices))


def _mixed_kernel(left, right, bandwidths, kernel):
    distances = torch.cdist(left, right).square()
    bandwidths = torch.as_tensor(bandwidths, dtype=distances.dtype, device=distances.device)
    squared = bandwidths.square().reshape(-1, 1, 1)
    if kernel == "rbf":
        values = torch.exp(-0.5 * distances.unsqueeze(0) / (squared + 1.0e-16))
    elif kernel == "imq":
        values = squared / (squared + distances.unsqueeze(0) + 1.0e-16)
    else:
        raise ValueError(f"Unsupported feature kernel: {kernel}")
    return values.mean(dim=0)


def blockwise_mmd_v(prediction, truth, bandwidths, *, kernel="imq", block_size=512):
    prediction = prediction.reshape(prediction.shape[0], -1)
    truth = truth.reshape(truth.shape[0], -1)
    if prediction.shape[1:] != truth.shape[1:]:
        raise ValueError("prediction and truth must have the same feature shape")
    if prediction.shape[0] == 0 or truth.shape[0] == 0:
        raise ValueError("cannot compute MMD for zero rows")
    if block_size <= 0:
        raise ValueError("block size must be positive")

    def kernel_sum(left, right):
        total = prediction.new_zeros(())
        for row_start in range(0, len(left), block_size):
            row = left[row_start : row_start + block_size]
            for column_start in range(0, len(right), block_size):
                column = right[column_start : column_start + block_size]
                total += _mixed_kernel(row, column, bandwidths, kernel).sum()
        return total

    m = prediction.shape[0]
    n = truth.shape[0]
    value = (
        kernel_sum(prediction, prediction) / m**2
        + kernel_sum(truth, truth) / n**2
        - 2.0 * kernel_sum(prediction, truth) / (m * n)
    )
    return value.clamp_min(0.0)


def partitioned_raw_mmd(
    prediction,
    truth,
    valid_mask,
    partitions,
    *,
    condition=None,
    mmd_kwargs=None,
):
    if condition is None:
        condition = prediction.new_empty((len(prediction), 0))
        defaults = {"local": False}
    else:
        defaults = {}
    kwargs = {**defaults, **(mmd_kwargs or {})}
    values = []
    for partition in partitions:
        indices = torch.as_tensor(partition, dtype=torch.long, device=prediction.device)
        indices = indices[valid_mask[indices]]
        value = compute_local_mmd(
            prediction[indices],
            truth[indices],
            condition[indices],
            **kwargs,
        )
        values.append(float(value))
    if not values:
        raise ValueError("at least one fixed partition is required")
    return float(np.mean(values)), sum(value < 0.0 for value in values) / len(values), values


def angular_checkpoint_metrics(
    prediction,
    truth,
    prediction_valid,
    truth_valid,
    *,
    feature_bandwidths,
    partitions,
    condition=None,
    raw_mmd_kwargs=None,
    block_size=512,
):
    prediction_valid = torch.as_tensor(prediction_valid, dtype=torch.bool, device=prediction.device)
    truth_valid = torch.as_tensor(truth_valid, dtype=torch.bool, device=truth.device)
    if len(prediction_valid) != len(prediction) or len(truth_valid) != len(truth):
        raise ValueError("angular validity must have one value per selected row")
    prediction_valid = prediction_valid & torch.isfinite(prediction).all(dim=1)
    truth_valid = truth_valid & torch.isfinite(truth).all(dim=1)
    if not prediction_valid.any():
        raise ValueError("checkpoint has no valid angular predictions")
    if not truth_valid.any():
        raise ValueError("panel has no valid truth angular samples")

    raw, negative_fraction, _ = partitioned_raw_mmd(
        prediction,
        truth,
        prediction_valid & truth_valid,
        partitions,
        condition=condition,
        mmd_kwargs=raw_mmd_kwargs,
    )
    pred_samples = prediction[prediction_valid]
    truth_samples = truth[truth_valid]
    return {
        "raw_angular_mmd": raw,
        "mmd_joint_fixed": float(
            blockwise_mmd_v(
                pred_samples,
                truth_samples,
                feature_bandwidths["joint"],
                block_size=block_size,
            )
        ),
        "mmd_wplus_fixed": float(
            blockwise_mmd_v(
                pred_samples[:, :3],
                truth_samples[:, :3],
                feature_bandwidths["wplus"],
                block_size=block_size,
            )
        ),
        "mmd_wminus_fixed": float(
            blockwise_mmd_v(
                pred_samples[:, 3:],
                truth_samples[:, 3:],
                feature_bandwidths["wminus"],
                block_size=block_size,
            )
        ),
        "pred_rest_frame_valid_fraction": float(prediction_valid.sum()) / len(prediction_valid),
        "negative_raw_mmd_batch_fraction": negative_fraction,
    }


def write_metrics_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["weighted_angular_mmd"] = 2000.0 * float(row["raw_angular_mmd"])
            writer.writerow({name: row[name] for name in CSV_COLUMNS})


def write_gradient_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GRADIENT_CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path, columns):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path} has an incompatible CSV schema")
        return list(reader)


def _atomic_write_csv(path, writer, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _wrap_angle(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _angular_plot_observables(prediction, truth, plot_bins):
    theta_bins = np.asarray(plot_bins["theta"], dtype=float)
    phi_bins = np.asarray(plot_bins["phi"], dtype=float)
    wrapped_bins = phi_bins / np.pi

    def observable(pred, true, label, bins, *, log=True, vmax=800.0):
        return {
            "pred": pred,
            "truth": true,
            "label": label,
            "bins": bins,
            "log": log,
            "vmax": vmax,
        }

    angular = [
        observable(
            prediction[:, 0] + prediction[:, 2],
            truth[:, 0] + truth[:, 2],
            r"$\sum_{+-}\theta^*_{\ell}$",
            np.linspace(2.0 * theta_bins[0], 2.0 * theta_bins[-1], len(theta_bins)) / np.pi,
        ),
        observable(
            prediction[:, 0] - prediction[:, 2],
            truth[:, 0] - truth[:, 2],
            r"$\Delta_{+-}\theta^*_{\ell}$",
            np.linspace(
                theta_bins[0] - theta_bins[-1],
                theta_bins[-1] - theta_bins[0],
                len(theta_bins),
            )
            / np.pi,
        ),
        observable(
            _wrap_angle(prediction[:, 1] + prediction[:, 3]),
            _wrap_angle(truth[:, 1] + truth[:, 3]),
            r"$\sum_{+-}\phi^*_{\ell}$",
            wrapped_bins,
            log=False,
            vmax=300.0,
        ),
        observable(
            _wrap_angle(prediction[:, 1] - prediction[:, 3]),
            _wrap_angle(truth[:, 1] - truth[:, 3]),
            r"$\Delta_{+-}\phi^*_{\ell}$",
            wrapped_bins,
            log=False,
            vmax=300.0,
        ),
    ]
    mixed_sum = []
    mixed_diff = []
    for theta_slot, phi_slot, signs in (
        (0, 1, "++"),
        (0, 3, "+-"),
        (2, 1, "-+"),
        (2, 3, "--"),
    ):
        mixed_sum.append(
            observable(
                _wrap_angle(prediction[:, theta_slot] + prediction[:, phi_slot]),
                _wrap_angle(truth[:, theta_slot] + truth[:, phi_slot]),
                rf"$\sum_{{{signs}}}\theta^*\phi^*$",
                wrapped_bins,
            )
        )
        mixed_diff.append(
            observable(
                _wrap_angle(prediction[:, theta_slot] - prediction[:, phi_slot]),
                _wrap_angle(truth[:, theta_slot] - truth[:, phi_slot]),
                rf"$\Delta_{{{signs}}}\theta^*\phi^*$",
                wrapped_bins,
            )
        )
    return angular, mixed_sum, mixed_diff


def save_angular_plots(output_dir, *, epoch, prediction, truth, plot_bins):
    matplotlib.rcParams["text.parse_math"] = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    angular, mixed_sum, mixed_diff = _angular_plot_observables(
        np.asarray(prediction), np.asarray(truth), plot_bins
    )
    definitions = (
        (
            "angular_1d",
            plot_angular_1d_grid,
            angular,
            "Angular sums and differences: 1D distributions",
            {},
        ),
        (
            "angular_2d",
            plot_angular_2d_grid,
            angular,
            "Angular sums and differences: 2D correlations",
            {},
        ),
        (
            "mixed_sum_1d",
            plot_angular_1d_grid,
            mixed_sum,
            "Mixed angular sums: 1D distributions",
            {"share_axes": True},
        ),
        (
            "mixed_sum_2d",
            plot_angular_2d_grid,
            mixed_sum,
            "Mixed angular sums: 2D correlations",
            {"shared_colorbar": True, "share_axes": True},
        ),
        (
            "mixed_diff_1d",
            plot_angular_1d_grid,
            mixed_diff,
            "Mixed angular differences: 1D distributions",
            {"share_axes": True},
        ),
        (
            "mixed_diff_2d",
            plot_angular_2d_grid,
            mixed_diff,
            "Mixed angular differences: 2D correlations",
            {"shared_colorbar": True, "share_axes": True},
        ),
    )
    paths = []
    for name, plotter, observables, title, kwargs in definitions:
        figure, _ = plotter(observables, title, **kwargs)
        path = output_dir / f"epoch_{epoch:04d}_{name}.png"
        try:
            figure.savefig(path, bbox_inches="tight")
        finally:
            plt.close(figure)
        paths.append(path)
    return paths


def angular_plot_paths(output_dir, epoch):
    output_dir = Path(output_dir)
    return [output_dir / f"epoch_{epoch:04d}_{family}.png" for family in PLOT_FAMILIES]


def _parameter_gradient(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def gradient_diagnostic_rows(model, features, targets, gradient_indices, *, epoch):
    if len(gradient_indices) != PARTITION_ROWS:
        raise ValueError("gradient diagnostics require exactly 512 persisted indices")
    model.eval()
    indices = torch.as_tensor(gradient_indices, dtype=torch.long, device=features.device)
    features = features[indices]
    targets = targets[indices]
    prediction, auxiliary = model(features, return_aux=True)
    _, losses = model._compute_losses(features, targets, prediction, auxiliary["cond"], auxiliary)
    residual = (prediction - targets[:, :8]).reshape(len(prediction), 2, 4)
    residual = residual / model.w_fourvec_scales
    references = {
        "huber_wplus": F.huber_loss(residual[:, 0], torch.zeros_like(residual[:, 0])),
        "huber_wminus": F.huber_loss(residual[:, 1], torch.zeros_like(residual[:, 1])),
    }
    weights = model._effective_loss_weights()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("gradient diagnostics require trainable model parameters")
    all_losses = {**losses, **references}
    gradients = {
        name: _parameter_gradient(loss, parameters, retain_graph=index < len(all_losses) - 1)
        for index, (name, loss) in enumerate(all_losses.items())
    }
    rows = []
    for name, loss in all_losses.items():
        weight = 0.5 * weights.get("huber", 0.0) if name in references else weights.get(name, 0.0)
        gradient = gradients[name]
        rows.append(
            {
                "epoch": epoch,
                "loss": name,
                "raw_loss": float(loss.detach()),
                "effective_weight": float(weight),
                "weighted_gradient_l2": float(abs(weight) * torch.linalg.vector_norm(gradient)),
                "cosine_huber_wplus": float(
                    F.cosine_similarity(gradient, gradients["huber_wplus"], dim=0, eps=1.0e-12)
                ),
                "cosine_huber_wminus": float(
                    F.cosine_similarity(gradient, gradients["huber_wminus"], dim=0, eps=1.0e-12)
                ),
            }
        )
    return rows


def _angular_angles(inputs, w_fourvectors):
    booster = Booster(inputs[..., :8], w_fourvectors)
    valid = booster.valid_rest_frame_mask()
    angles = inputs.new_full((len(inputs), 4), torch.nan)
    if valid.any():
        valid_angles = booster.lep_theta_phi_in_w_rest(booster.particles[valid])[:4]
        angles[valid] = torch.stack(valid_angles, dim=-1)
    return angles, valid


def _angular_features(inputs, w_fourvectors):
    angles, valid = _angular_angles(inputs, w_fourvectors)
    encoded = inputs.new_full((len(inputs), 6), torch.nan)
    encoded[valid] = angular_mmd_features(angles[valid])
    return encoded, valid


def _slot_huber(prediction, truth, scales, slot):
    prediction = prediction[:, slot]
    truth = truth[:, slot]
    valid = torch.isfinite(prediction).all(dim=1) & torch.isfinite(truth).all(dim=1)
    if not valid.any():
        raise ValueError("checkpoint has no finite W four-vector predictions")
    residual = (prediction[valid] - truth[valid]) / scales
    return float(F.huber_loss(residual, torch.zeros_like(residual)))


def _checkpoint_learning_rate(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    optimizer_states = checkpoint.get("optimizer_states", [])
    if optimizer_states and optimizer_states[0].get("param_groups"):
        return float(optimizer_states[0]["param_groups"][0]["lr"])
    return float(checkpoint["hyper_parameters"]["lr"])


def evaluate_checkpoint(
    checkpoint, features, targets, manifest, device, *, block_size, plot_output_dir=None
):
    model = (
        LightningWBoson.load_from_checkpoint(
            checkpoint.path, map_location="cpu", weights_only=False
        )
        .eval()
        .to(device)
    )
    model.trainer = SimpleNamespace(current_epoch=checkpoint.epoch)
    features = tensor_batch(features, device)
    targets = tensor_batch(targets, device)
    partitions = manifest_partition_positions(manifest)
    predictions = []
    conditions = []
    totals = []
    with torch.no_grad():
        for partition in partitions:
            batch_features = features[partition]
            batch_targets = targets[partition]
            prediction, auxiliary = model(batch_features, return_aux=True)
            total, _ = model._compute_losses(
                batch_features,
                batch_targets,
                prediction,
                auxiliary["cond"],
                auxiliary,
            )
            predictions.append(prediction)
            conditions.append(auxiliary["cond"])
            totals.append(float(total))
    prediction = torch.cat(predictions)
    condition = torch.cat(conditions)
    truth_angles, truth_valid = _angular_angles(features, targets[:, :8])
    pred_angles, pred_valid = _angular_angles(features, prediction)
    truth_angular = features.new_full((len(features), 6), torch.nan)
    pred_angular = features.new_full((len(features), 6), torch.nan)
    truth_angular[truth_valid] = angular_mmd_features(truth_angles[truth_valid])
    pred_angular[pred_valid] = angular_mmd_features(pred_angles[pred_valid])
    angular_metrics = angular_checkpoint_metrics(
        pred_angular,
        truth_angular,
        pred_valid,
        truth_valid,
        feature_bandwidths=manifest["feature_bandwidths"],
        partitions=partitions,
        condition=condition,
        raw_mmd_kwargs=legacy_raw_mmd_kwargs(model),
        block_size=block_size,
    )
    if plot_output_dir is not None:
        save_angular_plots(
            plot_output_dir,
            epoch=checkpoint.epoch,
            prediction=pred_angles.detach().cpu().numpy(),
            truth=truth_angles.detach().cpu().numpy(),
            plot_bins=manifest["plot_bins"],
        )
    # This is intentionally the validation loss recomputed on the fixed 4,096-row panel.
    return {
        "epoch": checkpoint.epoch,
        "val_loss": float(np.mean(totals)),
        **angular_metrics,
        "huber_wplus": _slot_huber(prediction, targets[:, :8], model.w_fourvec_scales, slice(0, 4)),
        "huber_wminus": _slot_huber(
            prediction, targets[:, :8], model.w_fourvec_scales, slice(4, 8)
        ),
        "learning_rate": _checkpoint_learning_rate(checkpoint.path),
    }


def legacy_raw_mmd_kwargs(model):
    kwargs = dict(model._mmd_kwargs("angular"))
    return {
        "local": False,
        "feature_kernel": kwargs["kernel"],
        "feature_bandwidth_multipliers": list(LEGACY_RAW_BANDWIDTH_MULTIPLIERS),
        "estimator": "u",
    }


def evaluate_checkpoint_gradients(checkpoint, features, targets, device):
    model = (
        LightningWBoson.load_from_checkpoint(
            checkpoint.path, map_location="cpu", weights_only=False
        )
        .eval()
        .to(device)
    )
    model.trainer = SimpleNamespace(current_epoch=checkpoint.epoch)
    features = tensor_batch(features, device)
    targets = tensor_batch(targets, device)
    return gradient_diagnostic_rows(
        model, features, targets, range(PARTITION_ROWS), epoch=checkpoint.epoch
    )


def expected_gradient_loss_names(checkpoint):
    saved = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    hyper_parameters = saved["hyper_parameters"]
    defaults = {
        "huber": 1.0,
        "higgs_mass": 0.0,
        "w_mass_huber": 0.0,
        "alpha_mmd": 0.0,
        "mass_mmd": 0.0,
        "angular_mmd": 0.0,
        "dmet": 0.0,
    }
    weights = {
        **defaults,
        **hyper_parameters.get("loss_weights", {}),
    }
    effective_weights = dict(weights)
    ramp_epochs = hyper_parameters.get("angular_mmd_ramp_epochs", 0)
    if ramp_epochs:
        progress = min(max(checkpoint.epoch / ramp_epochs, 0.0), 1.0)
        effective_weights["angular_mmd"] *= (1.0 - math.cos(math.pi * progress)) / 2.0
    adaptive = hyper_parameters.get("adaptive_loss_weights", False)
    names = [
        name
        for name in GRADIENT_LOSS_ORDER[:-2]
        if effective_weights[name] != 0.0 or (adaptive and weights[name] != 0.0)
    ]
    return (*names, "huber_wplus", "huber_wminus")


def _row_epoch(row):
    return int(float(row["epoch"]))


def _epoch_is_complete(epoch, expected_losses, metric_rows, gradient_rows, plot_dir):
    epoch_metrics = [row for row in metric_rows if _row_epoch(row) == epoch]
    epoch_gradients = [row for row in gradient_rows if _row_epoch(row) == epoch]
    return (
        len(epoch_metrics) == 1
        and len(epoch_gradients) == len(expected_losses)
        and {row["loss"] for row in epoch_gradients} == set(expected_losses)
        and all(path.is_file() for path in angular_plot_paths(plot_dir, epoch))
    )


def run_checkpoint_evaluations(
    *,
    checkpoints,
    features,
    targets,
    manifest,
    device,
    block_size,
    metrics_output,
    gradient_output,
    plot_dir,
):
    metric_rows = _read_csv(metrics_output, CSV_COLUMNS)
    gradient_rows = _read_csv(gradient_output, GRADIENT_CSV_COLUMNS)
    selected = np.asarray(manifest["validation_indices"], dtype=np.int64)
    gradient_selected = np.asarray(manifest["gradient_indices"], dtype=np.int64)
    loss_position = {name: index for index, name in enumerate(GRADIENT_LOSS_ORDER)}

    for checkpoint in sorted(checkpoints, key=lambda item: item.epoch):
        expected_losses = expected_gradient_loss_names(checkpoint)
        if _epoch_is_complete(
            checkpoint.epoch,
            expected_losses,
            metric_rows,
            gradient_rows,
            plot_dir,
        ):
            print(f"Skipped epoch {checkpoint.epoch}", flush=True)
            continue

        metric_row = evaluate_checkpoint(
            checkpoint,
            features[selected],
            targets[selected],
            manifest,
            device,
            block_size=block_size,
            plot_output_dir=plot_dir,
        )
        checkpoint_gradients = evaluate_checkpoint_gradients(
            checkpoint,
            features[gradient_selected],
            targets[gradient_selected],
            device,
        )
        if len(checkpoint_gradients) != len(expected_losses) or {
            row["loss"] for row in checkpoint_gradients
        } != set(expected_losses):
            raise ValueError(f"epoch {checkpoint.epoch} produced incomplete gradient diagnostics")

        metric_rows = [row for row in metric_rows if _row_epoch(row) != checkpoint.epoch] + [
            metric_row
        ]
        gradient_rows = [
            row for row in gradient_rows if _row_epoch(row) != checkpoint.epoch
        ] + checkpoint_gradients
        metric_rows.sort(key=_row_epoch)
        gradient_rows.sort(
            key=lambda row: (
                _row_epoch(row),
                loss_position.get(row["loss"], len(loss_position)),
                row["loss"],
            )
        )
        _atomic_write_csv(metrics_output, write_metrics_csv, metric_rows)
        _atomic_write_csv(gradient_output, write_gradient_csv, gradient_rows)
        print(f"Completed epoch {checkpoint.epoch}", flush=True)


def select_angular_checkpoint(rows, *, validity_floor):
    validity_floor = float(validity_floor)
    if not math.isfinite(validity_floor) or not 0.0 <= validity_floor <= 1.0:
        raise ValueError("validity_floor must be finite and between zero and one")

    eligible = []
    for row in rows:
        validity = float(row["pred_rest_frame_valid_fraction"])
        components = [
            float(row["mmd_joint_fixed"]),
            float(row["mmd_wplus_fixed"]),
            float(row["mmd_wminus_fixed"]),
        ]
        if not math.isfinite(validity) or not all(map(math.isfinite, components)):
            continue
        if validity < validity_floor:
            continue
        score = components[0] + max(components[1:])
        selected_row = dict(row)
        selected_row["angular_checkpoint_score"] = score
        eligible.append(selected_row)
    if not eligible:
        raise ValueError("no checkpoint meets the validity floor")
    return min(eligible, key=lambda row: (row["angular_checkpoint_score"], int(row["epoch"])))


def build_angular_selection(rows, checkpoints, *, validity_floor):
    selected = select_angular_checkpoint(rows, validity_floor=validity_floor)
    checkpoint_by_epoch = {}
    for checkpoint in checkpoints:
        path = checkpoint.path.resolve()
        existing = checkpoint_by_epoch.get(checkpoint.epoch)
        if existing is not None and existing != path:
            raise ValueError(f"multiple checkpoints found for epoch {checkpoint.epoch}")
        checkpoint_by_epoch[checkpoint.epoch] = path
    epoch = int(selected["epoch"])
    if epoch not in checkpoint_by_epoch:
        raise ValueError(f"selected epoch {epoch} has no checkpoint")
    return {
        **selected,
        "epoch": epoch,
        "validity_floor": float(validity_floor),
        "checkpoint": str(checkpoint_by_epoch[epoch]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic angular diagnostics. Checkpoints must be trusted local "
            "artifacts because loading requires weights_only=False."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--gradient-output", type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--validity-floor", type=float, default=1.0)
    parser.add_argument("--split", default="ggF_val")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints, duplicates = discover_unique_checkpoints(args.checkpoint_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {args.checkpoint_dir}")
    for duplicate, canonical in sorted(duplicates.items()):
        print(f"Skipping duplicate checkpoint {duplicate.name} ({canonical.name})", file=sys.stderr)

    with redirect_stdout(sys.stderr):
        features, targets, _, _ = load_data(args.data_path, categories=[args.split])
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        all_features = tensor_batch(features, torch.device("cpu"))
        all_targets = tensor_batch(targets, torch.device("cpu"))
        truth_angles, truth_valid = _angular_angles(all_features, all_targets[:, :8])
        filtered_indices = np.arange(len(features))
        manifest = build_manifest(filtered_indices, truth_angles, truth_valid, seed=args.seed)
        args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    validate_manifest(manifest, validation_size=len(features))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = args.plot_dir or args.output.with_name(f"{args.output.stem}_plots")
    gradient_output = args.gradient_output or args.output.with_name(
        f"{args.output.stem}_gradients.csv"
    )
    run_checkpoint_evaluations(
        checkpoints=checkpoints,
        features=features,
        targets=targets,
        manifest=manifest,
        device=device,
        block_size=args.block_size,
        metrics_output=args.output,
        gradient_output=gradient_output,
        plot_dir=plot_dir,
    )
    selection = build_angular_selection(
        _read_csv(args.output, CSV_COLUMNS),
        checkpoints,
        validity_floor=args.validity_floor,
    )
    selection_output = args.selection_output or args.output.with_name(
        f"{args.output.stem}_selection.json"
    )
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Selected epoch {selection['epoch']}: {selection['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
