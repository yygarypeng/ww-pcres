#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.load_data import load_data
from model import LightningWBoson
from model.losses import angular_mmd_features, compute_local_mmd
from physics.torchBoost import Booster
from scripts.evaluate_mmd_bandwidths import discover_unique_checkpoints, tensor_batch

VALIDATION_ROWS = 4096
PARTITION_ROWS = 512
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
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


def _fit_bandwidths(features):
    distances = torch.pdist(features.reshape(features.shape[0], -1))
    positive = distances[distances > 0]
    scale = positive.median() if positive.numel() else features.new_tensor(1.0)
    return [float(scale * multiplier) for multiplier in BANDWIDTH_MULTIPLIERS]


def build_manifest(filtered_indices, truth_angles, *, seed):
    filtered_indices = np.asarray(filtered_indices)
    truth_angles = torch.as_tensor(truth_angles, dtype=torch.float64)
    if filtered_indices.ndim != 1 or len(filtered_indices) != len(truth_angles):
        raise ValueError("filtered indices and truth angles must contain the same number of rows")
    if len(filtered_indices) < VALIDATION_ROWS:
        raise ValueError(f"at least {VALIDATION_ROWS} filtered validation rows are required")

    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(len(filtered_indices), size=VALIDATION_ROWS, replace=False)
    selected_indices = filtered_indices[selected_positions].astype(np.int64).tolist()
    features = angular_mmd_features(truth_angles[selected_positions])
    feature_bandwidths = {
        "joint": _fit_bandwidths(features),
        "wplus": _fit_bandwidths(features[:, :3]),
        "wminus": _fit_bandwidths(features[:, 3:]),
    }
    partitions = [
        selected_indices[start : start + PARTITION_ROWS]
        for start in range(0, VALIDATION_ROWS, PARTITION_ROWS)
    ]
    return {
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
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have the same shape")
    if prediction.shape[0] == 0:
        raise ValueError("cannot compute MMD for zero rows")
    if block_size <= 0:
        raise ValueError("block size must be positive")

    total = prediction.new_zeros(())
    for row_start in range(0, len(prediction), block_size):
        row_pred = prediction[row_start : row_start + block_size]
        row_truth = truth[row_start : row_start + block_size]
        for column_start in range(0, len(prediction), block_size):
            column_pred = prediction[column_start : column_start + block_size]
            column_truth = truth[column_start : column_start + block_size]
            total += (
                _mixed_kernel(row_pred, column_pred, bandwidths, kernel)
                + _mixed_kernel(row_truth, column_truth, bandwidths, kernel)
                - _mixed_kernel(row_pred, column_truth, bandwidths, kernel)
                - _mixed_kernel(row_truth, column_pred, bandwidths, kernel)
            ).sum()
    return (total / prediction.shape[0] ** 2).clamp_min(0.0)


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
    *,
    feature_bandwidths,
    partitions,
    condition=None,
    raw_mmd_kwargs=None,
    block_size=512,
):
    prediction_valid = torch.as_tensor(prediction_valid, dtype=torch.bool, device=prediction.device)
    if len(prediction_valid) != len(prediction):
        raise ValueError("prediction validity must have one value per selected row")
    finite = torch.isfinite(prediction).all(dim=1) & torch.isfinite(truth).all(dim=1)
    valid = prediction_valid & finite
    if not valid.any():
        raise ValueError("checkpoint has no valid angular predictions")

    raw, negative_fraction, _ = partitioned_raw_mmd(
        prediction,
        truth,
        valid,
        partitions,
        condition=condition,
        mmd_kwargs=raw_mmd_kwargs,
    )
    pred_valid = prediction[valid]
    truth_valid = truth[valid]
    return {
        "raw_angular_mmd": raw,
        "mmd_joint_fixed": float(
            blockwise_mmd_v(
                pred_valid,
                truth_valid,
                feature_bandwidths["joint"],
                block_size=block_size,
            )
        ),
        "mmd_wplus_fixed": float(
            blockwise_mmd_v(
                pred_valid[:, :3],
                truth_valid[:, :3],
                feature_bandwidths["wplus"],
                block_size=block_size,
            )
        ),
        "mmd_wminus_fixed": float(
            blockwise_mmd_v(
                pred_valid[:, 3:],
                truth_valid[:, 3:],
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
            row["weighted_angular_mmd"] = 2000.0 * row["raw_angular_mmd"]
            writer.writerow({name: row[name] for name in CSV_COLUMNS})


def _angular_features(inputs, w_fourvectors):
    booster = Booster(inputs[..., :8], w_fourvectors)
    valid = booster.valid_rest_frame_mask()
    angles = torch.stack(booster.lep_theta_phi_in_w_rest(), dim=-1)
    return angular_mmd_features(angles), valid


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


def evaluate_checkpoint(checkpoint, features, targets, manifest, device, *, block_size):
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
    partitions = [
        list(range(start, start + PARTITION_ROWS))
        for start in range(0, len(features), PARTITION_ROWS)
    ]
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
    truth_angular, truth_valid = _angular_features(features, targets[:, :8])
    pred_angular, pred_valid = _angular_features(features, prediction)
    if not truth_valid.all():
        raise RuntimeError("manifest contains a truth row without a valid rest frame")
    angular_metrics = angular_checkpoint_metrics(
        pred_angular,
        truth_angular,
        pred_valid,
        feature_bandwidths=manifest["feature_bandwidths"],
        partitions=partitions,
        condition=condition,
        raw_mmd_kwargs=model._mmd_kwargs("angular"),
        block_size=block_size,
    )
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
        truth_angles, truth_valid = _angular_features(all_features, all_targets[:, :8])
        filtered_indices = torch.nonzero(truth_valid, as_tuple=False).flatten()
        manifest = build_manifest(
            filtered_indices.numpy(), truth_angles[truth_valid], seed=args.seed
        )
        args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    selected = np.asarray(manifest["validation_indices"], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        evaluate_checkpoint(
            checkpoint,
            features[selected],
            targets[selected],
            manifest,
            device,
            block_size=args.block_size,
        )
        for checkpoint in checkpoints
    ]
    write_metrics_csv(args.output, rows)


if __name__ == "__main__":
    main()
