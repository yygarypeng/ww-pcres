#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import model.losses as loss_module
from data.load_data import load_data
from model import LightningWBoson


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    epoch: int
    global_step: int


def weighted_mean(values):
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        raise ValueError("cannot average values with zero total weight")
    return sum(value * weight for value, weight in values) / total_weight


def tensor_batch(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def valid_mmd_rows(prediction, truth, condition, *, local):
    valid = torch.isfinite(prediction).reshape(prediction.shape[0], -1).all(dim=1)
    valid &= torch.isfinite(truth).reshape(truth.shape[0], -1).all(dim=1)
    if local:
        valid &= torch.isfinite(condition).reshape(condition.shape[0], -1).all(dim=1)
    return valid


def per_bandwidth_mmd(
    prediction,
    truth,
    condition,
    *,
    local,
    feature_kernel,
    condition_kernel,
    feature_bandwidth_multipliers,
    condition_bandwidth_multipliers,
):
    return [
        loss_module.compute_local_mmd(
            prediction,
            truth,
            condition,
            local=local,
            feature_kernel=feature_kernel,
            condition_kernel=condition_kernel,
            feature_bandwidth_multipliers=[multiplier],
            condition_bandwidth_multipliers=condition_bandwidth_multipliers,
        )
        for multiplier in feature_bandwidth_multipliers
    ]


def per_bandwidth_mmd_diagnostics(
    prediction,
    truth,
    condition,
    loss_transform=None,
    **kwargs,
):
    prediction = prediction.detach().requires_grad_(True)
    truth = truth.detach()
    condition = condition.detach()
    diagnostics = []
    with torch.enable_grad():
        for mmd2 in per_bandwidth_mmd(prediction, truth, condition, **kwargs):
            loss = loss_module.transform_mmd_loss(mmd2, **(loss_transform or {}))
            gradient = torch.autograd.grad(loss, prediction)[0] * prediction.shape[0]
            diagnostics.append(
                (
                    mmd2.detach(),
                    loss.detach(),
                    gradient.square().sum().detach(),
                    gradient.numel(),
                )
            )
    return diagnostics


def aggregate_bandwidth_diagnostics(batches):
    mmd2 = weighted_mean([(raw_value, rows) for raw_value, _, _, _, rows in batches])
    loss = weighted_mean([(value, rows) for _, value, _, _, rows in batches])
    gradient_squared_sum = sum(squared_sum for _, _, squared_sum, _, _ in batches)
    gradient_elements = sum(elements for _, _, _, elements, _ in batches)
    return mmd2, loss, (gradient_squared_sum / gradient_elements) ** 0.5


def _checkpoint_metadata(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"]), int(checkpoint["global_step"])


def discover_unique_checkpoints(paths, metadata_loader=_checkpoint_metadata):
    grouped = defaultdict(list)
    for path in paths:
        path = Path(path)
        grouped[metadata_loader(path)].append(path)

    unique = []
    duplicates = {}
    for (epoch, global_step), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda path: (path.name == "last.ckpt", path.name))
        canonical = candidates[0]
        unique.append(CheckpointInfo(canonical, epoch, global_step))
        duplicates.update({duplicate: canonical for duplicate in candidates[1:]})
    return unique, duplicates


def result_label(checkpoint):
    return f"epoch-{checkpoint.epoch}-step-{checkpoint.global_step}"


def _capture_mmd_inputs(loss_fn, *args, **kwargs):
    captured = []

    def capture(prediction, truth, condition, **mmd_kwargs):
        captured.append((prediction, truth, condition, mmd_kwargs))
        return prediction.sum() * 0.0

    with patch.object(loss_module, "compute_local_mmd", side_effect=capture):
        loss_fn(*args, **kwargs)
    if not captured:
        return None
    if len(captured) != 1:
        raise RuntimeError(f"expected one MMD feature set, captured {len(captured)}")
    return captured[0]


def _batch_feature_inputs(model, features, targets):
    predictions, auxiliary = model(features, return_aux=True)
    condition = auxiliary["cond"]
    return {
        "alpha": _capture_mmd_inputs(
            loss_module.alpha_mmd,
            features,
            targets,
            predictions,
            condition,
            **model._mmd_kwargs("alpha"),
        ),
        "mass": _capture_mmd_inputs(
            loss_module.mass_mmd,
            features,
            targets,
            predictions,
            condition,
            model.mass_mmd_center,
            model.mass_mmd_scale,
            **model._mmd_kwargs("mass"),
        ),
        "angular": _capture_mmd_inputs(
            loss_module.angular_mmd,
            features,
            targets,
            predictions,
            condition,
            **model._mmd_kwargs("angular"),
        ),
    }


def evaluate_checkpoint(checkpoint, features, targets, batch_size, device):
    model = (
        LightningWBoson.load_from_checkpoint(
            checkpoint.path,
            map_location="cpu",
            weights_only=False,
        )
        .eval()
        .to(device)
    )
    feature_names = ("alpha", "mass", "angular")
    multipliers = {
        name: tuple(model.mmd_config[name]["bandwidth_multipliers"]) for name in feature_names
    }
    loss_transform = model.mmd_config.get("loss_transform")
    accumulated = {name: [[] for _ in multipliers[name]] for name in feature_names}

    for start in range(0, len(features), batch_size):
        with torch.no_grad():
            batch_features = tensor_batch(features[start : start + batch_size], device)
            batch_targets = tensor_batch(targets[start : start + batch_size], device)
            captured_features = _batch_feature_inputs(
                model,
                batch_features,
                batch_targets,
            )
        for name, captured in captured_features.items():
            if captured is None:
                continue
            prediction, truth, condition, kwargs = captured
            valid_rows = valid_mmd_rows(
                prediction,
                truth,
                condition,
                local=kwargs["local"],
            )
            prediction = prediction[valid_rows]
            truth = truth[valid_rows]
            condition = condition[valid_rows]
            if prediction.shape[0] == 0:
                continue
            diagnostics = per_bandwidth_mmd_diagnostics(
                prediction,
                truth,
                condition,
                loss_transform=loss_transform,
                **kwargs,
            )
            row_count = prediction.shape[0]
            for index, (mmd2, loss, gradient_squared_sum, gradient_elements) in enumerate(
                diagnostics
            ):
                accumulated[name][index].append(
                    (
                        float(mmd2),
                        float(loss),
                        float(gradient_squared_sum),
                        gradient_elements,
                        row_count,
                    )
                )

    results = {
        name: [aggregate_bandwidth_diagnostics(batches) for batches in bandwidth_batches]
        for name, bandwidth_batches in accumulated.items()
    }
    return multipliers, results, loss_transform


def print_results(all_results):
    for label, (multipliers, results, loss_transform) in all_results.items():
        print(label)
        for name in ("alpha", "mass", "angular"):
            diagnostics = results[name]
            pairs = "  ".join(
                f"{multiplier:g}:mmd2={mmd2:.8g},loss={loss:.8g},"
                f"prediction_gradient_rms={gradient_rms:.8g}"
                for multiplier, (mmd2, loss, gradient_rms) in zip(multipliers[name], diagnostics)
            )
            mixed_mmd2 = sum(mmd2 for mmd2, _, _ in diagnostics) / len(diagnostics)
            mixed_loss = loss_module.transform_mmd_loss(
                torch.as_tensor(mixed_mmd2),
                **(loss_transform or {}),
            )
            print(
                f"  {name:<7} {pairs}  mixed_mmd2={mixed_mmd2:.8g},"
                f"mixed_loss={float(mixed_loss):.8g}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each configured MMD feature bandwidth. Checkpoints must be trusted local "
            "artifacts because loading requires weights_only=False."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", default="ggF_val")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints, duplicates = discover_unique_checkpoints(args.checkpoint_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {args.checkpoint_dir}")
    for duplicate, canonical in sorted(duplicates.items()):
        print(f"Skipping duplicate checkpoint {duplicate.name} ({canonical.name})")

    features, targets, _, _ = load_data(args.data_path, categories=[args.split])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    for checkpoint in checkpoints:
        checkpoint_data = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
        default_batch_size = int(checkpoint_data["hyper_parameters"].get("batch_size", 512))
        batch_size = args.batch_size or default_batch_size
        label = result_label(checkpoint)
        print(f"Evaluating {label} on {device} with batch size {batch_size}...")
        all_results[label] = evaluate_checkpoint(
            checkpoint,
            features,
            targets,
            batch_size,
            device,
        )
    print_results(all_results)


if __name__ == "__main__":
    main()
