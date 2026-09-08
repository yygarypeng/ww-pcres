#!/usr/bin/env python3
import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.load_data import load_data
from model import LightningWBoson
from model.losses import higgs_mass_loss, invariant_mass2
from scripts.evaluate_mmd_bandwidths import (
    discover_unique_checkpoints,
    tensor_batch,
    weighted_mean,
)

QUANTILES = {
    "mass_median": 0.5,
    "mass_q16": 0.16,
    "mass_q84": 0.84,
    "mass_q025": 0.025,
    "mass_q975": 0.975,
}


def higgs_batch_metrics(predictions, *, target_mass, weight):
    predictions = predictions.detach().requires_grad_(True)
    with torch.enable_grad():
        raw_loss = higgs_mass_loss(
            predictions,
            target_mass=target_mass,
        )
        gradient = torch.autograd.grad(raw_loss, predictions)[0] * predictions.shape[0]

    pair = predictions.detach()[..., :4] + predictions.detach()[..., 4:8]
    mass2 = invariant_mass2(pair)
    if not torch.isfinite(mass2).all():
        raise ValueError("predicted Higgs mass-squared contains non-finite values")
    timelike = mass2 >= 0.0
    masses = torch.sqrt(mass2[timelike])
    distance = torch.abs(masses - target_mass)
    row_count = predictions.shape[0]
    gradient_squared_sum = gradient.square().sum()
    raw_gradient_rms = torch.sqrt(gradient_squared_sum / gradient.numel())

    return {
        "rows": row_count,
        "raw_higgs_loss": float(raw_loss.detach()),
        "weighted_higgs_loss": float((raw_loss.detach() * weight)),
        "prediction_gradient_rms": float(raw_gradient_rms.detach()),
        "weighted_prediction_gradient_rms": float(raw_gradient_rms.detach() * abs(weight)),
        "prediction_gradient_squared_sum": float(gradient_squared_sum.detach()),
        "weighted_prediction_gradient_squared_sum": float(
            gradient_squared_sum.detach() * weight**2
        ),
        "prediction_gradient_elements": gradient.numel(),
        "spacelike_count": int((~timelike).sum()),
        "within_10_count": int((distance <= 10.0).sum()),
        "within_20_count": int((distance <= 20.0).sum()),
        "timelike_masses": masses.cpu().tolist(),
    }


def aggregate_metrics(batches):
    if not batches:
        raise ValueError("cannot evaluate an empty split")
    rows = sum(batch["rows"] for batch in batches)
    if rows == 0:
        raise ValueError("cannot evaluate an empty split")

    result = {
        "rows": rows,
        "raw_higgs_loss": weighted_mean(
            [(batch["raw_higgs_loss"], batch["rows"]) for batch in batches]
        ),
        "weighted_higgs_loss": weighted_mean(
            [(batch["weighted_higgs_loss"], batch["rows"]) for batch in batches]
        ),
        "prediction_gradient_rms": (
            sum(batch["prediction_gradient_squared_sum"] for batch in batches)
            / sum(batch["prediction_gradient_elements"] for batch in batches)
        )
        ** 0.5,
        "weighted_prediction_gradient_rms": (
            sum(batch["weighted_prediction_gradient_squared_sum"] for batch in batches)
            / sum(batch["prediction_gradient_elements"] for batch in batches)
        )
        ** 0.5,
        "spacelike_fraction": sum(batch["spacelike_count"] for batch in batches) / rows,
        "within_10_fraction": sum(batch["within_10_count"] for batch in batches) / rows,
        "within_20_fraction": sum(batch["within_20_count"] for batch in batches) / rows,
    }
    masses = [mass for batch in batches for mass in batch["timelike_masses"]]
    if masses:
        mass_tensor = torch.tensor(masses, dtype=torch.float64)
        for name, quantile in QUANTILES.items():
            result[name] = float(torch.quantile(mass_tensor, quantile))
    else:
        result.update(dict.fromkeys(QUANTILES))
    return result


def evaluate_checkpoint(checkpoint, features, batch_size, device):
    if len(features) == 0:
        raise ValueError("cannot evaluate an empty split")
    model = (
        LightningWBoson.load_from_checkpoint(
            checkpoint.path,
            map_location="cpu",
            weights_only=False,
        )
        .eval()
        .to(device)
    )
    weight = model._effective_loss_weights()["higgs_mass"]
    batches = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            predictions = model(tensor_batch(features[start : start + batch_size], device))
            batches.append(
                higgs_batch_metrics(
                    predictions,
                    target_mass=model.higgs_mass_target,
                    weight=weight,
                )
            )
    return {
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.global_step,
        **aggregate_metrics(batches),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate checkpoint Higgs-mass constraints. Checkpoints must be trusted local "
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
        print(
            f"Skipping duplicate checkpoint {duplicate.name} ({canonical.name})",
            file=sys.stderr,
        )

    with redirect_stdout(sys.stderr):
        features, _, _, _ = load_data(args.data_path, categories=[args.split])
    if len(features) == 0:
        raise ValueError("cannot evaluate an empty split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for checkpoint in checkpoints:
        checkpoint_data = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
        default_batch_size = int(checkpoint_data["hyper_parameters"].get("batch_size", 512))
        batch_size = args.batch_size or default_batch_size
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        with redirect_stdout(sys.stderr):
            result = evaluate_checkpoint(checkpoint, features, batch_size, device)
        print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
