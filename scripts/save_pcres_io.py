import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train import train
from model import LightningWBoson

OUTPUT_NAMES = ("w0_px", "w0_py", "w0_pz", "w0_logE", "w1_px", "w1_py", "w1_pz", "w1_logE")


def find_latest_checkpoint(saved_path):
    candidates = list(Path(saved_path).expanduser().glob("**/checkpoints/*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {saved_path}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def plot_pcres_io(npz_path, max_scatter_points=50_000):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with np.load(npz_path) as data:
        outputs = data["outputs"]
        targets = data["targets"]

    n_features = min(outputs.shape[1], targets.shape[1], len(OUTPUT_NAMES))
    scatter_idx = np.linspace(0, len(outputs) - 1, min(len(outputs), max_scatter_points), dtype=int)
    print(f"Plotting parity and residuals for {n_features} features using {len(scatter_idx)} scatter points...")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n_features:
            ax.axis("off")
            continue

        pred = outputs[scatter_idx, idx]
        truth = targets[scatter_idx, idx]
        low = min(np.min(pred), np.min(truth))
        high = max(np.max(pred), np.max(truth))

        ax.scatter(truth, pred, s=3, alpha=0.25)
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(OUTPUT_NAMES[idx])
        ax.set_xlabel("Target")
        ax.set_ylabel("Output")

    parity_path = npz_path.with_name("pcres_io_parity.png")
    fig.savefig(parity_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n_features:
            ax.axis("off")
            continue

        residual = outputs[:, idx] - targets[:, idx]
        ax.hist(residual, bins=60, histtype="step", linewidth=1.5)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(OUTPUT_NAMES[idx])
        ax.set_xlabel("Output - target")
        ax.set_ylabel("Events")

    residual_path = npz_path.with_name("pcres_io_residuals.png")
    fig.savefig(residual_path, dpi=160)
    plt.close(fig)

    return parity_path, residual_path


def main():
    parser = argparse.ArgumentParser(description="Save model inputs, outputs, and targets")
    parser.add_argument("--config", "-c", default=str(REPO_ROOT / "configs/config.yaml"), help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    cfg = train.load_config(config_path)
    saved_path = Path(cfg["paths"]["saved_path"]).expanduser()
    checkpoint_path = find_latest_checkpoint(saved_path)
    output_path = saved_path / "pcres_io.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = train.main(train=False, config_path=config_path)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    if test_loader is None:
        raise RuntimeError("No test dataloader available. Check data.test_frac or explicit test split.")

    model = LightningWBoson.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
        strict=False,
    )
    model.eval().to(device)

    inputs_all = []
    outputs_all = []
    targets_all = []

    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            inputs_all.append(inputs.cpu().numpy())
            outputs_all.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    if not inputs_all:
        raise RuntimeError("Test dataloader is empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=np.concatenate(inputs_all, axis=0),
        outputs=np.concatenate(outputs_all, axis=0),
        targets=np.concatenate(targets_all, axis=0),
        checkpoint=str(checkpoint_path),
        config=str(config_path),
    )

    print(f"Saved model inputs/outputs to {output_path}")
    parity_path, residual_path = plot_pcres_io(output_path)
    print(f"Saved check plots to {parity_path} and {residual_path}")


if __name__ == "__main__":
    main()
