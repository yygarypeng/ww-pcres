import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from model import LightningWBoson
from train import load_config


def find_checkpoint(saved_path, checkpoint=None):
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(
        Path(saved_path).glob("**/checkpoints/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {saved_path}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Export a Lightning checkpoint to ONNX")
    parser.add_argument("--config", "-c", default=str(REPO_ROOT / "configs/config.yaml"), help="Path to YAML config file")
    parser.add_argument("--checkpoint", help="Specific .ckpt file to export")
    parser.add_argument("--output", default="hww_pcres_regressor.onnx", help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = find_checkpoint(cfg["paths"]["saved_path"], args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    model = LightningWBoson.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        strict=False,
    )
    model.eval().to("cpu")

    input_dim = int(model.hparams.input_dim)
    example_input = torch.randn(args.batch_size, input_dim, device="cpu")
    output_path = Path(args.output)

    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=["inputs"],
        output_names=["outputs"],
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        opset_version=args.opset,
        dynamic_axes={
            "inputs": {0: "batch_size"},
            "outputs": {0: "batch_size"},
        },
    )

    print(f"ONNX model exported to {output_path}")

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid and has dynamic batch size")
    except Exception as err:
        print(f"ONNX validation error: {err}")


if __name__ == "__main__":
    main()
