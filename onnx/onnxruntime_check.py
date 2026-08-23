import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from convert_to_onnx import find_checkpoint, make_valid_raw_inputs

from model import LightningWBoson
from train import load_config


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime output with PyTorch")
    parser.add_argument(
        "--config",
        "-c",
        default=str(REPO_ROOT / "configs/config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--checkpoint", help="Specific .ckpt file to compare against")
    parser.add_argument(
        "--onnx",
        default=str(REPO_ROOT / "onnx/models/hww_pcres_regressor.onnx"),
        help="ONNX model path",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Random comparison batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random input seed")
    args = parser.parse_args()

    import onnxruntime

    cfg = load_config(args.config)
    ckpt_path = find_checkpoint(cfg["paths"]["saved_path"], args.checkpoint)
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"Loading ONNX model from {onnx_path}")
    ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print(f"Comparing with PyTorch checkpoint: {ckpt_path}")
    pytorch_model = LightningWBoson.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
        weights_only=False,
        strict=False,
    )
    pytorch_model.eval()

    test_input = make_valid_raw_inputs(
        args.batch_size,
        input_dim=pytorch_model.hparams.input_dim,
        seed=args.seed,
    ).numpy()

    input_name = ort_session.get_inputs()[0].name
    ort_result = ort_session.run(None, {input_name: test_input})[0]

    with torch.no_grad():
        pytorch_output = (
            pytorch_model(torch.tensor(test_input, dtype=torch.float32)).detach().cpu().numpy()
        )

    diff = pytorch_output - ort_result
    max_abs_diff = float(np.max(np.abs(diff)))
    max_rel_diff = float(np.max(np.abs(diff) / (np.abs(pytorch_output) + 1e-16)))
    atol, rtol = 1e-3, 3e-3
    allclose = np.allclose(pytorch_output, ort_result, atol=atol, rtol=rtol)

    print(f"Input shape: {test_input.shape}")
    print(f"ONNX output shape: {ort_result.shape}")
    print(f"Maximum abs diff: {max_abs_diff}")
    print(f"Maximum rel diff: {max_rel_diff}")
    print(f"allclose(atol={atol}, rtol={rtol}): {allclose}")
    if not allclose:
        raise AssertionError("ONNX Runtime output differs from PyTorch beyond tolerance")


if __name__ == "__main__":
    main()
