import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from model import LightningWBoson
from train import load_config


class Opset11MultiheadAttention(nn.Module):
    def __init__(self, mha):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = mha.dropout
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        if query is not key or key is not value:
            raise ValueError("Opset11MultiheadAttention only supports self-attention")

        batch_size, seq_len, _ = query.shape
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask, -1.0e9)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(context), None


def replace_multihead_attention_for_opset11(module):
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            replacement = Opset11MultiheadAttention(child)
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            replace_multihead_attention_for_opset11(child)


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
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = find_checkpoint(cfg["paths"]["saved_path"], args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    model = LightningWBoson.load_for_inference(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        strict=False,
    )
    model.eval().to("cpu")
    if args.opset <= 11:
        replace_multihead_attention_for_opset11(model)

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
