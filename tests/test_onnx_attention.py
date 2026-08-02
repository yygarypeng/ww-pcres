import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import layers
from model.model import WBosonRegressor

CONVERTER_PATH = Path(__file__).resolve().parents[1] / "onnx" / "convert_to_onnx.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location("convert_to_onnx", CONVERTER_PATH)
converter = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(converter)
Opset11MultiheadAttention = converter.Opset11MultiheadAttention
replace_multihead_attention_for_opset11 = converter.replace_multihead_attention_for_opset11


class Opset11MultiheadAttentionTest(unittest.TestCase):
    def test_cross_attention_block_is_removed(self):
        self.assertFalse(hasattr(layers, "CrossAttentionBlock"))

    def test_self_attention_matches_pytorch(self):
        torch.manual_seed(4)
        source = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).eval()
        replacement = Opset11MultiheadAttention(source).eval()
        inputs = torch.randn(2, 5, 8)
        mask = torch.tensor(
            [
                [False, False, True, False, False],
                [False, True, False, False, True],
            ]
        )

        expected = source(
            inputs, inputs, inputs, key_padding_mask=mask, need_weights=False
        )[0]
        actual, weights = replacement(
            inputs, inputs, inputs, key_padding_mask=mask, need_weights=False
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertIsNone(weights)

    def test_rejects_cross_attention(self):
        source = nn.MultiheadAttention(8, 2, batch_first=True)
        replacement = Opset11MultiheadAttention(source)
        query = torch.randn(2, 3, 8)
        context = torch.randn(2, 5, 8)

        with self.assertRaisesRegex(ValueError, "only supports self-attention"):
            replacement(query, context, context, need_weights=False)

    def test_small_regressor_exports_and_runs_with_dynamic_batch(self):
        import onnx
        import onnxruntime

        torch.manual_seed(5)
        input_dim = 18
        native_model = WBosonRegressor(
            input_dim=input_dim,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(input_dim, dtype=np.float32),
            std_scale_train=np.ones(input_dim, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()
        inputs = torch.randn(3, input_dim)
        inputs[0, 8:12] = 0.0
        with torch.no_grad():
            expected = native_model(inputs).numpy()

        export_model = copy.deepcopy(native_model)
        replace_multihead_attention_for_opset11(export_model)
        self.assertTrue(all(
            isinstance(block.mha, Opset11MultiheadAttention)
            for block in export_model.sa_blocks
        ))
        self.assertFalse(hasattr(export_model, "event_pool"))
        self.assertEqual(export_model.num_tokens, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "small_regressor.onnx"
            torch.onnx.export(
                export_model,
                torch.randn(1, input_dim),
                output_path,
                input_names=["inputs"],
                output_names=["outputs"],
                opset_version=11,
                dynamic_axes={
                    "inputs": {0: "batch_size"},
                    "outputs": {0: "batch_size"},
                },
            )
            onnx.checker.check_model(onnx.load(output_path))

            session = onnxruntime.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"],
            )
            actual = session.run(["outputs"], {"inputs": inputs.numpy()})[0]

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
