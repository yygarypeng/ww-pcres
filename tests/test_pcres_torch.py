import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import pytorch_lightning as L
    from model.model import LightningWBoson

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class PcResTorchLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = Path("/root/work/ww-pcres/meta/logs/version_0/checkpoints/last.ckpt")

    def test_pure_torch_subprocess_without_lightning(self):
        code = f"""
import sys
# Block pytorch_lightning from being imported
sys.modules['pytorch_lightning'] = None

import torch
from pcres_torch import load_pcres_checkpoint

p = '{self.ckpt_path}'
model = load_pcres_checkpoint(p)
inputs = torch.tensor([[20., 0., 10., 25., -15., 5., -8., 20., 0., 0., 0., 0., 0., 0., 0., 0., 5., -3.]], dtype=torch.float32)

with torch.inference_mode():
    outputs = model(inputs)

assert outputs.shape == (1, 8), f"Wrong shape {{outputs.shape}}"
assert outputs.dtype == torch.float32, f"Wrong dtype {{outputs.dtype}}"
assert torch.isfinite(outputs).all(), "Outputs contains non-finite values"
assert 'pytorch_lightning' not in sys.modules or sys.modules['pytorch_lightning'] is None
print("SUBPROCESS_SUCCESS")
"""
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"Subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("SUBPROCESS_SUCCESS", res.stdout)

    @unittest.skipUnless(LIGHTNING_AVAILABLE, "PyTorch Lightning required for parity test")
    def test_parity_with_lightning_model(self):
        from pcres_torch import load_pcres_checkpoint

        if not self.ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {self.ckpt_path}")

        lightning_model = LightningWBoson.load_from_checkpoint(
            str(self.ckpt_path),
            map_location="cpu",
            weights_only=False,
        ).eval()

        torch_model = load_pcres_checkpoint(str(self.ckpt_path), map_location="cpu")

        # Test batch with padded jets and active jets
        x = torch.tensor(
            [
                [20.0, 0.0, 10.0, 25.0, -15.0, 5.0, -8.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -3.0],
                [10.0, -5.0, 15.0, 30.0, -10.0, 2.0, -5.0, 18.0, 12.0, 8.0, -4.0, 20.0, 0.0, 0.0, 0.0, 0.0, -2.0, 4.0],
                [5.0, 12.0, -8.0, 22.0, -8.0, -4.0, 12.0, 24.0, 10.0, -5.0, 3.0, 16.0, 6.0, 4.0, -2.0, 12.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        with torch.inference_mode():
            expected = lightning_model(x)
            actual = torch_model(x)

        torch.testing.assert_close(actual, expected)

    def test_roundtrip_synthetic_18_dim_checkpoint(self):
        from pcres_torch import PcResRegressor, load_pcres_checkpoint

        model = PcResRegressor(
            input_dim=18,
            d_model=16,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()

        ckpt = {
            "hyper_parameters": {
                "input_dim": 18,
                "d_model": 16,
                "num_heads": 2,
                "std_mean_train": np.zeros(18, dtype=np.float32),
                "std_scale_train": np.ones(18, dtype=np.float32),
                "attention_blocks": 1,
                "attention_dropout": 0.0,
                "decoder_dropout": 0.0,
            },
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        }

        x = torch.randn(2, 18)
        for s in (0, 4, 8, 12):
            x[:, s + 3] = torch.linalg.vector_norm(x[:, s : s + 3], dim=1) + 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synth18.ckpt"
            torch.save(ckpt, path)

            loaded = load_pcres_checkpoint(path)
            with torch.inference_mode():
                expected = model(x)
                actual = loaded(x)
            torch.testing.assert_close(actual, expected)

    def test_input_shape_validation(self):
        from pcres_torch import load_pcres_checkpoint

        if not self.ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {self.ckpt_path}")

        model = load_pcres_checkpoint(str(self.ckpt_path))

        with self.assertRaisesRegex(ValueError, "Expected input shape"):
            model(torch.randn(2, 10))


if __name__ == "__main__":
    unittest.main()
