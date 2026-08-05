import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch

from model import LightningWBoson


class InferenceCheckpointLoadingTest(unittest.TestCase):
    def test_loads_checkpoint_with_legacy_loss_weights_for_inference(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()
        inputs = torch.randn(2, 18)
        expected = model(inputs)
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": {
                **dict(model.hparams),
                "loss_weights": {"w_mass_mmd": 1.0},
            },
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.ckpt"
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "unsupported loss_weights key.*w_mass_mmd"):
                LightningWBoson.load_from_checkpoint(
                    checkpoint_path,
                    weights_only=False,
                    strict=False,
                )
            loaded = LightningWBoson.load_for_inference(
                checkpoint_path,
                weights_only=False,
                strict=False,
            ).eval()

        torch.testing.assert_close(loaded(inputs), expected)


if __name__ == "__main__":
    unittest.main()
