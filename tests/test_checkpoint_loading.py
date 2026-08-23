import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch

from model import LightningWBoson


class InferenceCheckpointLoadingTest(unittest.TestCase):
    @staticmethod
    def make_model():
        return LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()

    @staticmethod
    def make_inputs():
        inputs = torch.randn(2, 21)
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = (
                torch.linalg.vector_norm(inputs[:, start : start + 3], dim=1) + torch.rand(2) + 0.1
            )
        return inputs

    def test_current_preprocessing_checkpoint_round_trips(self):
        model = self.make_model()
        inputs = self.make_inputs()
        expected = model(inputs)
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "current.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(
                checkpoint_path,
                weights_only=False,
            ).eval()

        torch.testing.assert_close(loaded(inputs), expected)

    def test_angular_mmd_ramp_epochs_checkpoint_round_trips(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            angular_mmd_ramp_epochs=80,
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "angular-mmd-ramp.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

        self.assertEqual(model.angular_mmd_ramp_epochs, 80)
        self.assertEqual(loaded.angular_mmd_ramp_epochs, 80)
        self.assertEqual(dict(model.hparams)["angular_mmd_ramp_epochs"], 80)
        self.assertEqual(dict(loaded.hparams)["angular_mmd_ramp_epochs"], 80)

    def test_no_high_level_checkpoint_round_trips(self):
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
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = (
                torch.linalg.vector_norm(inputs[:, start : start + 3], dim=1) + torch.rand(2) + 0.1
            )
        expected = model(inputs)
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "no-high-level.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(
                checkpoint_path,
                weights_only=False,
            ).eval()

        torch.testing.assert_close(loaded(inputs), expected)
        self.assertIsNone(loaded.model.hl_embed)

    def test_rejects_checkpoint_with_invalid_normalization_statistics(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["std_mean_train"] = np.zeros(24, dtype=np.float32)
        checkpoint["hyper_parameters"]["std_scale_train"] = np.ones(24, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "invalid-statistics.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "normalization statistics must each contain"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_checkpoint_with_unsupported_input_width(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["input_dim"] = 22

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "unsupported-input-width.ckpt"
            torch.save(checkpoint, checkpoint_path)
            # Input dim check still active - should fail with ValueError
            with self.assertRaisesRegex(ValueError, "raw input contract requires input_dim"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)


if __name__ == "__main__":
    unittest.main()
