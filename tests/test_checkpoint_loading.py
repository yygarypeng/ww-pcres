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
            input_dim=22,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(24, dtype=np.float32),
            std_scale_train=np.ones(24, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()

    @staticmethod
    def make_inputs():
        inputs = torch.randn(2, 22)
        for start in (0, 4, 8, 12):
            inputs[:, start + 3] = torch.linalg.vector_norm(
                inputs[:, start:start + 3], dim=1
            ) + torch.rand(2) + 0.1
        return inputs

    def test_loads_checkpoint_with_legacy_loss_weights_for_inference(self):
        model = self.make_model()
        inputs = self.make_inputs()
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

    def test_rejects_checkpoint_missing_preprocessing_version(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"].pop("input_preprocessing_version", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "unversioned.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_checkpoint_with_obsolete_preprocessing_version(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["input_preprocessing_version"] = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "obsolete-version.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_checkpoint_with_legacy_raw_input_width(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["input_dim"] = 18

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-input-width.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_legacy_normalization_and_embedding_shapes(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["state_dict"]["model.norm.mean"] = torch.zeros(22)
        checkpoint["state_dict"]["model.norm.std"] = torch.ones(22)
        checkpoint["state_dict"]["model.hl_embed.weight"] = torch.zeros(8, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-shapes.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_checkpoint_with_legacy_normalization_hyperparameters(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["std_mean_train"] = np.zeros(22, dtype=np.float32)
        checkpoint["hyper_parameters"]["std_scale_train"] = np.ones(22, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-statistics.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)


if __name__ == "__main__":
    unittest.main()
