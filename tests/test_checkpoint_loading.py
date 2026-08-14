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
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()

    @staticmethod
    def make_inputs():
        inputs = torch.randn(2, 21)
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

    def test_physics_start_epoch_checkpoint_round_trips_without_legacy_key(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            physics_start_epoch=20,
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
            checkpoint_path = Path(tmpdir) / "physics-warmup.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

        self.assertEqual(model.physics_start_epoch, 20)
        self.assertEqual(loaded.physics_start_epoch, 20)
        self.assertEqual(dict(model.hparams)["physics_start_epoch"], 20)
        self.assertNotIn("mmd_start_epoch", dict(model.hparams))

    def test_legacy_mmd_start_epoch_checkpoint_migrates(self):
        model = self.make_model()
        hyper_parameters = dict(model.hparams)
        hyper_parameters.pop("physics_start_epoch", None)
        hyper_parameters["mmd_start_epoch"] = 17
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": hyper_parameters,
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-mmd-warmup.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

        self.assertEqual(loaded.physics_start_epoch, 17)
        self.assertNotIn("mmd_start_epoch", dict(loaded.hparams))

    def test_checkpoint_rejects_both_start_epoch_names(self):
        model = self.make_model()
        hyper_parameters = dict(model.hparams)
        hyper_parameters["mmd_start_epoch"] = 17
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": hyper_parameters,
            "pytorch-lightning_version": L.__version__,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "conflicting-warmup.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "cannot both be set"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_mmd_transform_checkpoint_round_trips(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_config={"loss_transform": {"kind": "sqrt", "epsilon": "0.002"}},
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
            checkpoint_path = Path(tmpdir) / "mmd-transform.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

        expected = {"kind": "sqrt", "epsilon": 0.002}
        self.assertEqual(model.mmd_config["loss_transform"], expected)
        self.assertEqual(dict(model.hparams)["mmd_config"]["loss_transform"], expected)
        self.assertEqual(loaded.mmd_config["loss_transform"], expected)

    def test_legacy_checkpoint_without_mmd_transform_preserves_compatibility_mode(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["mmd_config"].pop("loss_transform", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-mmd.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

        self.assertNotIn("loss_transform", loaded.mmd_config)

    def test_angular_mmd_schedule_checkpoint_round_trips(self):
        schedule = {
            "initial_multiplier": 0.1,
            "hold_epochs": 10,
            "full_weight_epoch": 80,
        }
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            angular_mmd_schedule=schedule,
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
            checkpoint_path = Path(tmpdir) / "scheduled.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded = LightningWBoson.load_from_checkpoint(
                checkpoint_path,
                weights_only=False,
            )

        self.assertEqual(loaded.angular_mmd_schedule, schedule)

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
        checkpoint["hyper_parameters"]["input_dim"] = 22

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-input-width.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_legacy_normalization_shapes(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["state_dict"]["model.norm.mean"] = torch.zeros(24)
        checkpoint["state_dict"]["model.norm.std"] = torch.ones(24)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-shapes.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_legacy_six_input_hl_embedding(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["state_dict"]["model.hl_embed.weight"] = torch.zeros(8, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-hl-embedding.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_legacy_six_value_condition_buffers(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["state_dict"]["model.cond_norm.mean"] = torch.zeros(6)
        checkpoint["state_dict"]["model.cond_norm.std"] = torch.ones(6)
        checkpoint["hyper_parameters"]["mmd_cond_mean_train"] = np.zeros(6)
        checkpoint["hyper_parameters"]["mmd_cond_scale_train"] = np.ones(6)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-condition.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)

    def test_rejects_checkpoint_with_legacy_normalization_hyperparameters(self):
        model = self.make_model()
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        checkpoint["hyper_parameters"]["std_mean_train"] = np.zeros(24, dtype=np.float32)
        checkpoint["hyper_parameters"]["std_scale_train"] = np.ones(24, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy-statistics.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "retraining required"):
                LightningWBoson.load_from_checkpoint(checkpoint_path, weights_only=False)


if __name__ == "__main__":
    unittest.main()
