import unittest
import unittest.mock
from types import SimpleNamespace

import numpy as np
import pytest
from pytorch_lightning.callbacks import EarlyStopping

from train import train as train_module
from train.train import (
    build_training_callbacks,
    clean_training_output,
    prime_csv_metric_header,
)


class TrainingTest(unittest.TestCase):
    def test_run_training_routes_absolute_mmd_config(self):
        mmd_config = {
            "alpha": {"kernel": "imq", "bandwidths": [0.2, 0.4]},
            "mass": {"kernel": "imq", "bandwidths": [0.5]},
            "angular": {"kernel": "imq", "bandwidths": [0.05, 0.5, 5.0]},
        }
        params = {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1.0e-4,
            "loss_weights": {"angular_mmd": 2000.0},
            "d_model": 8,
            "n_heads": 2,
        }
        cfg = {"parameters": params, "mmd": mmd_config}
        datamodule = SimpleNamespace(train_dataloader=lambda: [object()], test_ds=None)

        with (
            unittest.mock.patch.object(train_module, "LightningWBoson") as model_class,
            unittest.mock.patch.object(
                train_module,
                "build_training_callbacks",
                return_value=[SimpleNamespace()],
            ),
            unittest.mock.patch.object(train_module, "clean_training_output"),
            unittest.mock.patch.object(train_module, "create_loggers", return_value=([], None)),
            unittest.mock.patch.object(train_module, "Trainer"),
        ):
            train_module.run_training(
                cfg,
                datamodule,
                18,
                (np.zeros(18), np.ones(18)),
                "unused-output",
                False,
            )

        self.assertEqual(model_class.call_args.kwargs["mmd_config"], mmd_config)

    def test_csv_header_primes_configured_gradient_cosines(self):
        writer = SimpleNamespace(metrics_keys=[])
        logger = SimpleNamespace(experiment=writer)
        model = SimpleNamespace(
            loss_weights={"w_fourvec": 1.0, "higgs_mass": 2.0, "dmet": 0.0},
            adaptive_loss_names=["higgs_mass"],
            log_loss_gradient_cosines=True,
        )

        prime_csv_metric_header(logger, model)

        self.assertIn("grad_cos/higgs_mass__total", writer.metrics_keys)
        self.assertIn("grad_cos/higgs_mass__rest", writer.metrics_keys)
        self.assertIn("grad_norm/higgs_mass", writer.metrics_keys)
        self.assertNotIn("grad_cos/w_fourvec__total", writer.metrics_keys)
        self.assertNotIn("grad_cos/w_fourvec__rest", writer.metrics_keys)
        self.assertNotIn("grad_norm/w_fourvec", writer.metrics_keys)


def test_clean_training_output_does_not_follow_directory_symlinks(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    link = tmp_path / "output-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        clean_training_output(link)

    assert marker.exists()


class TrainingCallbacksTest(unittest.TestCase):
    def test_build_training_callbacks_routes_early_stopping_settings(self):
        callbacks = build_training_callbacks(
            {
                "early_stopping_patience": 7,
                "early_stopping_min_delta": 0.01,
            }
        )

        early_stopping = callbacks[1]
        self.assertIsInstance(early_stopping, EarlyStopping)
        self.assertEqual(early_stopping.patience, 7)
        self.assertEqual(early_stopping.min_delta, -0.01)

    def test_build_training_callbacks_defaults_early_stopping_patience(self):
        self.assertEqual(build_training_callbacks({})[1].patience, 32)

    def test_checkpoint_save_top_k_defaults_and_config_override(self):
        callbacks = build_training_callbacks({})
        self.assertEqual(callbacks[0].save_top_k, 16)

        overridden = build_training_callbacks({"checkpoint_save_top_k": 3})
        self.assertEqual(overridden[0].save_top_k, 3)
        self.assertTrue(overridden[0].save_last)


class TrainingInputStatsTest(unittest.TestCase):
    def test_build_datamodule_keeps_raw_inputs_and_fits_neural_stats_on_train_only(self):
        x_train = np.full((3, 18), 1.0, dtype=np.float32)
        y_train = np.zeros((3, 10), dtype=np.float32)
        x_val = np.full((2, 18), 2.0, dtype=np.float32)
        y_val = np.zeros((2, 10), dtype=np.float32)
        x_test = np.full((2, 18), 3.0, dtype=np.float32)
        y_test = np.zeros((2, 10), dtype=np.float32)
        neural_stats = (np.zeros(18, dtype=np.float32), np.ones(18, dtype=np.float32))
        cfg = {"parameters": {"batch_size": 2}, "data": {}}

        with (
            unittest.mock.patch.object(
                train_module.data,
                "load_presplit_data",
                return_value=(x_train, y_train, x_val, y_val, x_test, y_test),
            ),
            unittest.mock.patch.object(train_module, "WBosonDataModule") as datamodule,
            unittest.mock.patch.object(
                train_module,
                "compute_neural_input_stats",
                return_value=neural_stats,
            ),
        ):
            result = train_module.build_datamodule(cfg, "unused.h5")

        datamodule.assert_called_once()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1], 18)
        self.assertEqual(result[2][0].shape, (18,))
        self.assertEqual(result[2][1].shape, (18,))
