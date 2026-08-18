import unittest
import unittest.mock
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer.states import TrainerFn

from train import train as train_module
from train.train import (
    DeferredEarlyStopping,
    apply_cli_overrides,
    build_training_callbacks,
    parse_args,
    prime_csv_metric_header,
)


class TrainingOverrideTest(unittest.TestCase):
    def test_parser_accepts_higgs_mass_weight(self):
        with unittest.mock.patch(
            "sys.argv", ["train.py", "--higgs-mass-weight", "4.5"]
        ):
            args = parse_args()

        self.assertEqual(args.higgs_mass_weight, 4.5)

    def test_higgs_mass_weight_override_changes_only_higgs_weight(self):
        config = {
            "parameters": {
                "loss_weights": {"huber": 300.0, "higgs_mass": 3.0},
            }
        }

        updated = apply_cli_overrides(config, Namespace(higgs_mass_weight=0.0))

        self.assertEqual(updated["parameters"]["loss_weights"]["higgs_mass"], 0.0)
        self.assertEqual(updated["parameters"]["loss_weights"]["huber"], 300.0)

    def test_absent_higgs_mass_weight_preserves_yaml_value(self):
        config = {"parameters": {"loss_weights": {"higgs_mass": 3.0}}}

        updated = apply_cli_overrides(config, Namespace(higgs_mass_weight=None))

        self.assertEqual(updated["parameters"]["loss_weights"]["higgs_mass"], 3.0)

    def test_rejects_invalid_higgs_mass_weight_override(self):
        for value in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                config = {"parameters": {"loss_weights": {"higgs_mass": 3.0}}}
                with self.assertRaisesRegex(ValueError, "higgs_mass_weight"):
                    apply_cli_overrides(config, Namespace(higgs_mass_weight=value))

    def test_run_training_routes_higgs_mass_parameters(self):
        params = {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1.0e-4,
            "loss_weights": {"higgs_mass": 3.0},
            "higgs_mass_target": 126.0,
            "higgs_mass_scale": 9.0,
            "higgs_mass_delta": 1.5,
            "d_model": 8,
            "n_heads": 2,
        }
        cfg = {"parameters": params}
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
                21,
                (np.zeros(21), np.ones(21)),
                (np.zeros(3), np.ones(3)),
                np.ones(3),
                (0.0, 1.0),
                np.ones(2),
                "unused-output",
                SimpleNamespace(resume_from=None),
            )

        self.assertEqual(model_class.call_args.kwargs["higgs_mass_target"], 126.0)
        self.assertEqual(model_class.call_args.kwargs["higgs_mass_scale"], 9.0)
        self.assertEqual(model_class.call_args.kwargs["higgs_mass_delta"], 1.5)

    def test_run_training_routes_angular_mmd_ramp_epochs(self):
        params = {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1.0e-4,
            "loss_weights": {"huber": 1.0},
            "angular_mmd_ramp_epochs": 80,
            "d_model": 8,
            "n_heads": 2,
        }
        cfg = {"parameters": params}
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
                21,
                (np.zeros(21), np.ones(21)),
                (np.zeros(3), np.ones(3)),
                np.ones(3),
                (0.0, 1.0),
                np.ones(2),
                "unused-output",
                SimpleNamespace(resume_from=None),
            )

        self.assertEqual(model_class.call_args.kwargs["angular_mmd_ramp_epochs"], 80)

    def test_applies_training_overrides(self):
        config = {
            "parameters": {"seed": 114},
            "paths": {"saved_path": "outputs/default"},
        }
        args = Namespace(
            saved_path="outputs/query-115",
            seed=115,
            epochs=12,
            max_events_per_category=5000,
        )

        updated = apply_cli_overrides(config, args)

        self.assertEqual(updated["parameters"]["seed"], 115)
        self.assertEqual(updated["parameters"]["epochs"], 12)
        self.assertEqual(updated["data"]["max_events_per_category"], 5000)
        self.assertEqual(updated["paths"]["saved_path"], "outputs/query-115")

    def test_none_overrides_preserve_configuration(self):
        config = {
            "parameters": {"seed": 114},
            "paths": {"saved_path": "outputs/default"},
        }
        args = Namespace(
            saved_path=None,
            seed=None,
            epochs=None,
            max_events_per_category=None,
        )

        updated = apply_cli_overrides(config, args)

        self.assertEqual(updated, config)

    def test_parser_exposes_no_pooling_mode(self):
        with unittest.mock.patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertFalse(hasattr(args, "pooling_mode"))

    def test_csv_header_primes_configured_gradient_cosines(self):
        writer = SimpleNamespace(metrics_keys=[])
        logger = SimpleNamespace(experiment=writer)
        model = SimpleNamespace(
            loss_weights={"huber": 1.0, "higgs_mass": 2.0, "dmet": 0.0},
            adaptive_loss_names=["higgs_mass"],
            log_loss_gradient_cosines=True,
        )

        prime_csv_metric_header(logger, model)

        self.assertIn("grad_cos/higgs_mass__total", writer.metrics_keys)
        self.assertIn("grad_cos/higgs_mass__rest", writer.metrics_keys)
        self.assertNotIn("grad_cos/huber__total", writer.metrics_keys)
        self.assertNotIn("grad_cos/huber__rest", writer.metrics_keys)


class DeferredEarlyStoppingTest(unittest.TestCase):
    def _fake_trainer(self, current_epoch):
        return SimpleNamespace(
            current_epoch=current_epoch,
            state=SimpleNamespace(fn=TrainerFn.FITTING),
            sanity_checking=False,
        )

    def test_skips_checks_before_start_epoch(self):
        callback = DeferredEarlyStopping(start_epoch=80, monitor="val_loss")

        with unittest.mock.patch.object(EarlyStopping, "_run_early_stopping_check") as check:
            callback.on_validation_end(self._fake_trainer(79), None)

        check.assert_not_called()

    def test_checks_at_start_epoch(self):
        callback = DeferredEarlyStopping(start_epoch=80, monitor="val_loss")

        with unittest.mock.patch.object(EarlyStopping, "_run_early_stopping_check") as check:
            callback.on_validation_end(self._fake_trainer(80), None)

        check.assert_called_once()

    def test_zero_start_epoch_checks_immediately(self):
        callback = DeferredEarlyStopping(start_epoch=0, monitor="val_loss")

        with unittest.mock.patch.object(EarlyStopping, "_run_early_stopping_check") as check:
            callback.on_validation_end(self._fake_trainer(0), None)

        check.assert_called_once()

    def test_rejects_negative_start_epoch(self):
        with self.assertRaisesRegex(ValueError, "start_epoch"):
            DeferredEarlyStopping(start_epoch=-1, monitor="val_loss")

    def test_build_training_callbacks_routes_ramp_epochs(self):
        callbacks = build_training_callbacks(
            {
                "angular_mmd_ramp_epochs": 80,
                "early_stopping_patience": 7,
                "early_stopping_min_delta": 0.01,
            }
        )

        early_stopping = callbacks[1]
        self.assertIsInstance(early_stopping, DeferredEarlyStopping)
        self.assertEqual(early_stopping.start_epoch, 80)
        self.assertEqual(early_stopping.patience, 7)
        self.assertEqual(early_stopping.min_delta, -0.01)

    def test_build_training_callbacks_defaults_to_zero_start_epoch(self):
        callbacks = build_training_callbacks({})

        self.assertEqual(callbacks[1].start_epoch, 0)


class TrainingScaleTest(unittest.TestCase):
    def test_mass_mmd_standardization_uses_training_truth_only(self):
        targets = np.zeros((3, 10), dtype=np.float32)
        targets[:, 8:10] = np.array(
            [[20.0, 30.0], [40.0, 50.0], [60.0, 70.0]],
            dtype=np.float32,
        )

        center, scale = train_module.compute_mass_mmd_standardization(targets)

        transformed = np.arcsinh((targets[:, 8:10].reshape(-1) / 80.4) ** 2)
        expected_center = np.median(transformed)
        q25, q75 = np.percentile(transformed, [25.0, 75.0])
        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(scale, (q75 - q25) / 1.349)

    def test_build_datamodule_computes_dmet_scales_from_training_split(self):
        x_train = np.zeros((3, 21), dtype=np.float32)
        y_train = np.zeros((3, 10), dtype=np.float32)
        x_val = np.ones((2, 21), dtype=np.float32)
        y_val = np.ones((2, 10), dtype=np.float32)
        x_test = np.full((2, 21), 2.0, dtype=np.float32)
        y_test = np.full((2, 10), 2.0, dtype=np.float32)
        expected_scales = np.array([2.0, 3.0], dtype=np.float32)
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
                 return_value=(np.zeros(21), np.ones(21)),
            ),
            unittest.mock.patch.object(
                train_module.data,
                "compute_mmd_condition_stats",
                 return_value=(np.zeros(3), np.ones(3)),
            ),
            unittest.mock.patch.object(
                train_module,
                "compute_dmet_scales",
                return_value=expected_scales,
            ) as compute_dmet_scales,
        ):
            result = train_module.build_datamodule(cfg, "unused.h5")

        datamodule.return_value.setup.assert_called_once_with()
        np.testing.assert_array_equal(compute_dmet_scales.call_args.args[0], x_train)
        np.testing.assert_array_equal(compute_dmet_scales.call_args.args[1], y_train)
        np.testing.assert_array_equal(result[-1], expected_scales)

    def test_build_datamodule_keeps_raw_inputs_and_fits_neural_stats_on_train_only(self):
        x_train = np.full((3, 21), 1.0, dtype=np.float32)
        y_train = np.zeros((3, 10), dtype=np.float32)
        x_val = np.full((2, 21), 2.0, dtype=np.float32)
        y_val = np.zeros((2, 10), dtype=np.float32)
        x_test = np.full((2, 21), 3.0, dtype=np.float32)
        y_test = np.zeros((2, 10), dtype=np.float32)
        neural_stats = (np.zeros(21, dtype=np.float32), np.ones(21, dtype=np.float32))
        mmd_stats = (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
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
            ) as compute_neural_input_stats,
            unittest.mock.patch.object(
                train_module.data,
                "compute_mmd_condition_stats",
                return_value=mmd_stats,
            ) as compute_mmd_condition_stats,
        ):
            result = train_module.build_datamodule(cfg, "unused.h5")

        datamodule.assert_called_once()
        dm_args = datamodule.call_args.args
        dm_kwargs = datamodule.call_args.kwargs
        np.testing.assert_array_equal(dm_args[0], x_train)
        np.testing.assert_array_equal(dm_kwargs["X_val"], x_val)
        np.testing.assert_array_equal(dm_kwargs["X_test"], x_test)
        compute_neural_input_stats.assert_called_once()
        compute_mmd_condition_stats.assert_called_once()
        np.testing.assert_array_equal(compute_neural_input_stats.call_args.args[0], x_train)
        np.testing.assert_array_equal(compute_mmd_condition_stats.call_args.args[0], x_train)
        self.assertEqual(result[1], 21)
        self.assertEqual(result[2][0].shape, (21,))
        self.assertEqual(result[2][1].shape, (21,))


if __name__ == "__main__":
    unittest.main()
