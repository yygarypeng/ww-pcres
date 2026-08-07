import unittest
import unittest.mock
from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from train import train as train_module
from train.train import apply_cli_overrides, parse_args, prime_csv_metric_header


class TrainingOverrideTest(unittest.TestCase):
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
        x_train = np.zeros((3, 22), dtype=np.float32)
        y_train = np.zeros((3, 10), dtype=np.float32)
        x_val = np.ones((2, 22), dtype=np.float32)
        y_val = np.ones((2, 10), dtype=np.float32)
        x_test = np.full((2, 22), 2.0, dtype=np.float32)
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
                 train_module.data,
                 "compute_standardization_stats",
                 return_value=((np.zeros(22), np.ones(22)), None),
            ),
            unittest.mock.patch.object(
                train_module.data,
                "compute_mmd_condition_stats",
                return_value=(np.zeros(6), np.ones(6)),
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


if __name__ == "__main__":
    unittest.main()
