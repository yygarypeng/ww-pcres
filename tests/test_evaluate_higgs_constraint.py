import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import scripts.evaluate_higgs_constraint as higgs_script
from scripts.evaluate_higgs_constraint import aggregate_metrics, higgs_batch_metrics
from scripts.evaluate_mmd_bandwidths import CheckpointInfo


def higgs_predictions(masses):
    predictions = torch.zeros(len(masses), 8, dtype=torch.float64)
    predictions[:, 3] = torch.as_tensor(masses, dtype=torch.float64) / 2.0
    predictions[:, 7] = torch.as_tensor(masses, dtype=torch.float64) / 2.0
    return predictions


class HiggsConstraintEvaluationTest(unittest.TestCase):
    def test_counts_shell_boundaries_and_spacelike_rows(self):
        predictions = higgs_predictions([105.0, 115.0, 125.0, 135.0, 145.0])
        predictions[0, 0] = 110.0

        metrics = higgs_batch_metrics(
            predictions,
            target_mass=125.0,
            weight=3.0,
        )

        self.assertEqual(metrics["rows"], 5)
        self.assertEqual(metrics["timelike_masses"], [115.0, 125.0, 135.0, 145.0])
        self.assertEqual(metrics["spacelike_count"], 1)
        self.assertEqual(metrics["within_10_count"], 3)
        self.assertEqual(metrics["within_20_count"], 4)

    def test_quantiles_and_weighted_loss_and_gradient_scaling(self):
        predictions = higgs_predictions([100.0, 120.0, 140.0, 160.0]).requires_grad_()

        metrics = higgs_batch_metrics(
            predictions,
            target_mass=125.0,
            weight=2.5,
        )
        result = aggregate_metrics([metrics])

        expected = torch.quantile(
            torch.tensor([100.0, 120.0, 140.0, 160.0], dtype=torch.float64),
            torch.tensor([0.5, 0.16, 0.84, 0.025, 0.975], dtype=torch.float64),
        )
        self.assertAlmostEqual(result["weighted_higgs_loss"], 2.5 * result["raw_higgs_loss"])
        self.assertAlmostEqual(
            result["weighted_prediction_gradient_rms"],
            2.5 * result["prediction_gradient_rms"],
        )
        for key, value in zip(
            ("mass_median", "mass_q16", "mass_q84", "mass_q025", "mass_q975"),
            expected.tolist(),
        ):
            self.assertAlmostEqual(result[key], value)

    def test_empty_timelike_predictions_return_null_quantiles(self):
        predictions = torch.tensor(
            [[10.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

        result = aggregate_metrics(
            [
                higgs_batch_metrics(
                    predictions,
                    target_mass=125.0,
                    weight=1.0,
                )
            ]
        )

        for key in ("mass_median", "mass_q16", "mass_q84", "mass_q025", "mass_q975"):
            self.assertIsNone(result[key])
        self.assertEqual(result["spacelike_fraction"], 1.0)

    def test_aggregation_weights_batch_metrics_by_rows(self):
        small = higgs_batch_metrics(
            higgs_predictions([125.0]),
            target_mass=125.0,
            weight=4.0,
        )
        large = higgs_batch_metrics(
            higgs_predictions([145.0, 145.0, 145.0]),
            target_mass=125.0,
            weight=4.0,
        )

        result = aggregate_metrics([small, large])

        self.assertEqual(result["rows"], 4)
        self.assertAlmostEqual(
            result["raw_higgs_loss"],
            (small["raw_higgs_loss"] + 3 * large["raw_higgs_loss"]) / 4,
        )
        self.assertAlmostEqual(
            result["prediction_gradient_rms"],
            (
                (small["prediction_gradient_rms"] ** 2 + 3 * large["prediction_gradient_rms"] ** 2)
                / 4
            )
            ** 0.5,
        )

    def test_gradient_rms_is_invariant_to_batch_partitioning(self):
        predictions = higgs_predictions([95.0, 115.0, 130.0, 170.0])
        kwargs = {"target_mass": 125.0, "weight": 3.0}

        single_batch = aggregate_metrics([higgs_batch_metrics(predictions, **kwargs)])
        partitioned = aggregate_metrics(
            [
                higgs_batch_metrics(predictions[:1], **kwargs),
                higgs_batch_metrics(predictions[1:3], **kwargs),
                higgs_batch_metrics(predictions[3:], **kwargs),
            ]
        )

        self.assertAlmostEqual(
            single_batch["prediction_gradient_rms"],
            partitioned["prediction_gradient_rms"],
        )
        self.assertAlmostEqual(
            partitioned["weighted_prediction_gradient_rms"],
            3.0 * partitioned["prediction_gradient_rms"],
        )

    def test_main_redirects_loader_chatter_to_stderr(self):
        checkpoint = CheckpointInfo(Path("model.ckpt"), 2, 10)
        args = SimpleNamespace(
            checkpoint_dir=Path("checkpoints"),
            data_path=Path("data.h5"),
            split="ggF_val",
            batch_size=4,
        )

        def noisy_load_data(*_args, **_kwargs):
            print("loader chatter")
            return np.ones((1, 21)), np.ones((1, 10)), None, None

        def noisy_evaluate_checkpoint(*_args, **_kwargs):
            print("model chatter")
            return result

        result = {"epoch": 2, "global_step": 10, "rows": 1}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(higgs_script, "parse_args", return_value=args),
            patch.object(
                higgs_script,
                "discover_unique_checkpoints",
                return_value=([checkpoint], {}),
            ),
            patch.object(higgs_script, "load_data", side_effect=noisy_load_data),
            patch.object(
                higgs_script.torch,
                "load",
                return_value={"hyper_parameters": {"batch_size": 8}},
            ),
            patch.object(
                higgs_script,
                "evaluate_checkpoint",
                side_effect=noisy_evaluate_checkpoint,
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            higgs_script.main()

        self.assertEqual(json.loads(stdout.getvalue()), result)
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        self.assertIn("loader chatter", stderr.getvalue())
        self.assertIn("model chatter", stderr.getvalue())

    def test_help_warns_that_checkpoints_must_be_trusted(self):
        stdout = io.StringIO()
        with (
            patch("sys.argv", ["evaluate_higgs_constraint.py", "--help"]),
            redirect_stdout(stdout),
            self.assertRaises(SystemExit),
        ):
            higgs_script.parse_args()

        help_text = " ".join(stdout.getvalue().split())
        self.assertIn("trusted local", help_text)
        self.assertIn("weights_only=False", help_text)

    def test_rejects_empty_aggregation(self):
        with self.assertRaisesRegex(ValueError, "empty split"):
            aggregate_metrics([])


if __name__ == "__main__":
    unittest.main()
