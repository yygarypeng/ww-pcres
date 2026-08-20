import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import scripts.evaluate_mmd_bandwidths as mmd_script
from model.losses import compute_local_mmd
from scripts.evaluate_mmd_bandwidths import (
    CheckpointInfo,
    _capture_mmd_inputs,
    discover_unique_checkpoints,
    per_bandwidth_mmd,
    per_bandwidth_mmd_diagnostics,
    print_results,
    result_label,
    tensor_batch,
    valid_mmd_rows,
    weighted_mean,
)


class BandwidthEvaluationTest(unittest.TestCase):
    def test_capture_returns_none_when_loss_has_no_valid_feature_rows(self):
        def zero_without_mmd_call(prediction):
            return prediction.sum() * 0.0

        captured = _capture_mmd_inputs(zero_without_mmd_call, torch.zeros(2, 1))

        self.assertIsNone(captured)

    def test_tensor_batch_matches_training_float32_dtype(self):
        values = np.ones((2, 3), dtype=np.float64)

        batch = tensor_batch(values, torch.device("cpu"))

        self.assertEqual(batch.dtype, torch.float32)

    def test_weighted_mean_uses_valid_row_counts(self):
        self.assertEqual(weighted_mean([(2.0, 2), (8.0, 1)]), 4.0)

    def test_single_bandwidth_mean_matches_feature_kernel_mixture(self):
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        condition = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        feature_multipliers = [0.01, 0.1, 1.0, 10.0, 100.0]
        condition_multipliers = [0.1, 1.0, 10.0]

        individual = per_bandwidth_mmd(
            prediction,
            truth,
            condition,
            local=True,
            feature_kernel="imq",
            condition_kernel="imq",
            feature_bandwidth_multipliers=feature_multipliers,
            condition_bandwidth_multipliers=condition_multipliers,
        )
        mixed = compute_local_mmd(
            prediction,
            truth,
            condition,
            local=True,
            feature_kernel="imq",
            condition_kernel="imq",
            feature_bandwidth_multipliers=feature_multipliers,
            condition_bandwidth_multipliers=condition_multipliers,
        )

        torch.testing.assert_close(torch.stack(individual).mean(), mixed)

    def test_bandwidth_diagnostics_match_values_and_include_gradients(self):
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        condition = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        kwargs = {
            "local": True,
            "feature_kernel": "imq",
            "condition_kernel": "imq",
            "feature_bandwidth_multipliers": [0.1, 1.0, 10.0],
            "condition_bandwidth_multipliers": [0.1, 1.0, 10.0],
        }

        diagnostics = per_bandwidth_mmd_diagnostics(prediction, truth, condition, **kwargs)
        values = per_bandwidth_mmd(prediction, truth, condition, **kwargs)

        for (mmd2, loss, gradient_squared_sum, gradient_elements), expected_value in zip(
            diagnostics, values
        ):
            self.assertTrue(torch.isfinite(mmd2))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(gradient_squared_sum))
            self.assertGreaterEqual(float(gradient_squared_sum), 0.0)
            self.assertEqual(gradient_elements, prediction.numel())
            torch.testing.assert_close(mmd2, expected_value)
            torch.testing.assert_close(loss, expected_value)
        self.assertTrue(any(float(item[2]) > 0.0 for item in diagnostics))

    def test_bandwidth_diagnostics_remove_mean_reduction_gradient_scaling(self):
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]], requires_grad=True)
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        condition = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        kwargs = {
            "local": True,
            "feature_kernel": "imq",
            "condition_kernel": "imq",
            "feature_bandwidth_multipliers": [1.0],
            "condition_bandwidth_multipliers": [0.1, 1.0, 10.0],
        }

        value = per_bandwidth_mmd(prediction, truth, condition, **kwargs)[0]
        expected_gradient = torch.autograd.grad(value, prediction)[0] * prediction.shape[0]
        _, _, gradient_squared_sum, gradient_elements = per_bandwidth_mmd_diagnostics(
            prediction, truth, condition, **kwargs
        )[0]

        torch.testing.assert_close(gradient_squared_sum, expected_gradient.square().sum())
        self.assertEqual(gradient_elements, prediction.numel())

    def test_aggregate_bandwidth_diagnostics_uses_global_squared_rms(self):
        batches = [
            (2.0, 1.0, 1.0, 1, 1),
            (8.0, 3.0, 27.0, 3, 3),
        ]

        mmd2, loss, gradient_rms = mmd_script.aggregate_bandwidth_diagnostics(batches)

        self.assertEqual(mmd2, 6.5)
        self.assertEqual(loss, 2.5)
        self.assertAlmostEqual(gradient_rms, 7.0**0.5)
        self.assertNotAlmostEqual(gradient_rms, 2.5)

    def test_mixed_loss_equals_mean_mmd2(self):
        diagnostics = [(0.0, 0.0, 0.0), (3.75, 3.75, 0.0)]
        multipliers = {name: (1.0, 2.0) for name in ("alpha", "mass", "angular")}
        results = {name: diagnostics for name in multipliers}

        with patch("builtins.print") as mock_print:
            print_results({"epoch-1-step-2": (multipliers, results)})

        output = "\n".join(call.args[0] for call in mock_print.call_args_list)
        self.assertIn("mixed_mmd2=1.875", output)
        self.assertIn("mixed_loss=1.875", output)

    def test_printed_results_label_values_and_gradients(self):
        multipliers = {name: (1.0, 2.0) for name in ("alpha", "mass", "angular")}
        results = {name: [(0.25, 0.25, 0.125), (0.5, 0.5, 0.25)] for name in multipliers}

        with patch("builtins.print") as mock_print:
            print_results({"epoch-1-step-2": (multipliers, results)})

        output = "\n".join(call.args[0] for call in mock_print.call_args_list)
        self.assertIn("mmd2=", output)
        self.assertIn("loss=", output)
        self.assertIn("prediction_gradient_rms=", output)
        self.assertIn("mixed_mmd2=0.375", output)
        self.assertIn("mixed_loss=0.375", output)
        for line in output.splitlines()[1:]:
            self.assertNotIn(
                "prediction_gradient_rms",
                line.split("mixed_mmd2=", maxsplit=1)[1],
            )

    def test_help_warns_that_checkpoints_must_be_trusted(self):
        stdout = io.StringIO()
        with (
            patch("sys.argv", ["evaluate_mmd_bandwidths.py", "--help"]),
            redirect_stdout(stdout),
            self.assertRaises(SystemExit),
        ):
            mmd_script.parse_args()

        help_text = " ".join(stdout.getvalue().split())
        self.assertIn("trusted local", help_text)
        self.assertIn("weights_only=False", help_text)

    def test_discovers_unique_checkpoints_by_epoch_and_global_step(self):
        metadata = {
            Path("last.ckpt"): (147, 375032),
            Path("reg-epoch=130.ckpt"): (130, 331954),
            Path("reg-epoch=147.ckpt"): (147, 375032),
            Path("reg-epoch=143.ckpt"): (143, 364896),
        }

        discovered, duplicates = discover_unique_checkpoints(
            metadata.keys(),
            metadata_loader=metadata.__getitem__,
        )

        self.assertEqual(
            [(item.epoch, item.global_step) for item in discovered],
            [(130, 331954), (143, 364896), (147, 375032)],
        )
        self.assertEqual(duplicates, {Path("last.ckpt"): Path("reg-epoch=147.ckpt")})

    def test_result_label_distinguishes_steps_within_an_epoch(self):
        first = result_label(CheckpointInfo(Path("first.ckpt"), 5, 100))
        second = result_label(CheckpointInfo(Path("second.ckpt"), 5, 101))

        self.assertNotEqual(first, second)

    def test_valid_mmd_rows_matches_compute_local_mmd_filter(self):
        prediction = torch.tensor([[0.0], [float("nan")], [2.0]])
        truth = torch.tensor([[0.0], [1.0], [float("inf")]])
        condition = torch.tensor([[0.0], [1.0], [2.0]])

        mask = valid_mmd_rows(prediction, truth, condition, local=True)

        torch.testing.assert_close(mask, torch.tensor([True, False, False]))


if __name__ == "__main__":
    unittest.main()
