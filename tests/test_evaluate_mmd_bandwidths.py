import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import scripts.evaluate_mmd_bandwidths as mmd_script
from model.losses import compute_mmd
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
    def test_main_does_not_load_checkpoint_metadata_with_explicit_batch_size(self):
        checkpoint = CheckpointInfo(Path("model.ckpt"), 2, 10)
        args = SimpleNamespace(
            checkpoint_dir=Path("checkpoints"),
            data_path=Path("data.h5"),
            split="ggF_val",
            batch_size=4,
        )
        with (
            patch.object(mmd_script, "parse_args", return_value=args),
            patch.object(
                mmd_script,
                "discover_unique_checkpoints",
                return_value=([checkpoint], {}),
            ),
            patch.object(
                mmd_script,
                "load_data",
                return_value=(np.ones((1, 18)), np.ones((1, 10))),
            ),
            patch.object(mmd_script.torch, "load") as checkpoint_loader,
            patch.object(mmd_script, "evaluate_checkpoint", return_value={}) as evaluate,
            patch.object(mmd_script, "print_results"),
        ):
            mmd_script.main()

        checkpoint_loader.assert_not_called()
        assert evaluate.call_args.args[3] == 4

    def test_checkpoint_evaluation_applies_captured_valid_mask(self):
        class Model:
            mmd_config = {
                name: {"kernel": "imq", "bandwidths": [1.0]}
                for name in ("alpha", "mass", "angular")
            }

            def eval(self):
                return self

            def to(self, _device):
                return self

        prediction = torch.tensor([[0.0], [1.0]])
        truth = torch.tensor([[0.0], [2.0]])
        captured_features = {
            name: (
                prediction,
                truth,
                {"kernel": "imq", "bandwidths": [1.0], "valid_mask": torch.tensor([True, False])},
            )
            for name in ("alpha", "mass", "angular")
        }
        row_counts = []

        def diagnostics(prediction, truth, *, kernel, bandwidths):
            del truth, kernel, bandwidths
            row_counts.append(len(prediction))
            zero = prediction.new_zeros(())
            return [(zero, zero, prediction.numel())]

        checkpoint = CheckpointInfo(Path("model.ckpt"), epoch=1, global_step=2)
        with (
            patch.object(mmd_script.LightningWBoson, "load_from_checkpoint", return_value=Model()),
            patch.object(mmd_script, "_batch_feature_inputs", return_value=captured_features),
            patch.object(mmd_script, "per_bandwidth_mmd_diagnostics", side_effect=diagnostics),
        ):
            mmd_script.evaluate_checkpoint(
                checkpoint,
                np.zeros((2, 18)),
                np.zeros((2, 10)),
                batch_size=2,
                device=torch.device("cpu"),
            )

        self.assertEqual(row_counts, [1, 1, 1])

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

    def test_mixture_combines_single_bandwidths_before_the_square_root(self):
        """The mixture is a quadratic mean; averaging MMDs would understate it."""
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]

        individual = torch.stack(
            per_bandwidth_mmd(
                prediction,
                truth,
                kernel="imq",
                bandwidths=bandwidths,
            )
        )
        mixed = compute_mmd(
            prediction,
            truth,
            kernel="imq",
            bandwidths=bandwidths,
        )

        torch.testing.assert_close(individual.square().mean().sqrt(), mixed)
        self.assertLess(individual.mean().item(), mixed.item())

    def test_bandwidth_diagnostics_match_values_and_include_gradients(self):
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        kwargs = {
            "kernel": "imq",
            "bandwidths": [0.1, 1.0, 10.0],
        }

        diagnostics = per_bandwidth_mmd_diagnostics(prediction, truth, **kwargs)
        values = per_bandwidth_mmd(prediction, truth, **kwargs)

        for (mmd, gradient_squared_sum, gradient_elements), expected_value in zip(
            diagnostics, values
        ):
            self.assertTrue(torch.isfinite(mmd))
            self.assertTrue(torch.isfinite(gradient_squared_sum))
            self.assertGreaterEqual(float(gradient_squared_sum), 0.0)
            self.assertEqual(gradient_elements, prediction.numel())
            torch.testing.assert_close(mmd, expected_value)
        self.assertTrue(any(float(item[1]) > 0.0 for item in diagnostics))

    def test_bandwidth_diagnostics_remove_mean_reduction_gradient_scaling(self):
        prediction = torch.tensor([[0.0], [0.5], [1.0], [1.5]], requires_grad=True)
        truth = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
        kwargs = {
            "kernel": "imq",
            "bandwidths": [1.0],
        }

        value = per_bandwidth_mmd(prediction, truth, **kwargs)[0]
        expected_gradient = torch.autograd.grad(value, prediction)[0] * prediction.shape[0]
        _, gradient_squared_sum, gradient_elements = per_bandwidth_mmd_diagnostics(
            prediction, truth, **kwargs
        )[0]

        torch.testing.assert_close(gradient_squared_sum, expected_gradient.square().sum())
        self.assertEqual(gradient_elements, prediction.numel())

    def test_aggregate_bandwidth_diagnostics_uses_global_squared_rms(self):
        batches = [
            (2.0, 1.0, 1, 1),
            (8.0, 27.0, 3, 3),
        ]

        mmd, gradient_rms = mmd_script.aggregate_bandwidth_diagnostics(batches)

        self.assertEqual(mmd, 6.5)
        self.assertAlmostEqual(gradient_rms, 7.0**0.5)
        self.assertNotAlmostEqual(gradient_rms, 2.5)

    def test_mixed_mmd_is_the_quadratic_mean_of_the_bandwidths(self):
        # compute_mmd mixes as sqrt(mean MMD^2), so 0 and 3.75 mix to 3.75/sqrt(2),
        # not to their arithmetic mean of 1.875.
        diagnostics = [(0.0, 0.0), (3.75, 0.0)]
        multipliers = {name: (1.0, 2.0) for name in ("alpha", "mass", "angular")}
        results = {name: diagnostics for name in multipliers}

        with patch("builtins.print") as mock_print:
            print_results({"epoch-1-step-2": (multipliers, results)})

        output = "\n".join(call.args[0] for call in mock_print.call_args_list)
        self.assertIn(f"mixed_mmd={(3.75**2 / 2) ** 0.5:.8g}", output)
        self.assertNotIn("mixed_mmd=1.875", output)

    def test_printed_results_label_values_and_gradients(self):
        multipliers = {name: (1.0, 2.0) for name in ("alpha", "mass", "angular")}
        results = {name: [(0.25, 0.125), (0.5, 0.25)] for name in multipliers}

        with patch("builtins.print") as mock_print:
            print_results({"epoch-1-step-2": (multipliers, results)})

        output = "\n".join(call.args[0] for call in mock_print.call_args_list)
        self.assertIn("mmd=", output)
        self.assertIn("prediction_gradient_rms=", output)
        self.assertIn(f"mixed_mmd={((0.25**2 + 0.5**2) / 2) ** 0.5:.8g}", output)
        for line in output.splitlines()[1:]:
            self.assertNotIn(
                "prediction_gradient_rms",
                line.split("mixed_mmd=", maxsplit=1)[1],
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

    def test_valid_mmd_rows_matches_compute_mmd_filter(self):
        prediction = torch.tensor([[0.0], [float("nan")], [2.0]])
        truth = torch.tensor([[0.0], [1.0], [float("inf")]])
        mask = valid_mmd_rows(prediction, truth)

        torch.testing.assert_close(mask, torch.tensor([True, False, False]))
