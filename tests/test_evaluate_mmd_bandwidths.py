import unittest
from pathlib import Path

import numpy as np
import torch

from model.losses import compute_local_mmd
from scripts.evaluate_mmd_bandwidths import (
    _capture_mmd_inputs,
    CheckpointInfo,
    discover_unique_checkpoints,
    result_label,
    valid_mmd_rows,
    per_bandwidth_mmd,
    tensor_batch,
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
