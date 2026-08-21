import csv
import json

import numpy as np
import pytest
import torch

from model.losses import compute_local_mmd
from scripts.diagnose_angular_degradation import (
    CSV_COLUMNS,
    angular_checkpoint_metrics,
    blockwise_mmd_v,
    build_manifest,
    partitioned_raw_mmd,
    write_metrics_csv,
)


def _angles(rows):
    values = torch.arange(rows, dtype=torch.float64)
    return torch.stack(
        [
            torch.remainder(values * 0.17, torch.pi),
            torch.remainder(values * 0.31 + torch.pi, 2.0 * torch.pi) - torch.pi,
            torch.remainder(values * 0.23, torch.pi),
            torch.remainder(values * 0.41 + torch.pi, 2.0 * torch.pi) - torch.pi,
        ],
        dim=-1,
    )


def test_manifest_is_deterministic_serializable_and_preserves_fixed_selection():
    filtered_indices = np.arange(10_000, 14_300)
    truth_angles = _angles(len(filtered_indices))

    first = build_manifest(filtered_indices, truth_angles, seed=73)
    second = build_manifest(filtered_indices, truth_angles, seed=73)

    assert first == second
    json.dumps(first, allow_nan=False)
    assert len(first["validation_indices"]) == 4096
    assert len(set(first["validation_indices"])) == 4096
    assert set(first["validation_indices"]) <= set(filtered_indices)
    assert first["gradient_indices"] == first["validation_indices"][:512]
    assert len(first["partitions"]) == 8
    assert all(len(partition) == 512 for partition in first["partitions"])
    assert [index for part in first["partitions"] for index in part] == first[
        "validation_indices"
    ]
    assert set(first["plot_bins"]) == {"theta", "phi"}
    assert first["plot_bins"]["theta"][0] == 0.0
    assert first["plot_bins"]["theta"][-1] == pytest.approx(np.pi)
    assert first["plot_bins"]["phi"][0] == pytest.approx(-np.pi)
    assert first["plot_bins"]["phi"][-1] == pytest.approx(np.pi)
    assert set(first["feature_bandwidths"]) == {"joint", "wplus", "wminus"}
    assert all(
        np.isfinite(value) and value > 0.0
        for bandwidths in first["feature_bandwidths"].values()
        for value in bandwidths
    )


@pytest.mark.parametrize("kernel", ["imq", "rbf"])
def test_blockwise_v_statistic_matches_exact_kernel_matrix(kernel):
    prediction = torch.tensor(
        [[-0.5, 0.2], [0.0, 0.7], [0.6, -0.1], [1.0, 0.4]], dtype=torch.float64
    )
    truth = torch.tensor(
        [[-0.4, 0.3], [0.1, 0.8], [0.4, -0.2], [0.9, 0.6]], dtype=torch.float64
    )
    bandwidths = [0.3, 0.8, 1.7]

    expected = compute_local_mmd(
        prediction,
        truth,
        torch.empty((len(truth), 0), dtype=truth.dtype),
        local=False,
        feature_kernel=kernel,
        feature_bandwidths=bandwidths,
        estimator="v",
    )
    actual = blockwise_mmd_v(
        prediction,
        truth,
        bandwidths,
        kernel=kernel,
        block_size=2,
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-14, rtol=1.0e-12)


def test_invalid_predictions_remain_in_valid_fraction_denominator():
    truth = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]], dtype=torch.float64
    ).repeat(1, 6)
    prediction = truth.clone()
    prediction[1] = torch.nan
    prediction[4] = torch.inf
    valid = torch.tensor([True, False, True, True, False, True])

    metrics = angular_checkpoint_metrics(
        prediction,
        truth,
        valid,
        feature_bandwidths={
            "joint": [0.5],
            "wplus": [0.5],
            "wminus": [0.5],
        },
        partitions=[[0, 1, 2], [3, 4, 5]],
        block_size=2,
    )

    assert metrics["pred_rest_frame_valid_fraction"] == pytest.approx(4 / 6)
    assert metrics["mmd_joint_fixed"] == pytest.approx(0.0, abs=1.0e-14)


def test_partitioned_raw_mmd_uses_fixed_partitions_and_counts_negative_batches():
    truth = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    prediction = torch.tensor([[0.5], [0.0], [3.0], [4.0]])
    partitions = [[0, 1], [2, 3]]

    raw_mmd, negative_fraction, values = partitioned_raw_mmd(
        prediction,
        truth,
        torch.ones(4, dtype=torch.bool),
        partitions,
    )

    expected = [
        compute_local_mmd(
            prediction[partition],
            truth[partition],
            torch.empty((2, 0)),
            local=False,
        )
        for partition in partitions
    ]
    assert values == pytest.approx([float(value) for value in expected])
    assert raw_mmd == pytest.approx(float(torch.stack(expected).mean()))
    assert expected[0] < 0.0 < expected[1]
    assert negative_fraction == 0.5


def test_csv_schema_and_angular_weight_are_exact(tmp_path):
    output_path = tmp_path / "metrics.csv"
    row = {
        "epoch": 7,
        "val_loss": 1.5,
        "raw_angular_mmd": -0.25,
        "mmd_joint_fixed": 0.1,
        "mmd_wplus_fixed": 0.2,
        "mmd_wminus_fixed": 0.3,
        "huber_wplus": 0.4,
        "huber_wminus": 0.5,
        "pred_rest_frame_valid_fraction": 0.75,
        "negative_raw_mmd_batch_fraction": 0.125,
        "learning_rate": 0.0002,
    }

    write_metrics_csv(output_path, [row])

    with output_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CSV_COLUMNS)
        written = next(reader)
    assert float(written["weighted_angular_mmd"]) == -500.0
    assert list(written) == list(CSV_COLUMNS)
