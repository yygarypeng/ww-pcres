import csv
import json

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch import nn

from model.losses import compute_local_mmd
from scripts.diagnose_angular_degradation import (
    CSV_COLUMNS,
    angular_checkpoint_metrics,
    blockwise_mmd_v,
    build_manifest,
    gradient_diagnostic_rows,
    manifest_partition_positions,
    partitioned_raw_mmd,
    save_angular_plots,
    validate_manifest,
    write_gradient_csv,
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
    filtered_indices = np.arange(4_300)
    truth_angles = _angles(len(filtered_indices))
    truth_valid = torch.ones(len(filtered_indices), dtype=torch.bool)
    truth_valid[::7] = False

    first = build_manifest(filtered_indices, truth_angles, truth_valid, seed=73)
    second = build_manifest(filtered_indices, truth_angles, truth_valid, seed=73)

    assert first == second
    json.dumps(first, allow_nan=False)
    validate_manifest(first, validation_size=len(filtered_indices))
    assert len(first["validation_indices"]) == 4096
    assert len(set(first["validation_indices"])) == 4096
    assert set(first["validation_indices"]) <= set(filtered_indices)
    assert any(not truth_valid[index] for index in first["validation_indices"])
    assert first["gradient_indices"] == first["validation_indices"][:512]
    assert len(first["partitions"]) == 8
    assert all(len(partition) == 512 for partition in first["partitions"])
    assert [index for part in first["partitions"] for index in part] == first["validation_indices"]
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
    truth = torch.tensor([[-0.4, 0.3], [0.1, 0.8], [0.4, -0.2], [0.9, 0.6]], dtype=torch.float64)
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


def test_blockwise_v_statistic_matches_unequal_count_reference():
    prediction = torch.tensor([[-0.5], [0.1], [0.8]], dtype=torch.float64)
    truth = torch.tensor([[-0.4], [0.0], [0.4], [0.9], [1.2]], dtype=torch.float64)
    bandwidths = torch.tensor([0.3, 0.8], dtype=torch.float64)

    def kernel(left, right):
        distance = torch.cdist(left, right).square().unsqueeze(0)
        squared = bandwidths.square().reshape(-1, 1, 1)
        return (squared / (squared + distance + 1.0e-16)).mean(dim=0)

    expected = (
        kernel(prediction, prediction).mean()
        + kernel(truth, truth).mean()
        - 2.0 * kernel(prediction, truth).mean()
    ).clamp_min(0.0)

    actual = blockwise_mmd_v(prediction, truth, bandwidths, block_size=2)

    torch.testing.assert_close(actual, expected, atol=1.0e-14, rtol=1.0e-12)


def test_invalid_prediction_and_truth_rows_use_independent_fixed_samples():
    truth = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]], dtype=torch.float64).repeat(
        1, 6
    )
    prediction = truth.clone()
    prediction[1] = torch.nan
    truth[4] = torch.nan
    prediction_valid = torch.tensor([True, False, True, True, True, True])
    truth_valid = torch.tensor([True, True, True, True, False, True])
    expected = blockwise_mmd_v(
        prediction[prediction_valid],
        truth[truth_valid],
        [0.5],
        block_size=2,
    )

    metrics = angular_checkpoint_metrics(
        prediction,
        truth,
        prediction_valid,
        truth_valid,
        feature_bandwidths={
            "joint": [0.5],
            "wplus": [0.5],
            "wminus": [0.5],
        },
        partitions=[[0, 1, 2], [3, 4, 5]],
        block_size=2,
    )

    assert metrics["pred_rest_frame_valid_fraction"] == pytest.approx(5 / 6)
    assert metrics["mmd_joint_fixed"] == pytest.approx(float(expected))


def _valid_manifest():
    selected = list(range(4096))
    return {
        "seed": 1,
        "validation_indices": selected,
        "gradient_indices": selected[:512],
        "partitions": [selected[start : start + 512] for start in range(0, 4096, 512)],
        "plot_bins": {"theta": [0.0, 1.0], "phi": [-1.0, 0.0, 1.0]},
        "feature_bandwidths": {
            "joint": [0.5],
            "wplus": [0.25, 0.5],
            "wminus": [1.0],
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest["validation_indices"].pop(), "4,096"),
        (lambda manifest: manifest["validation_indices"].__setitem__(1, 0), "unique"),
        (lambda manifest: manifest["validation_indices"].__setitem__(0, 5000), "in range"),
        (lambda manifest: manifest["gradient_indices"].reverse(), "gradient"),
        (lambda manifest: manifest["partitions"].reverse(), "partitions"),
        (lambda manifest: manifest["plot_bins"]["theta"].reverse(), "plot bins"),
        (lambda manifest: manifest["plot_bins"].__setitem__("theta", ["bad"]), "plot bins"),
        (
            lambda manifest: manifest["feature_bandwidths"]["joint"].__setitem__(0, 0.0),
            "bandwidth",
        ),
        (
            lambda manifest: manifest["feature_bandwidths"].__setitem__("joint", ["bad"]),
            "bandwidth",
        ),
    ],
)
def test_manifest_validation_rejects_broken_invariants(mutate, message):
    manifest = _valid_manifest()
    mutate(manifest)

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest, validation_size=4300)


def test_persisted_partitions_are_mapped_to_selected_panel_positions():
    manifest = _valid_manifest()
    manifest["validation_indices"] = [1000 + index for index in manifest["validation_indices"]]
    manifest["gradient_indices"] = manifest["validation_indices"][:512]
    manifest["partitions"] = [
        manifest["validation_indices"][start : start + 512] for start in range(0, 4096, 512)
    ]

    positions = manifest_partition_positions(manifest)

    assert positions == [list(range(start, start + 512)) for start in range(0, 4096, 512)]


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


def test_angular_plots_use_fixed_events_bins_names_and_close_figures(tmp_path, monkeypatch):
    prediction = _angles(5).numpy()
    truth = (_angles(5) + 0.1).numpy()
    theta_bins = np.linspace(0.0, np.pi, 5)
    phi_bins = np.linspace(-np.pi, np.pi, 7)
    calls = []

    def plotter(observables, title, **kwargs):
        calls.append((observables, title, kwargs))
        return plt.figure(), None

    monkeypatch.setattr("scripts.diagnose_angular_degradation.plot_angular_1d_grid", plotter)
    monkeypatch.setattr("scripts.diagnose_angular_degradation.plot_angular_2d_grid", plotter)
    figures_before = plt.get_fignums()

    paths = save_angular_plots(
        tmp_path,
        epoch=7,
        prediction=prediction,
        truth=truth,
        plot_bins={"theta": theta_bins, "phi": phi_bins},
    )

    assert [path.name for path in paths] == [
        "epoch_0007_angular_1d.png",
        "epoch_0007_angular_2d.png",
        "epoch_0007_mixed_sum_1d.png",
        "epoch_0007_mixed_sum_2d.png",
        "epoch_0007_mixed_diff_1d.png",
        "epoch_0007_mixed_diff_2d.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == figures_before
    assert len(calls) == 6
    np.testing.assert_allclose(calls[0][0][0]["pred"], prediction[:, 0] + prediction[:, 2])
    np.testing.assert_allclose(calls[0][0][0]["bins"], np.linspace(0.0, 2.0, 5))
    np.testing.assert_allclose(calls[0][0][2]["bins"], phi_bins / np.pi)
    np.testing.assert_allclose(calls[2][0][0]["pred"], prediction[:, 0] + prediction[:, 1])
    assert calls[2][2] == {"share_axes": True}
    assert calls[3][2] == {"shared_colorbar": True, "share_axes": True}


class _GradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_scale = nn.Parameter(torch.tensor([0.2, 0.4]))
        self.w_fourvec_scales = torch.ones(4)
        self.loss_weights = {"huber": 50.0, "sum": -2.0, "disabled": 0.0}
        self.seen_features = None

    def forward(self, features, return_aux=False):
        values = features[:, :1]
        prediction = torch.cat(
            [
                values * self.slot_scale[0].repeat(4),
                values * self.slot_scale[1].repeat(4),
            ],
            dim=1,
        )
        if return_aux:
            return prediction, {"cond": features.new_empty((len(features), 0))}
        return prediction

    def _compute_batch_losses(self, features, targets):
        self.seen_features = features.detach().clone()
        prediction = self(features)
        residual = prediction - targets[:, :8]
        huber = torch.nn.functional.huber_loss(residual, torch.zeros_like(residual))
        losses = {"huber": huber, "sum": self.slot_scale.sum()}
        return 50.0 * huber - 2.0 * losses["sum"], losses

    def _effective_loss_weights(self):
        return self.loss_weights


def test_gradient_rows_use_persisted_batch_and_report_weighted_reference_cosines():
    model = _GradientModel().train()
    features = torch.ones((600, 2))
    features[512:] = 9.0
    targets = torch.zeros((600, 8))
    indices = list(range(511, -1, -1))

    rows = gradient_diagnostic_rows(model, features, targets, indices, epoch=3)

    assert not model.training
    torch.testing.assert_close(model.seen_features, features[indices])
    assert [row["loss"] for row in rows] == [
        "huber",
        "sum",
        "huber_wplus",
        "huber_wminus",
    ]
    by_name = {row["loss"]: row for row in rows}
    assert by_name["huber"]["raw_loss"] == pytest.approx(0.05)
    assert by_name["huber_wplus"]["raw_loss"] == pytest.approx(0.02)
    assert by_name["huber_wminus"]["raw_loss"] == pytest.approx(0.08)
    assert by_name["huber"]["raw_loss"] == pytest.approx(
        0.5 * (by_name["huber_wplus"]["raw_loss"] + by_name["huber_wminus"]["raw_loss"])
    )
    assert by_name["huber"]["effective_weight"] == 50.0
    assert by_name["huber_wplus"]["effective_weight"] == 25.0
    assert by_name["huber_wminus"]["effective_weight"] == 25.0
    assert by_name["huber"]["weighted_gradient_l2"] == pytest.approx(5 * np.sqrt(5))
    assert by_name["sum"]["weighted_gradient_l2"] == pytest.approx(2 * np.sqrt(2))
    assert by_name["huber_wplus"]["weighted_gradient_l2"] == pytest.approx(5.0)
    assert by_name["huber_wminus"]["weighted_gradient_l2"] == pytest.approx(10.0)
    assert by_name["sum"]["cosine_huber_wplus"] == pytest.approx(1 / np.sqrt(2))
    assert by_name["sum"]["cosine_huber_wminus"] == pytest.approx(1 / np.sqrt(2))
    assert by_name["huber_wplus"]["cosine_huber_wplus"] == pytest.approx(1.0)
    assert by_name["huber_wplus"]["cosine_huber_wminus"] == pytest.approx(0.0)
    assert all(row["epoch"] == 3 for row in rows)


def test_gradient_csv_has_stable_schema(tmp_path):
    path = tmp_path / "gradients.csv"
    row = {
        "epoch": 2,
        "loss": "huber",
        "raw_loss": 0.5,
        "effective_weight": 50.0,
        "weighted_gradient_l2": 3.0,
        "cosine_huber_wplus": 0.25,
        "cosine_huber_wminus": -0.75,
    }

    write_gradient_csv(path, [row])

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(row)
        assert next(reader)["loss"] == "huber"
