import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data.preprocessing import neural_input_features_numpy


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "visualize.ipynb"


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def load_cell_namespace(test_features, mean=None, scale=None):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "input-gaussianization")
    mean = np.arange(24, dtype=np.float64) / 10.0 if mean is None else mean
    scale = np.arange(1, 25, dtype=np.float64) / 10.0 if scale is None else scale
    norm = SimpleNamespace(mean=FakeTensor(mean), std=FakeTensor(scale))
    namespace = {
        "dm": SimpleNamespace(X=test_features),
        "model": SimpleNamespace(model=SimpleNamespace(norm=norm)),
        "train_features": test_features,
        "np": np,
        "pd": pd,
        "plt": plt,
        "display": lambda value: None,
    }
    exec("".join(cell["source"]), namespace)
    return notebook, cell, namespace


def histogram_total(artist):
    vertices = artist.get_path().vertices
    horizontal_segments = np.diff(vertices[:, 0]) != 0.0
    return vertices[:-1, 1][horizontal_segments].sum()


def sample_features():
    features = (np.arange(4 * 22).reshape(4, 22) + 1).astype(np.float64)
    features[0, 8:12] = 0.0
    features[1, 12:16] = 0.0
    features[:, 20] = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]
    features[:, 21] = [-np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
    return features


def test_uses_production_transform_checkpoint_stats_and_preserves_raw_inputs():
    raw = sample_features()
    raw_before = raw.copy()

    _, cell, namespace = load_cell_namespace(raw)
    expected = neural_input_features_numpy(raw)

    source = "".join(cell["source"])
    assert "from data.preprocessing import neural_input_features_numpy" in source
    assert "def _gaussianized_input_features" not in source
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_allclose(namespace["test_candidate"], expected)
    np.testing.assert_allclose(namespace["checkpoint_mean"], np.arange(24) / 10.0)
    np.testing.assert_allclose(namespace["checkpoint_std"], np.arange(1, 25) / 10.0)
    np.testing.assert_allclose(
        namespace["features_z"],
        (expected - namespace["checkpoint_mean"]) / namespace["checkpoint_std"],
    )
    plt.close("all")


def test_jet_summaries_exclude_padding_and_scalar_summaries_use_all_events():
    raw = sample_features()
    _, _, namespace = load_cell_namespace(raw, mean=np.zeros(24), scale=np.ones(24))
    transformed = neural_input_features_numpy(raw)
    summary = namespace["summary_df"]
    jet0_padding = (raw[:, 8:12] == 0.0).all(axis=1)
    jet1_padding = (raw[:, 12:16] == 0.0).all(axis=1)
    jet0_present = ~jet0_padding
    jet1_present = ~jet1_padding

    assert summary.loc["lep+ px", "n_events"] == len(raw)
    assert summary.loc["MET px", "n_events"] == len(raw)
    assert summary.loc["jet0 px", "n_events"] == np.count_nonzero(jet0_present)
    assert summary.loc["jet1 px", "n_events"] == np.count_nonzero(jet1_present)
    assert summary.loc["jet0 px", "n_present"] == np.count_nonzero(jet0_present)
    assert summary.loc["jet0 px", "n_padding"] == np.count_nonzero(jet0_padding)
    assert summary.loc["jet1 px", "n_present"] == np.count_nonzero(jet1_present)
    assert summary.loc["jet1 px", "n_padding"] == np.count_nonzero(jet1_padding)
    assert summary.loc["jet0 px", "padding_fraction"] == np.mean(jet0_padding)
    assert summary.loc["jet1 px", "padding_fraction"] == np.mean(jet1_padding)
    assert summary.loc["jet0 px", "z_mean"] == np.mean(transformed[jet0_present, 8])
    assert summary.loc["jet1 px", "z_mean"] == np.mean(transformed[jet1_present, 12])
    assert len(summary) == 20
    plt.close("all")


def test_mixed_jet_histograms_use_dataset_fractions_and_include_padding_location():
    raw = sample_features()
    mean = np.zeros(24)
    mean[8:16] = 700.0
    scale = np.full(24, 100.0)

    _, _, namespace = load_cell_namespace(raw, mean=mean, scale=scale)
    gaussian_figure = plt.figure(plt.get_fignums()[-2])
    normal = namespace["scipy_stats"].norm

    for axis in gaussian_figure.axes[8:16]:
        present_artist, padding_artist = axis.patches
        present_fraction = 3.0 / len(raw)
        padding_fraction = 1.0 / len(raw)
        reference_line = axis.lines[0]
        reference_x = reference_line.get_xdata()
        bin_width = np.diff(axis.get_xlim()) / 60.0

        assert np.isclose(histogram_total(present_artist), present_fraction)
        assert np.isclose(histogram_total(padding_artist), padding_fraction)
        assert axis.get_xlim()[0] < -7.0
        assert axis.get_xlim()[1] >= 6.0
        np.testing.assert_allclose(
            reference_line.get_ydata(),
            normal.pdf(reference_x) * bin_width * present_fraction,
        )

    scalar_axis = gaussian_figure.axes[0]
    scalar_reference = scalar_axis.lines[0]
    scalar_x = scalar_reference.get_xdata()
    scalar_bin_width = np.diff(scalar_axis.get_xlim()) / 60.0
    np.testing.assert_allclose(
        scalar_reference.get_ydata(),
        normal.pdf(scalar_x) * scalar_bin_width,
    )
    plt.close("all")


def test_fully_padded_jet_slot_has_visible_padding_and_no_present_event_summaries():
    raw = sample_features()
    raw[:, 12:16] = 0.0

    _, _, namespace = load_cell_namespace(raw, mean=np.zeros(24), scale=np.ones(24))
    summary = namespace["summary_df"].loc[
        ["jet1 px", "jet1 py", "jet1 pz", "jet1 log1p(E)"]
    ]
    distribution_metrics = [
        "z_mean", "z_std", "skew", "kurtosis", "|z|>2 [%]", "KS p-value"
    ]
    gaussian_figure = plt.figure(plt.get_fignums()[-2])
    padding_artists = [axis.patches[0] for axis in gaussian_figure.axes[12:16]]
    legend_labels = {
        text.get_text() for text in gaussian_figure.legends[0].get_texts()
    }

    assert (summary["n_events"] == 0).all()
    assert (summary["n_present"] == 0).all()
    assert (summary["n_padding"] == len(raw)).all()
    assert (summary["padding_fraction"] == 1.0).all()
    assert summary[distribution_metrics].isna().all().all()
    assert all(
        np.allclose(artist.get_edgecolor(), matplotlib.colors.to_rgba("tab:blue"))
        for artist in padding_artists
    )
    assert all(
        np.isclose(histogram_total(artist), 1.0)
        for artist in padding_artists
    )
    assert {"Present fraction / bin", "Padding fraction / bin"} <= legend_labels
    plt.close("all")


def test_periodic_features_have_separate_geometric_diagnostics():
    _, _, namespace = load_cell_namespace(sample_features())
    angular = namespace["angular_summary_df"]
    input_figure = plt.figure(plt.get_fignums()[-2])
    angular_figure = plt.figure(plt.get_fignums()[-1])
    periodic_axes = input_figure.axes[20:24]

    assert list(angular.index) == ["dphi_ll", "dphi_llmet"]
    assert angular["all_finite"].all()
    assert angular["within_bounds"].all()
    assert np.all(angular["max_unit_circle_error"] < 1e-12)
    assert len(input_figure.axes) == 24
    assert all(axis.patches for axis in periodic_axes)
    assert all(not axis.lines for axis in periodic_axes)
    assert all(np.allclose(axis.get_xlim(), (-1.05, 1.05)) for axis in periodic_axes)
    assert all(
        label.get_fontsize() == 8
        for axis in angular_figure.axes
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
    )
    assert not any("sin(" in name or "cos(" in name for name in namespace["summary_df"].index)
    assert "KS p-value" not in angular.columns
    plt.close("all")
