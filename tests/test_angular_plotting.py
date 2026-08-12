import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pytest

from notebooks.plottingtool import plot_angular_1d_grid, plot_angular_2d_grid


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "visualize.ipynb"
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid escape sequence:DeprecationWarning"
)


def _observables():
    return [
        {
            "pred": np.array([0.2, 0.3, 1.2, np.nan]) * np.pi,
            "truth": np.array([0.2, 1.2, 1.3, 0.4]) * np.pi,
            "label": f"angle {index}",
            "bins": np.array([0.0, 1.0, 2.0, 3.0]),
            "log": index % 2 == 0,
            "vmax": 10,
        }
        for index in range(4)
    ]


def _atlas_label_count(fig):
    texts = list(fig.texts)
    texts.extend(text for ax in fig.axes for text in ax.texts)
    return sum("Internal Simulation" in text.get_text() for text in texts)


def test_plot_angular_1d_grid_returns_histogram_and_raw_ratio_axes():
    fig, (hist_axes, ratio_axes) = plot_angular_1d_grid(
        _observables(), "Angular distributions", share_axes=True
    )

    try:
        assert hist_axes.shape == (2, 2)
        assert ratio_axes.shape == (2, 2)
        assert hist_axes[0, 0].get_shared_x_axes().joined(
            hist_axes[0, 0], hist_axes[1, 1]
        )
        assert hist_axes[0, 0].get_shared_y_axes().joined(
            hist_axes[0, 0], hist_axes[1, 1]
        )
        assert not ratio_axes[0, 0].get_shared_y_axes().joined(
            ratio_axes[0, 0], hist_axes[0, 0]
        )

        ratio_line = next(
            line for line in ratio_axes[0, 0].lines if line.get_label() == "Pred/True"
        )
        np.testing.assert_allclose(ratio_line.get_xdata(), [0.5, 1.5])
        np.testing.assert_allclose(ratio_line.get_ydata(), [2.0, 0.5])
        assert np.isfinite(ratio_line.get_ydata()).all()
        assert ratio_axes[0, 0].get_ylim() == pytest.approx((0.5, 1.5))
        assert not any(label.get_visible() for label in hist_axes[0, 0].get_xticklabels())
        np.testing.assert_allclose(fig.get_size_inches(), [10, 10])
        fig.canvas.draw()
        assert all(
            hist_ax.get_position().y0 - ratio_ax.get_position().y1 < 0.02
            for hist_ax, ratio_ax in zip(hist_axes.flat, ratio_axes.flat)
        )
        assert _atlas_label_count(fig) == 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("count", [0, 3, 5])
@pytest.mark.parametrize("plotter", [plot_angular_1d_grid, plot_angular_2d_grid])
def test_angular_grids_require_exactly_four_observables(plotter, count):
    observables = (_observables() * 2)[:count]
    with pytest.raises(ValueError, match="exactly four"):
        plotter(observables, "invalid")


@pytest.mark.parametrize("plotter", [plot_angular_1d_grid, plot_angular_2d_grid])
def test_angular_grids_reject_mismatched_shapes(plotter):
    observables = _observables()
    observables[0]["truth"] = np.array([0.1, 0.2])
    figures_before = plt.get_fignums()

    with pytest.raises(ValueError, match="does not match truth shape"):
        plotter(observables, "invalid")
    assert plt.get_fignums() == figures_before


@pytest.mark.parametrize("plotter", [plot_angular_1d_grid, plot_angular_2d_grid])
def test_angular_grids_reject_observables_without_finite_pairs(plotter):
    observables = _observables()
    observables[0]["pred"] = np.array([np.nan, np.inf])
    observables[0]["truth"] = np.array([np.nan, -np.inf])
    figures_before = plt.get_fignums()

    with pytest.raises(ValueError, match="no finite prediction/truth pairs"):
        plotter(observables, "invalid")
    assert plt.get_fignums() == figures_before


@pytest.mark.parametrize("shared_colorbar, expected_axes", [(False, 8), (True, 5)])
def test_plot_angular_2d_grid_preserves_panel_and_colorbar_modes(
    shared_colorbar, expected_axes
):
    fig, axes = plot_angular_2d_grid(
        _observables(),
        "Angular correlations",
        shared_colorbar=shared_colorbar,
        share_axes=True,
    )

    try:
        assert axes.shape == (2, 2)
        assert len(fig.axes) == expected_axes
        assert isinstance(axes[0, 0].collections[0].norm, LogNorm)
        assert not isinstance(axes[0, 1].collections[0].norm, LogNorm)
        assert "RMSE = 0.52" in axes[0, 0].get_title()
        assert axes[0, 0].get_aspect() == pytest.approx(1.0)
        np.testing.assert_allclose(axes[0, 0].lines[0].get_xdata(), [0.0, 3.0])
        np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), [0.0, 3.0])
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 1])
        assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 1])
        assert _atlas_label_count(fig) == 0
    finally:
        plt.close(fig)


def test_visualize_notebook_compiles_and_uses_exported_angular_helpers():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec")

    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)

    defined_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert not {
        "_prepare_angular_data",
        "plot_angular_1d_grid",
        "plot_angular_2d_grid",
    } & defined_names

    plottingtool_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "notebooks.plottingtool"
        for alias in node.names
    }
    assert {"plot_angular_1d_grid", "plot_angular_2d_grid"} <= plottingtool_imports

    assignments = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {
        "angular_observables",
        "mixed_sum_observables",
        "mixed_diff_observables",
    } <= assignments

    angular_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"plot_angular_1d_grid", "plot_angular_2d_grid"}
    ]
    assert len(angular_calls) == 6


def test_truth_angle_feature_diagnostic_executes_with_periodic_inputs(capsys):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if "# Truth-angle conditioning feature diagnostic" in "".join(cell["source"])
    ]
    assert len(matching_cells) == 1

    event_count = 32
    phase = np.linspace(-0.25, 0.25, event_count)
    wrapped_phi = np.where(
        np.arange(event_count) % 2 == 0,
        -np.pi + phase,
        np.pi + phase,
    )
    lepton_pt = np.linspace(25.0, 80.0, event_count)
    train_features = np.zeros((event_count, 21), dtype=float)
    train_features[:, 0] = lepton_pt * np.cos(wrapped_phi)
    train_features[:, 1] = lepton_pt * np.sin(wrapped_phi)
    train_features[:, 2] = np.linspace(-30.0, 30.0, event_count)
    train_features[:, 3] = np.sqrt(
        train_features[:, 0] ** 2
        + train_features[:, 1] ** 2
        + train_features[:, 2] ** 2
    )
    train_features[:, 4] = -0.7 * train_features[:, 0]
    train_features[:, 5] = -0.7 * train_features[:, 1]
    train_features[:, 6] = np.linspace(20.0, -20.0, event_count)
    train_features[:, 7] = np.sqrt(
        train_features[:, 4] ** 2
        + train_features[:, 5] ** 2
        + train_features[:, 6] ** 2
    )
    train_features[:, 16] = 30.0 * np.cos(wrapped_phi + 0.4)
    train_features[:, 17] = 30.0 * np.sin(wrapped_phi + 0.4)
    train_features[0, 3] = 0.0
    train_features[0, 7] = 0.0

    true_ang = np.column_stack(
        (
            np.linspace(0.2, 2.8, event_count),
            wrapped_phi,
            np.linspace(2.8, 0.2, event_count),
            wrapped_phi + 0.2,
            np.linspace(0.4, 2.4, event_count),
            np.linspace(-1.0, 1.0, event_count),
            wrapped_phi - 0.3,
            wrapped_phi + 0.5,
        )
    )
    namespace = {
        "np": np,
        "plt": plt,
        "train_features": train_features,
        "angular_valid": np.ones(event_count, dtype=bool),
        "true_ang": true_ang,
    }

    exec("".join(matching_cells[0]["source"]), namespace)

    matrix = namespace["feature_target_association"]
    features = namespace["hl_angle_features"]
    assert matrix.shape == (len(features), 8)
    assert np.nanmax(np.abs(matrix)) <= 1.0
    assert len(namespace["feature_target_association_fig"].axes) == 1
    np.testing.assert_allclose(
        namespace["feature_target_association_fig"].get_size_inches(), [12.0, 12.0]
    )
    assert namespace["feature_target_association_fig"].axes[0].get_aspect() == pytest.approx(1.0)
    assert {"m_ll", "deta_ll", "dphi_ll", "dphi_ll_MET"} <= features.keys()
    assert np.isnan(features["m_ll"][0][0])
    lplus_phi_index = list(features).index("phi_lplus")
    assert matrix[lplus_phi_index, 1] > 0.95

    association = namespace["_association_strength"]
    two_direction_phi = np.tile([0.0, np.pi], 4)
    assert association(two_direction_phi, np.cos(two_direction_phi), True, False) > 0.95
    assert np.isnan(
        association(
            np.array([-3.1, -1.0, 1.0, 3.1]),
            np.array([-3.0, -0.8, 0.9, 3.0]),
            True,
            True,
        )
    )
    assert "Strongest reconstructed features by truth target" in capsys.readouterr().out
    plt.close(namespace["feature_target_association_fig"])
