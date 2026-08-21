import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from notebooks import plottingtool
from notebooks.plottingtool import (
    LOSS_COMPONENTS,
    _metric_series,
    _prepare_loss_plot_data,
    plot_gradient_cosine_heatmaps,
    plot_loss_curves,
)


def sparse_metrics(epochs, component_values, weights):
    rows = []
    for epoch in epochs:
        train = {"epoch": epoch}
        val = {"epoch": epoch}
        train_total = 0.0
        val_total = 0.0
        for name, (train_values, val_values) in component_values.items():
            train[f"{name}_loss"] = train_values[epoch]
            val[f"val_{name}_loss"] = val_values[epoch]
            train_total += weights[name] * train_values[epoch]
            val_total += weights[name] * val_values[epoch]
        train["loss"] = train_total
        val["val_loss"] = val_total
        rows.extend([train, val])
    return pd.DataFrame(rows)


def test_static_data_keeps_epoch_zero_and_reconstructs_totals():
    weights = {"huber": 2.0, "dmet": 3.0}
    df = sparse_metrics(
        epochs=[0, 1, 2],
        component_values={
            "huber": ([3.0, 2.0, 1.0], [3.2, 1.8, 1.9]),
            "dmet": ([2.0, 1.5, 1.0], [2.1, 1.4, 1.6]),
        },
        weights=weights,
    )
    cfg = {
        "parameters": {
            "loss_weights": weights,
            "adaptive_loss_weights": False,
        }
    }

    data = _prepare_loss_plot_data(df, cfg)

    assert data["best_epoch"] == 1
    assert data["final_epoch"] == 2
    assert data["raw"]["train"]["huber"].index.tolist() == [0, 1, 2]
    pd.testing.assert_series_equal(
        data["contributions"]["train"]["huber"],
        pd.Series([6.0, 4.0, 2.0], index=pd.Index([0, 1, 2], name="epoch")),
    )
    assert data["mismatches"] == []


def test_partial_loss_weights_inherit_model_defaults():
    expected_weights = {
        "huber": 1.0,
        "higgs_mass": 0.0,
        "alpha_mmd": 0.0,
        "mass_mmd": 0.0,
        "w_mass_huber": 0.0,
        "angular_mmd": 0.0,
        "dmet": 2.0,
    }
    df = sparse_metrics(
        epochs=[0],
        component_values={
            name: ([1.0], [1.0])
            for name in expected_weights
        },
        weights=expected_weights,
    )
    cfg = {"parameters": {"loss_weights": {"dmet": 2.0}}}

    data = _prepare_loss_plot_data(df, cfg)

    assert {
        name: series.iloc[0]
        for name, series in data["weights"].items()
    } == expected_weights
    assert data["mismatches"] == []


def test_logged_weights_override_static_config_and_use_last_duplicate():
    df = sparse_metrics(
        epochs=[0, 1, 2],
        component_values={
            "huber": ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]),
            "dmet": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        },
        weights={"huber": 1.5, "dmet": 3.0},
    )
    weight_rows = pd.DataFrame(
        [
            {"epoch": 1, "loss_weight/huber": 1.5},
            {"epoch": 1, "loss_weight/huber": 2.0},
            {"epoch": 2, "loss_weight/huber": 4.0},
        ]
    )
    df = pd.concat([df, weight_rows], ignore_index=True)
    cfg = {
        "parameters": {
            "loss_weights": {"huber": 1.5, "dmet": 3.0},
            "adaptive_loss_weights": False,
        }
    }

    data = _prepare_loss_plot_data(df, cfg)

    pd.testing.assert_series_equal(
        data["weights"]["huber"],
        pd.Series([1.5, 2.0, 4.0], index=pd.Index([0, 1, 2], name="epoch"), name="huber"),
    )
    pd.testing.assert_series_equal(
        data["weights"]["dmet"],
        pd.Series([3.0, 3.0, 3.0], index=pd.Index([0, 1, 2], name="epoch"), name="dmet"),
    )


def test_sparse_logged_weights_seed_initial_value_then_forward_fill():
    df = sparse_metrics(
        epochs=[0, 1, 2, 3],
        component_values={"huber": ([1.0] * 4, [1.0] * 4)},
        weights={"huber": 2.0},
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {"epoch": 1, "loss_weight/huber": 4.0},
                    {"epoch": 3, "loss_weight/huber": 8.0},
                ]
            ),
        ],
        ignore_index=True,
    )

    data = _prepare_loss_plot_data(
        df,
        {"parameters": {"loss_weights": {"huber": 2.0}}},
    )

    pd.testing.assert_series_equal(
        data["weights"]["huber"],
        pd.Series(
            [2.0, 4.0, 4.0, 8.0],
            index=pd.Index([0, 1, 2, 3], name="epoch"),
            name="huber",
        ),
    )


def test_component_only_epoch_gets_forward_filled_effective_weight():
    df = pd.DataFrame(
        [
            {"epoch": 0, "loss": 5.0, "val_loss": 6.0},
            {"epoch": 1, "loss": 4.0, "val_loss": 3.0, "loss_weight/huber": 2.0},
            {"epoch": 2, "huber_loss": 4.0, "val_huber_loss": 5.0},
        ]
    )

    data = _prepare_loss_plot_data(
        df,
        {"parameters": {"loss_weights": {"huber": 1.0}}},
    )

    pd.testing.assert_series_equal(
        data["weights"]["huber"],
        pd.Series(
            [1.0, 2.0, 2.0],
            index=pd.Index([0, 1, 2], name="epoch"),
            name="huber",
        ),
    )
    assert data["contributions"]["train"]["huber"].loc[2] == 8.0
    assert data["contributions"]["val"]["huber"].loc[2] == 10.0


def test_metric_series_keeps_last_value_for_duplicate_epoch():
    df = pd.DataFrame(
        {
            "epoch": [1, 0, 1],
            "huber_loss": [10.0, 2.0, 3.0],
        }
    )

    result = _metric_series(df, "huber_loss")

    pd.testing.assert_series_equal(
        result,
        pd.Series([2.0, 3.0], index=pd.Index([0, 1], name="epoch"), name="huber_loss"),
    )


def test_plot_loss_curves_builds_two_slide_subplot_figures(tmp_path, capsys):
    component_names = [name for name, _ in LOSS_COMPONENTS]
    weights = {name: index + 1.0 for index, name in enumerate(component_names)}
    df = sparse_metrics(
        epochs=[0, 1],
        component_values={
            name: (
                [index + 2.0, index + 1.0],
                [index + 2.5, index + 1.5],
            )
            for index, name in enumerate(component_names)
        },
        weights=weights,
    )
    df.loc[df["val_loss"].notna(), "val_loss"] += 1.0
    metrics_path = tmp_path / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    cfg = {"parameters": {"loss_weights": weights, "adaptive_loss_weights": False}}

    diagnostics = plot_loss_curves(metrics_path, cfg)

    assert diagnostics["mismatches"] == ["val"]
    assert [len(figure.axes) for figure in diagnostics["figures"]] == [8, 8]
    assert "summary" not in diagnostics
    assert "contribution_shares" not in diagnostics

    raw_figure, weighted_figure = diagnostics["figures"]
    assert (raw_figure.axes[0].get_subplotspec().get_gridspec().nrows,
            raw_figure.axes[0].get_subplotspec().get_gridspec().ncols) == (2, 4)
    assert (weighted_figure.axes[0].get_subplotspec().get_gridspec().nrows,
            weighted_figure.axes[0].get_subplotspec().get_gridspec().ncols) == (2, 4)

    for index, (name, label) in enumerate(LOSS_COMPONENTS):
        raw_axis = raw_figure.axes[index]
        weighted_axis = weighted_figure.axes[index]
        raw_lines = {line.get_label(): line for line in raw_axis.lines}
        weighted_lines = {line.get_label(): line for line in weighted_axis.lines}
        expected_train = [index + 2.0, index + 1.0]
        expected_val = [index + 2.5, index + 1.5]

        assert raw_axis.get_title(loc="left") == label
        assert weighted_axis.get_title(loc="left") == label
        np.testing.assert_array_equal(raw_lines[f"{name}:train"].get_ydata(), expected_train)
        np.testing.assert_array_equal(raw_lines[f"{name}:val"].get_ydata(), expected_val)
        np.testing.assert_array_equal(
            weighted_lines[f"{name}:train"].get_ydata(),
            np.asarray(expected_train) * weights[name],
        )
        np.testing.assert_array_equal(
            weighted_lines[f"{name}:val"].get_ydata(),
            np.asarray(expected_val) * weights[name],
        )
        assert raw_lines[f"{name}:train"].get_color() == "tab:blue"
        assert weighted_lines[f"{name}:train"].get_color() == "tab:blue"
        assert raw_lines[f"{name}:val"].get_color() == "tab:orange"
        assert weighted_lines[f"{name}:val"].get_color() == "tab:orange"
        assert raw_lines[f"{name}:train"].get_linestyle() == "-"
        assert raw_lines[f"{name}:val"].get_linestyle() == "--"
        np.testing.assert_array_equal(raw_lines["best_epoch"].get_xdata(), [1, 1])
        np.testing.assert_array_equal(weighted_lines["best_epoch"].get_xdata(), [1, 1])
        assert raw_axis.get_yscale() == "linear"
        assert weighted_axis.get_yscale() == "linear"

    total_axis = raw_figure.axes[len(component_names)]
    total_lines = {line.get_label(): line for line in total_axis.lines}
    assert total_axis.get_title(loc="left") == "Logged Total"
    np.testing.assert_array_equal(total_lines["total:train"].get_ydata(), [168.0, 140.0])
    np.testing.assert_array_equal(total_lines["total:val"].get_ydata(), [183.0, 155.0])
    assert total_lines["total:train"].get_color() == "tab:blue"
    assert total_lines["total:val"].get_color() == "tab:orange"
    np.testing.assert_array_equal(total_lines["best_epoch"].get_xdata(), [1, 1])
    assert raw_figure.axes[7].axison
    assert not weighted_figure.axes[7].axison
    assert len(raw_figure.legends) == 0
    assert len(weighted_figure.legends) == 0
    assert raw_figure.axes[3].get_legend() is not None
    assert weighted_figure.axes[3].get_legend() is not None
    for figure, legend_index in (
        (raw_figure, 3),
        (weighted_figure, 3),
    ):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title_bounds = figure._suptitle.get_window_extent(renderer)
        legend_bounds = figure.axes[legend_index].get_legend().get_window_extent(renderer)
        assert not title_bounds.overlaps(legend_bounds)
        for axis in figure.axes:
            if not axis.axison:
                continue
            component_title_bounds = axis.title.get_window_extent(renderer)
            assert figure.bbox.contains(component_title_bounds.x0, component_title_bounds.y0)
            assert figure.bbox.contains(component_title_bounds.x1, component_title_bounds.y1)
    output = capsys.readouterr().out
    assert (
        r"  $\Delta \mathrm{MET}$: train=7 val=7.5 "
        "weighted_train=49 weighted_val=52.5"
    ) in output
    assert "could not be reconstructed exactly for: validation" in output
    plt.close("all")


def test_removed_loss_dashboard_helpers_are_absent():
    assert not hasattr(plottingtool, "_contribution_shares")
    assert not hasattr(plottingtool, "_loss_summary")
    assert not hasattr(plottingtool, "Normalize")


def test_plot_loss_curves_reports_missing_csv(tmp_path, capsys):
    result = plot_loss_curves(tmp_path / "missing.csv", {"parameters": {}})

    assert result is None
    assert "No metrics.csv found" in capsys.readouterr().out


def test_plot_loss_curves_reports_empty_csv(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("")

    result = plot_loss_curves(metrics_path, {"parameters": {}})

    assert result is None
    assert "Cannot plot losses" in capsys.readouterr().out


def test_plot_loss_curves_reports_when_all_components_are_unavailable(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {"epoch": [0, 1], "loss": [2.0, 1.0], "val_loss": [3.0, 2.0]}
    ).to_csv(metrics_path, index=False)

    diagnostics = plot_loss_curves(metrics_path, {"parameters": {}})

    raw_figure, weighted_figure = diagnostics["figures"]
    assert [len(figure.axes) for figure in diagnostics["figures"]] == [8, 8]
    assert all(not axis.axison for axis in raw_figure.axes[:7])
    assert raw_figure.axes[7].axison
    assert all(not axis.axison for axis in weighted_figure.axes)
    assert "summary" not in diagnostics
    assert "contribution_shares" not in diagnostics
    assert len(diagnostics["unavailable"]) == len(LOSS_COMPONENTS)
    assert "Unavailable train/validation pairs" in capsys.readouterr().out
    plt.close("all")


def test_plot_gradient_cosine_heatmaps_reports_missing_columns(capsys):
    df = pd.DataFrame({"epoch": [0], "loss": [1.0]})

    plot_gradient_cosine_heatmaps(df)

    assert "No grad_cos columns found" in capsys.readouterr().out
