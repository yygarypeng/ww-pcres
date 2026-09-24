import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from notebooks.plottingtool import (
    LOSS_COMPONENTS,
    _metric_series,
    _prepare_loss_plot_data,
    plot_gradient_cosine_heatmaps,
    plot_gradient_norms,
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
    weights = {"w_fourvec": 2.0, "dmet": 3.0}
    df = sparse_metrics(
        epochs=[0, 1, 2],
        component_values={
            "w_fourvec": ([3.0, 2.0, 1.0], [3.2, 1.8, 1.9]),
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
    assert data["raw"]["train"]["w_fourvec"].index.tolist() == [0, 1, 2]
    pd.testing.assert_series_equal(
        data["contributions"]["train"]["w_fourvec"],
        pd.Series([6.0, 4.0, 2.0], index=pd.Index([0, 1, 2], name="epoch")),
    )
    assert data["mismatches"] == []


def test_partial_loss_weights_inherit_model_defaults():
    expected_weights = {
        "w_fourvec": 1.0,
        "higgs_mass": 0.0,
        "alpha_mmd": 0.0,
        "w_mass_mmd": 0.0,
        "w_mass": 0.0,
        "angular_mmd": 0.0,
        "dmet": 2.0,
    }
    df = sparse_metrics(
        epochs=[0],
        component_values={name: ([1.0], [1.0]) for name in expected_weights},
        weights=expected_weights,
    )
    cfg = {"parameters": {"loss_weights": {"dmet": 2.0}}}

    data = _prepare_loss_plot_data(df, cfg)

    assert {name: series.iloc[0] for name, series in data["weights"].items()} == expected_weights
    assert data["mismatches"] == []


def test_logged_weights_override_static_config_and_use_last_duplicate():
    df = sparse_metrics(
        epochs=[0, 1, 2],
        component_values={
            "w_fourvec": ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]),
            "dmet": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        },
        weights={"w_fourvec": 1.5, "dmet": 3.0},
    )
    weight_rows = pd.DataFrame(
        [
            {"epoch": 1, "loss_weight/w_fourvec": 1.5},
            {"epoch": 1, "loss_weight/w_fourvec": 2.0},
            {"epoch": 2, "loss_weight/w_fourvec": 4.0},
        ]
    )
    df = pd.concat([df, weight_rows], ignore_index=True)
    cfg = {
        "parameters": {
            "loss_weights": {"w_fourvec": 1.5, "dmet": 3.0},
            "adaptive_loss_weights": False,
        }
    }

    data = _prepare_loss_plot_data(df, cfg)

    pd.testing.assert_series_equal(
        data["weights"]["w_fourvec"],
        pd.Series([1.5, 2.0, 4.0], index=pd.Index([0, 1, 2], name="epoch"), name="w_fourvec"),
    )
    pd.testing.assert_series_equal(
        data["weights"]["dmet"],
        pd.Series([3.0, 3.0, 3.0], index=pd.Index([0, 1, 2], name="epoch"), name="dmet"),
    )


def test_sparse_logged_weights_seed_initial_value_then_forward_fill():
    df = sparse_metrics(
        epochs=[0, 1, 2, 3],
        component_values={"w_fourvec": ([1.0] * 4, [1.0] * 4)},
        weights={"w_fourvec": 2.0},
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {"epoch": 1, "loss_weight/w_fourvec": 4.0},
                    {"epoch": 3, "loss_weight/w_fourvec": 8.0},
                ]
            ),
        ],
        ignore_index=True,
    )

    data = _prepare_loss_plot_data(
        df,
        {"parameters": {"loss_weights": {"w_fourvec": 2.0}}},
    )

    pd.testing.assert_series_equal(
        data["weights"]["w_fourvec"],
        pd.Series(
            [2.0, 4.0, 4.0, 8.0],
            index=pd.Index([0, 1, 2, 3], name="epoch"),
            name="w_fourvec",
        ),
    )


def test_component_only_epoch_gets_forward_filled_effective_weight():
    df = pd.DataFrame(
        [
            {"epoch": 0, "loss": 5.0, "val_loss": 6.0},
            {"epoch": 1, "loss": 4.0, "val_loss": 3.0, "loss_weight/w_fourvec": 2.0},
            {"epoch": 2, "w_fourvec_loss": 4.0, "val_w_fourvec_loss": 5.0},
        ]
    )

    data = _prepare_loss_plot_data(
        df,
        {"parameters": {"loss_weights": {"w_fourvec": 1.0}}},
    )

    pd.testing.assert_series_equal(
        data["weights"]["w_fourvec"],
        pd.Series(
            [1.0, 2.0, 2.0],
            index=pd.Index([0, 1, 2], name="epoch"),
            name="w_fourvec",
        ),
    )
    assert data["contributions"]["train"]["w_fourvec"].loc[2] == 8.0
    assert data["contributions"]["val"]["w_fourvec"].loc[2] == 10.0


def test_metric_series_keeps_last_value_for_duplicate_epoch():
    df = pd.DataFrame(
        {
            "epoch": [1, 0, 1],
            "w_fourvec_loss": [10.0, 2.0, 3.0],
        }
    )

    result = _metric_series(df, "w_fourvec_loss")

    pd.testing.assert_series_equal(
        result,
        pd.Series([2.0, 3.0], index=pd.Index([0, 1], name="epoch"), name="w_fourvec_loss"),
    )


def test_plot_loss_curves_builds_two_slide_subplot_figures(tmp_path, capsys):
    component_names = [name for name, _ in LOSS_COMPONENTS]
    assert "higgs_fourvec" in component_names
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
    assert [len(figure.axes) for figure in diagnostics["figures"]] == [12, 9]
    assert "summary" not in diagnostics
    assert "contribution_shares" not in diagnostics

    raw_figure, weighted_figure = diagnostics["figures"]
    assert (
        raw_figure.axes[0].get_subplotspec().get_gridspec().nrows,
        raw_figure.axes[0].get_subplotspec().get_gridspec().ncols,
    ) == (4, 3)
    assert (
        weighted_figure.axes[0].get_subplotspec().get_gridspec().nrows,
        weighted_figure.axes[0].get_subplotspec().get_gridspec().ncols,
    ) == (3, 3)

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
    np.testing.assert_array_equal(total_lines["total:train"].get_ydata(), [330.0, 285.0])
    np.testing.assert_array_equal(total_lines["total:val"].get_ydata(), [353.5, 308.5])
    assert total_lines["total:train"].get_color() == "tab:blue"
    assert total_lines["total:val"].get_color() == "tab:orange"
    np.testing.assert_array_equal(total_lines["best_epoch"].get_xdata(), [1, 1])
    assert all(not axis.axison for axis in raw_figure.axes[len(component_names) + 1 :])
    assert all(axis.axison for axis in weighted_figure.axes)
    assert len(raw_figure.legends) == 0
    assert len(weighted_figure.legends) == 0
    assert raw_figure.axes[2].get_legend() is not None
    assert weighted_figure.axes[2].get_legend() is not None
    for figure, legend_index in (
        (raw_figure, 2),
        (weighted_figure, 2),
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
    # The panels above the blank cells keep their epoch tick labels.
    for axis in raw_figure.axes[7:10]:
        assert all(tick.label1.get_visible() for tick in axis.xaxis.get_major_ticks())
    assert not any(tick.label1.get_visible() for tick in raw_figure.axes[4].xaxis.get_major_ticks())
    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())
    assert "component train val weighted_train weighted_val" in normalized_output
    assert r"$\Delta \mathrm{MET}$ 9 9.5 81 85.5" in normalized_output
    assert "could not be reconstructed exactly for: validation" in output
    plt.close("all")


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
    pd.DataFrame({"epoch": [0, 1], "loss": [2.0, 1.0], "val_loss": [3.0, 2.0]}).to_csv(
        metrics_path, index=False
    )

    diagnostics = plot_loss_curves(metrics_path, {"parameters": {}})

    raw_figure, weighted_figure = diagnostics["figures"]
    assert [len(figure.axes) for figure in diagnostics["figures"]] == [12, 9]
    assert all(not axis.axison for axis in raw_figure.axes[:9])
    assert raw_figure.axes[9].axison
    assert all(not axis.axison for axis in raw_figure.axes[10:])
    assert all(not axis.axison for axis in weighted_figure.axes)
    assert "summary" not in diagnostics
    assert "contribution_shares" not in diagnostics
    assert len(diagnostics["unavailable"]) == len(LOSS_COMPONENTS)
    assert "Unavailable train/validation pairs" in capsys.readouterr().out
    plt.close("all")


def test_plot_gradient_cosine_heatmaps_reports_missing_columns(capsys):
    df = pd.DataFrame({"epoch": [0], "loss": [1.0]})

    figures = plot_gradient_cosine_heatmaps(df)

    assert figures == {}
    assert "No grad_cos columns found" in capsys.readouterr().out


def test_plot_gradient_cosine_heatmaps_returns_each_generated_figure():
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "grad_cos/a__b": [0.1, 0.2],
            "grad_cos/a__total": [0.3, 0.4],
            "grad_cos/a__rest": [0.5, 0.6],
        }
    )

    figures = plot_gradient_cosine_heatmaps(df)

    assert set(figures) == {
        "gradient_cosine_pairwise_history",
        "gradient_cosine_pairwise_latest",
        "gradient_cosine_total_history",
        "gradient_cosine_rest_history",
    }
    assert all(isinstance(figure, plt.Figure) for figure in figures.values())
    plt.close("all")


def test_plot_gradient_norms_reports_missing_columns(capsys):
    df = pd.DataFrame({"epoch": [0], "loss": [1.0]})

    assert plot_gradient_norms(df) is None
    assert "No populated grad_norm columns found" in capsys.readouterr().out


def test_plot_gradient_norms_shows_magnitude_and_share(capsys):
    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "grad_norm/angular_mmd": [0.03, 0.02, 0.01],
            "grad_norm/higgs_mass": [0.01, 0.02, 0.03],
        }
    )

    fig = plot_gradient_norms(df)

    magnitude, budget = fig.axes
    assert magnitude.get_yscale() == "log"
    assert magnitude.get_ylabel() == r"$\|w\,\nabla_{\theta} L\|$"
    assert budget.get_ylim() == (0.0, 1.0)

    lines = {line.get_label(): line for line in magnitude.get_lines()}
    assert set(lines) == {"angular_mmd", "higgs_mass"}
    np.testing.assert_allclose(lines["higgs_mass"].get_ydata(), [0.01, 0.02, 0.03])

    # The same term keeps its colour in both panels, and the shares fill the axis.
    stack_colors = [tuple(patch.get_facecolor()[0][:3]) for patch in budget.collections]
    line_colors = [tuple(lines[name].get_color()[:3]) for name in ("angular_mmd", "higgs_mass")]
    assert stack_colors == line_colors

    out = capsys.readouterr().out
    assert "Gradient budget at epoch 2" in out
    assert "75.0%" in out  # higgs_mass 0.03 of 0.04 at the last epoch
    plt.close("all")


def test_plot_gradient_norms_uses_one_readable_shared_legend():
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "grad_norm/angular_mmd": [0.03, 0.02],
            "grad_norm/higgs_mass": [0.01, 0.02],
        }
    )

    fig = plot_gradient_norms(df)

    magnitude, budget = fig.axes
    assert magnitude.get_legend() is None
    assert budget.get_legend() is None
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert {text.get_text() for text in legend.get_texts()} == {"angular_mmd", "higgs_mass"}
    assert all(text.get_fontsize() >= 11 for text in legend.get_texts())

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert legend_box.y1 < min(axis.get_window_extent(renderer).y0 for axis in fig.axes)
    assert len({round(text.get_window_extent(renderer).x0) for text in legend.get_texts()}) > 1
    assert budget.yaxis.get_major_formatter()(0.5) == "50%"
    plt.close("all")


def test_plot_gradient_norms_survives_a_single_logged_epoch(recwarn):
    df = pd.DataFrame({"epoch": [0], "grad_norm/higgs_mass": [1.0], "grad_norm/w_mass": [3.0]})

    fig = plot_gradient_norms(df)

    assert fig is not None
    assert not [w for w in recwarn if "identical low and high xlims" in str(w.message)]
    plt.close("all")
