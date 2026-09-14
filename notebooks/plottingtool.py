import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wasserstein_distance

hep.style.use(hep.style.ATLAS)
ATLAS_LABEL_TEXT = "Internal Simulation"


def _rel_err_func(a, b):
    if np.isnan(a).any() or np.isnan(b).any():
        print("Warning: NaN values detected in input arrays.")
    if np.any(b == 0):
        print("Warning: Zero values detected in denominator array.")
    mask = ~np.isnan(a) & ~np.isnan(b) & (b != 0)
    return (a[mask] - b[mask]) / b[mask]


def _rmse(pred, truth):
    mask = np.isfinite(pred) & np.isfinite(truth)
    if np.sum(mask) != len(pred):
        print(f"Warning: {len(pred) - np.sum(mask)} invalid entries found")
    pred = pred[mask]
    truth = truth[mask]
    return np.sqrt(np.mean((pred - truth) ** 2))


def _fmt_metric(value, threshold=1e-2):
    """Fixed-point for typical magnitudes, scientific notation once that would round to 0.00."""
    if value == 0 or abs(value) >= threshold:
        return f"{value:.2f}"
    return f"{value:.2e}"


def _emd_title(label, pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    pred = pred[np.isfinite(pred)]
    truth = truth[np.isfinite(truth)]
    if pred.size == 0 or truth.size == 0:
        return f"{label}  EMD = n/a"
    return f"{label}  EMD = {_fmt_metric(wasserstein_distance(pred, truth))}"


def _hist2d_kwargs(bins_edges, vmax, log):
    kwargs = {"bins": [bins_edges, bins_edges], "cmap": "viridis"}
    if log:
        kwargs["norm"] = LogNorm(vmin=1, vmax=vmax)
    else:
        kwargs.update(vmin=1, vmax=vmax)
    return kwargs


def _apply_atlas_label(ax, color, loc=2, text=ATLAS_LABEL_TEXT):
    txt = hep.atlas.label(text, data=True, loc=loc, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    return txt


def _set_square_ticks(ax, bins_edges, n_ticks=5, pad=10):
    ax.tick_params(axis="both", which="major", pad=pad)
    ticks = np.linspace(bins_edges[0], bins_edges[-1], n_ticks)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_aspect("equal", adjustable="box")


def _plot_hist_ratio(
    ax,
    pred_counts,
    truth_counts,
    bin_edges,
    *,
    ratio_ylim=(0.5, 1.5),
    ratio_text="Pred/True",
    show_ylabel=True,
):
    pred_counts = np.asarray(pred_counts, dtype=float)
    truth_counts = np.asarray(truth_counts, dtype=float)
    bin_edges = np.asarray(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    valid = (pred_counts > 0) & (truth_counts > 0)
    ratio = np.full_like(pred_counts, np.nan, dtype=float)
    ratio_err = np.full_like(pred_counts, np.nan, dtype=float)
    ratio[valid] = pred_counts[valid] / truth_counts[valid]
    ratio_err[valid] = ratio[valid] * np.sqrt(1.0 / pred_counts[valid] + 1.0 / truth_counts[valid])

    truth_rel_err = np.full_like(truth_counts, np.nan, dtype=float)
    truth_mask = truth_counts > 0
    truth_rel_err[truth_mask] = 1.0 / np.sqrt(truth_counts[truth_mask])
    band_low = 1.0 - truth_rel_err
    band_high = 1.0 + truth_rel_err
    ax.fill_between(
        np.repeat(bin_edges, 2)[1:-1],
        np.repeat(band_low, 2),
        np.repeat(band_high, 2),
        color="gray",
        alpha=0.25,
        linewidth=0,
    )

    y_min, y_max = ratio_ylim
    span = y_max - y_min
    in_view = valid & (ratio >= y_min) & (ratio <= y_max)
    overflow_high = valid & (ratio > y_max)
    overflow_low = valid & (ratio < y_min)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.3)
    errorbar = ax.errorbar(
        bin_centers[in_view],
        ratio[in_view],
        yerr=ratio_err[in_view],
        fmt="o",
        color="black",
        markersize=3,
        linewidth=1,
        capsize=0,
        label=ratio_text,
    )
    errorbar.lines[0].set_label(ratio_text)

    y_top_from = y_max - 0.15 * span
    y_bot_from = y_min + 0.15 * span
    for x in bin_centers[overflow_high]:
        ax.annotate(
            "",
            xy=(x, y_max),
            xytext=(x, y_top_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )
    for x in bin_centers[overflow_low]:
        ax.annotate(
            "",
            xy=(x, y_min),
            xytext=(x, y_bot_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )

    ax.set_ylabel(ratio_text if show_ylabel else "")
    ax.set_ylim(*ratio_ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", which="major", pad=10)


def _prepare_angular_data(observable):
    pred = np.asarray(observable["pred"]) / np.pi
    truth = np.asarray(observable["truth"]) / np.pi
    if pred.shape != truth.shape:
        raise ValueError(
            f"{observable['label']}: prediction shape {pred.shape} "
            f"does not match truth shape {truth.shape}"
        )

    finite = np.isfinite(pred) & np.isfinite(truth)
    if not finite.any():
        raise ValueError(f"{observable['label']}: no finite prediction/truth pairs")
    return pred[finite], truth[finite]


def _prepare_angular_observables(observables):
    if len(observables) != 4:
        raise ValueError("angular grid plotting requires exactly four observables")
    return [_prepare_angular_data(observable) for observable in observables]


def plot_angular_1d_grid(
    observables, title, share_axes=False, pred_label="Pred", truth_label="True"
):
    prepared_data = _prepare_angular_observables(observables)
    fig = plt.figure(figsize=(10, 10))
    outer_grid = fig.add_gridspec(
        2,
        2,
        left=0.1,
        right=0.96,
        bottom=0.08,
        top=0.86,
        wspace=0.22,
        hspace=0.32,
    )
    hist_axes = np.empty((2, 2), dtype=object)
    ratio_axes = np.empty((2, 2), dtype=object)

    first_hist_ax = None
    for index, (observable, (pred, truth)) in enumerate(zip(observables, prepared_data)):
        row, column = divmod(index, 2)
        panel_grid = outer_grid[row, column].subgridspec(2, 1, height_ratios=(3.5, 1), hspace=0.02)
        shared_hist_ax = first_hist_ax if share_axes else None
        ax = fig.add_subplot(panel_grid[0], sharex=shared_hist_ax, sharey=shared_hist_ax)
        if first_hist_ax is None:
            first_hist_ax = ax
        rax = fig.add_subplot(panel_grid[1], sharex=ax)
        hist_axes[row, column] = ax
        ratio_axes[row, column] = rax

        bins = observable["bins"]
        pred_counts, _ = np.histogram(pred, bins=bins)
        truth_counts, _ = np.histogram(truth, bins=bins)
        ax.hist(
            pred,
            bins=bins,
            linewidth=2,
            color="red",
            histtype="step",
            label=pred_label,
        )
        ax.hist(
            truth,
            bins=bins,
            linewidth=2,
            color="blue",
            histtype="step",
            label=truth_label,
        )
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(
            _emd_title(
                observable["label"],
                np.asarray(observable["pred"]) / np.pi,
                np.asarray(observable["truth"]) / np.pi,
            ),
            fontsize=14,
            pad=7,
        )
        ax.set_ylabel("Events" if column == 0 else "", fontsize=12)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if share_axes:
            ax.label_outer()

        _plot_hist_ratio(
            rax,
            pred_counts,
            truth_counts,
            bins,
            ratio_ylim=(0.5, 1.5),
            ratio_text=f"{pred_label}/{truth_label}",
            show_ylabel=column == 0,
        )
        if column == 0:
            rax.yaxis.label.set_fontsize(12)
        rax.tick_params(axis="both", labelsize=10)

    handles, labels = hist_axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        fontsize=12,
        ncols=2,
    )
    fig.supxlabel(r"Observable [rad/$\pi$]", fontsize=12)
    fig.suptitle(title, fontsize=17, fontweight="semibold")
    plt.show()
    return fig, (hist_axes, ratio_axes)


def plot_angular_2d_grid(
    observables,
    title,
    shared_colorbar=False,
    share_axes=False,
    pred_label="Pred",
    truth_label="True",
):
    prepared_data = _prepare_angular_observables(observables)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 7.6),
        sharex=share_axes,
        sharey=share_axes,
        constrained_layout=True,
    )
    shared_mappable = None
    for ax, observable, (pred, truth) in zip(axes.flat, observables, prepared_data):
        bins = observable["bins"]
        rmse = np.sqrt(np.mean((pred - truth) ** 2))
        image = ax.hist2d(
            pred, truth, **_hist2d_kwargs(bins, observable["vmax"], observable["log"])
        )[3]
        ax.plot(
            [bins[0], bins[-1]],
            [bins[0], bins[-1]],
            color="gainsboro",
            linestyle="--",
            linewidth=1.3,
        )
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bins[0], bins[-1])
        ax.set_title(f"{observable['label']}  RMSE = {_fmt_metric(rmse)}", fontsize=13, pad=7)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_aspect("equal", adjustable="box")
        if share_axes:
            ax.label_outer()

        if shared_colorbar:
            shared_mappable = image
        else:
            colorbar = fig.colorbar(image, ax=ax, label="Events", pad=0.02)
            colorbar.ax.tick_params(labelsize=10)
            colorbar.set_label("Events", fontsize=11)

    fig.supxlabel(rf"{pred_label} [rad/$\pi$]", fontsize=12)
    fig.supylabel(rf"{truth_label} [rad/$\pi$]", fontsize=12)
    fig.suptitle(title, fontsize=17, fontweight="semibold")
    if shared_colorbar:
        colorbar = fig.colorbar(
            shared_mappable,
            ax=axes.ravel().tolist(),
            label="Events",
            shrink=0.88,
            pad=0.02,
        )
        colorbar.ax.tick_params(labelsize=10)
        colorbar.set_label("Events", fontsize=11)
    plt.show()
    return fig, axes


def plot_1d_hist(
    pred,
    truth,
    name,
    bins_edges=np.linspace(-200, 200, 51),
    unit="GeV",
    color="black",
    savepath=None,
    ratio_ylim=(0.5, 1.5),
    ratio_text="Pred/True",
):
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.1},
    )

    # Top panel: truth and prediction histograms
    pred_counts, _ = np.histogram(pred, bins=bins_edges)
    truth_counts, _ = np.histogram(truth, bins=bins_edges)

    legend_text = ratio_text.split("/", 1) if "/" in ratio_text else [ratio_text, "True"]
    ax.hist(pred, bins=bins_edges, linewidth=2, color="red", histtype="step", label=legend_text[0])
    ax.hist(
        truth, bins=bins_edges, linewidth=2, color="blue", histtype="step", label=legend_text[1]
    )
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Events", loc="top")
    ax.set_title(_emd_title(name, pred, truth), loc="right")

    _apply_atlas_label(ax, color)
    ax.set_ylim(0, 1.15 * max(pred_counts.max(), truth_counts.max(), 1))
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="major", pad=12)

    _plot_hist_ratio(
        rax,
        pred_counts,
        truth_counts,
        bins_edges,
        ratio_ylim=ratio_ylim,
        ratio_text=ratio_text,
    )
    rax.set_xlabel(name + " [" + unit + "]", loc="right")

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_2d_hist(
    pred,
    truth,
    name,
    bins_edges=np.linspace(-200, 200, 51),
    log=False,
    unit="GeV",
    color="black",
    xlabel="Pred",
    ylabel="True",
    vmax=1e2,
    savepath=None,
):
    err = 0.2
    cor_mask = np.abs(_rel_err_func(pred, truth)) <= err  # set 20% relative error cut
    fig, ax = plt.subplots()
    ax.hist2d(pred, truth, **_hist2d_kwargs(bins_edges, vmax, log))

    ax.plot(bins_edges, bins_edges, color="gainsboro", linestyle="--")

    ax.set_xlabel(f"{xlabel} [{unit}]")
    ax.set_ylabel(f"{ylabel} [{unit}]")
    ax.set_title(f"{name}" + f" (RMSE: {_fmt_metric(_rmse(pred, truth))})", loc="right")
    print(f"Rel err < 20%: {100 * np.sum(cor_mask) / len(truth):.2f} %")

    _apply_atlas_label(ax, color)
    _set_square_ticks(ax, bins_edges)

    x_min, x_max = bins_edges[0], bins_edges[-1]
    y_min, y_max = bins_edges[0], bins_edges[-1]
    x_margin = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + x_margin)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_2d_res_hist(
    pred,
    truth,
    name_pos,
    name_neg,
    bins_edges=np.linspace(-200, 200, 51),
    log=False,
    unit="GeV",
    color="black",
    vmax=5e3,
    savepath=None,
):
    fig, ax = plt.subplots()
    ax.hist2d(pred, truth, **_hist2d_kwargs(bins_edges, vmax, log))
    ax.set_xlabel(rf"$\Delta_\text{{res}}${name_pos} [{unit}]")
    ax.set_ylabel(rf"$\Delta_\text{{res}}${name_neg} [{unit}]")
    _apply_atlas_label(ax, color, loc=0, text="   " + ATLAS_LABEL_TEXT)
    _set_square_ticks(ax, bins_edges)
    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


LOSS_COMPONENTS = [
    ("w_fourvec", r"$W$ 4-vec L1"),
    ("higgs_fourvec", r"$H$ 4-vec L1"),
    ("higgs_mass", r"$m_H$ L1"),
    ("alpha_mmd", r"$\alpha$ MMD"),
    ("w_mass_mmd", r"Joint $m_W$ MMD"),
    ("w_mass", r"$m_W$ L1"),
    ("angular_mmd", "Angular MMD"),
    ("dmet", r"$\Delta \mathrm{MET}$"),
]

DEFAULT_LOSS_WEIGHTS = {
    "w_fourvec": 1.0,
    "higgs_fourvec": 0.0,
    "higgs_mass": 0.0,
    "alpha_mmd": 0.0,
    "w_mass_mmd": 0.0,
    "w_mass": 0.0,
    "angular_mmd": 0.0,
    "dmet": 0.0,
}


def _metric_series(df, column):
    if "epoch" not in df.columns or column not in df.columns:
        return pd.Series(dtype=float, name=column)
    values = df.loc[df[column].notna() & df["epoch"].notna(), ["epoch", column]]
    values = values.drop_duplicates("epoch", keep="last").sort_values("epoch")
    return values.set_index("epoch")[column]


def _epoch_weights(df, name, epochs, initial_weight):
    result = pd.Series(float(initial_weight), index=epochs, dtype=float, name=name)
    logged = _metric_series(df, f"loss_weight/{name}")
    if logged.empty:
        return result
    combined_epochs = result.index.union(logged.index).sort_values()
    combined = logged.reindex(combined_epochs).ffill().fillna(float(initial_weight))
    return combined.reindex(epochs).rename(name)


def _prepare_loss_plot_data(df, cfg):
    if df.empty:
        raise ValueError("metrics.csv is empty")
    if "epoch" not in df.columns:
        raise ValueError("metrics.csv has no epoch column")

    totals = {
        "train": _metric_series(df, "loss"),
        "val": _metric_series(df, "val_loss"),
    }
    if totals["val"].empty:
        raise ValueError("metrics.csv has no populated val_loss")

    best_epoch = totals["val"].idxmin()
    common_total_epochs = totals["train"].index.intersection(totals["val"].index)
    final_epoch = common_total_epochs.max() if len(common_total_epochs) else None
    params = cfg.get("parameters", {})
    configured_weights = {**DEFAULT_LOSS_WEIGHTS, **params.get("loss_weights", {})}
    raw = {"train": {}, "val": {}}
    weights = {}
    contributions = {"train": {}, "val": {}}
    unavailable = []

    for name, label in LOSS_COMPONENTS:
        train = _metric_series(df, f"{name}_loss")
        val = _metric_series(df, f"val_{name}_loss")
        if train.empty or val.empty:
            unavailable.append(label)
            continue
        raw["train"][name] = train
        raw["val"][name] = val
        component_epochs = (
            totals["train"]
            .index.union(totals["val"].index)
            .union(train.index)
            .union(val.index)
            .sort_values()
        )
        weights[name] = _epoch_weights(
            df,
            name,
            component_epochs,
            configured_weights.get(name, 0.0),
        )
        contributions["train"][name] = train * weights[name].reindex(train.index)
        contributions["val"][name] = val * weights[name].reindex(val.index)

    mismatches = []
    for stage in ("train", "val"):
        if not contributions[stage]:
            continue
        reconstructed = pd.concat(contributions[stage], axis=1).sum(axis=1, min_count=1)
        comparison = pd.concat(
            [reconstructed.rename("reconstructed"), totals[stage]], axis=1
        ).dropna()
        if not comparison.empty and not np.allclose(
            comparison["reconstructed"], comparison[totals[stage].name], rtol=1e-5, atol=1e-8
        ):
            mismatches.append(stage)

    return {
        "totals": totals,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "raw": raw,
        "weights": weights,
        "contributions": contributions,
        "unavailable": unavailable,
        "mismatches": mismatches,
    }


def _value_at(series, epoch):
    if epoch is None or epoch not in series.index:
        return None
    return float(series.loc[epoch])


def _fmt_value(value):
    return "n/a" if value is None else f"{value:.6g}"


def _top_right_visible_axis(axes):
    visible = [axis for axis in axes if axis.axison]
    return (
        max(visible, key=lambda axis: (axis.get_position().x0, axis.get_position().y1))
        if visible
        else axes[0]
    )


def plot_loss_curves(metrics_path, cfg):
    print(f"Using metrics file: {metrics_path}")
    if not metrics_path.exists():
        print(f"No metrics.csv found at {metrics_path}")
        return None
    try:
        data = _prepare_loss_plot_data(pd.read_csv(metrics_path), cfg)
    except (ValueError, pd.errors.EmptyDataError) as error:
        print(f"Cannot plot losses: {error}")
        return None

    best_epoch = data["best_epoch"]
    rows = [
        {
            "component": "total",
            "train": _fmt_value(_value_at(data["totals"]["train"], best_epoch)),
            "val": _fmt_value(_value_at(data["totals"]["val"], best_epoch)),
            "weighted_train": "",
            "weighted_val": "",
        }
    ]
    for name, label in LOSS_COMPONENTS:
        if name not in data["raw"]["val"]:
            continue
        rows.append(
            {
                "component": label,
                "train": _fmt_value(_value_at(data["raw"]["train"][name], best_epoch)),
                "val": _fmt_value(_value_at(data["raw"]["val"][name], best_epoch)),
                "weighted_train": _fmt_value(
                    _value_at(data["contributions"]["train"][name], best_epoch)
                ),
                "weighted_val": _fmt_value(
                    _value_at(data["contributions"]["val"][name], best_epoch)
                ),
            }
        )
    print(f"Best loss values at best epoch {best_epoch} (min val_loss):")
    print(pd.DataFrame(rows).to_string(index=False))
    fig_raw, raw_axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, layout="constrained")
    fig_weighted, weighted_axes = plt.subplots(
        3, 3, figsize=(14, 11), sharex=True, layout="constrained"
    )
    raw_axes = raw_axes.ravel()
    weighted_axes = weighted_axes.ravel()
    stages = (
        ("train", "Training", "tab:blue", "-"),
        ("val", "Validation", "tab:orange", "--"),
    )

    visible_raw_axes = []
    visible_weighted_axes = []
    for index, (name, label) in enumerate(LOSS_COMPONENTS):
        raw_axis = raw_axes[index]
        weighted_axis = weighted_axes[index]
        if name not in data["raw"]["train"]:
            raw_axis.set_axis_off()
            weighted_axis.set_axis_off()
            continue
        for stage, _, color, linestyle in stages:
            raw_series = data["raw"][stage][name]
            raw_axis.plot(
                raw_series.index,
                raw_series.values,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                label=f"{name}:{stage}",
            )
            contribution = data["contributions"][stage][name]
            weighted_axis.plot(
                contribution.index,
                contribution.values,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                label=f"{name}:{stage}",
            )
        raw_axis.set_title(label, loc="left")
        weighted_axis.set_title(label, loc="left")
        visible_raw_axes.append(raw_axis)
        visible_weighted_axes.append(weighted_axis)

    total_axis = raw_axes[len(LOSS_COMPONENTS)]
    for stage, _, color, linestyle in stages:
        total = data["totals"][stage]
        total_axis.plot(
            total.index,
            total.values,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            label=f"total:{stage}",
        )
    total_axis.set_title("Logged Total", loc="left")
    visible_raw_axes.append(total_axis)
    for axis in weighted_axes[len(LOSS_COMPONENTS) :]:
        axis.set_axis_off()

    stage_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.5, label=label)
        for _, label, color, linestyle in stages
    ]
    stage_handles.append(
        Line2D(
            [0], [0], color="0.35", linestyle=":", linewidth=1, label=f"Best epoch: {best_epoch:g}"
        )
    )
    raw_legend_axis = _top_right_visible_axis(raw_axes)
    weighted_legend_axis = _top_right_visible_axis(weighted_axes)

    for figure, axes, title, ylabel, legend_axis in (
        (fig_raw, visible_raw_axes, "Raw loss histories", "Loss", raw_legend_axis),
        (
            fig_weighted,
            visible_weighted_axes,
            "Weighted loss contributions",
            "Weighted loss",
            weighted_legend_axis,
        ),
    ):
        for axis in axes:
            axis.axvline(
                best_epoch,
                color="0.35",
                linestyle=":",
                linewidth=1,
                label="best_epoch",
            )
            axis.grid(axis="y", color="0.9", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
        figure.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
        figure.suptitle(title, fontsize=16, fontweight="semibold", y=0.99)
        figure.supxlabel("Epoch")
        figure.supylabel(ylabel)
        legend_axis.legend(
            handles=stage_handles,
            frameon=False,
            loc="upper right",
            fontsize=9,
        )

    if data["unavailable"]:
        print("Unavailable train/validation pairs: " + ", ".join(data["unavailable"]))
    if data["mismatches"]:
        display_stages = ["validation" if stage == "val" else stage for stage in data["mismatches"]]
        print(
            "Warning: weighted contributions could not be reconstructed exactly for: "
            + ", ".join(display_stages)
            + ". Logger timing or aggregation may differ from epoch reconstruction."
        )

    plt.show()
    data["figures"] = [fig_raw, fig_weighted]
    return data


def _plot_grad_cos_heatmap(heatmap, x_col, ylabel, title):
    plt.figure(figsize=(16, max(8, 0.35 * len(heatmap))))
    plt.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(label="Gradient cosine")
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    tick_positions = np.linspace(
        0, len(heatmap.columns) - 1, min(10, len(heatmap.columns)), dtype=int
    )
    plt.xticks(tick_positions, heatmap.columns[tick_positions])
    plt.xlabel(x_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_gradient_cosine_heatmaps(df):
    grad_cols = sorted(col for col in df.columns if col.startswith("grad_cos/"))
    if not grad_cols:
        print(
            "No grad_cos columns found in metrics. Enable parameters.log_loss_gradient_cosines and rerun training."
        )
        return
    populated_counts = df[grad_cols].notna().sum()
    grad_cols = sorted(populated_counts[populated_counts > 0].index)
    if not grad_cols:
        print(
            f"Found {len(populated_counts)} grad_cos columns in metrics, but all values are empty."
        )
        print(
            "No gradient-cosine values were emitted. Older training code required adaptive_loss_weights: true even when log_loss_gradient_cosines was enabled."
        )
        print("Rerun with the current independent gradient-cosine logging implementation.")
        return

    total_grad_cols = [col for col in grad_cols if col.endswith("__total")]
    rest_grad_cols = [col for col in grad_cols if col.endswith("__rest")]
    pair_grad_cols = [
        col for col in grad_cols if col not in total_grad_cols and col not in rest_grad_cols
    ]
    df_grad = df[df[grad_cols].notna().any(axis=1)].copy()
    x_col = "epoch" if "epoch" in df_grad.columns else "step"

    latest = df_grad[grad_cols].ffill().iloc[-1]

    if pair_grad_cols:
        heatmap = df_grad.set_index(x_col)[pair_grad_cols].ffill().T
        heatmap.index = [
            col.replace("grad_cos/", "").replace("__", " vs ") for col in heatmap.index
        ]
        _plot_grad_cos_heatmap(heatmap, x_col, "Loss pair", "Pairwise Similarity Over Training")

        pair_latest = latest[pair_grad_cols].dropna()
        loss_names = sorted(
            {name for col in pair_grad_cols for name in col.replace("grad_cos/", "").split("__")}
        )
        matrix = pd.DataFrame(np.nan, index=loss_names, columns=loss_names)
        for name in loss_names:
            matrix.loc[name, name] = 1.0
        for col, value in pair_latest.items():
            name_a, name_b = col.replace("grad_cos/", "").split("__")
            matrix.loc[name_a, name_b] = value
            matrix.loc[name_b, name_a] = value

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Gradient cosine")
        plt.xticks(range(len(loss_names)), loss_names, rotation=45, ha="right")
        plt.yticks(range(len(loss_names)), loss_names)
        plt.title(f"Pairwise Similarity Matrix ({x_col} {df_grad[x_col].iloc[-1]})")
        plt.tight_layout()
        plt.show()
    else:
        print("No populated pairwise grad_cos metrics found.")

    if total_grad_cols:
        total_heatmap = df_grad.set_index(x_col)[total_grad_cols].ffill().T
        total_heatmap.index = [
            col.replace("grad_cos/", "").replace("__total", "") for col in total_heatmap.index
        ]
        total_heatmap = total_heatmap.dropna(how="all")

        if total_heatmap.empty:
            print("No populated loss-vs-total grad_cos metrics found.")
        else:
            _plot_grad_cos_heatmap(
                total_heatmap, x_col, "Loss", "Loss-vs-Total Similarity Over Training"
            )
    else:
        print("No populated loss-vs-total grad_cos metrics found.")

    if rest_grad_cols:
        rest_heatmap = df_grad.set_index(x_col)[rest_grad_cols].ffill().T
        rest_heatmap.index = [
            col.replace("grad_cos/", "").replace("__rest", "") for col in rest_heatmap.index
        ]
        rest_heatmap = rest_heatmap.dropna(how="all")

        if rest_heatmap.empty:
            print("No populated loss-vs-rest grad_cos metrics found.")
        else:
            _plot_grad_cos_heatmap(
                rest_heatmap, x_col, "Loss", "Loss-vs-Rest Similarity Over Training"
            )
    else:
        print("No populated loss-vs-rest grad_cos metrics found.")

    print(
        f"Visualized {len(pair_grad_cols)} pairwise, {len(total_grad_cols)} total, "
        f"and {len(rest_grad_cols)} rest grad_cos metrics across {len(df_grad)} logged rows."
    )


def plot_gradient_norms(df):
    """Plot how hard each loss term pulls: absolute magnitude and share of the total.

    The cosine panels show gradient direction; these show magnitude. A term can
    hold a negligible share of the loss value and still dominate the update, so
    the share panel is the one that says who is actually steering training.
    """
    norm_cols = sorted(col for col in df.columns if col.startswith("grad_norm/"))
    if norm_cols:
        populated = df[norm_cols].notna().sum()
        norm_cols = sorted(populated[populated > 0].index)
    if not norm_cols:
        print(
            "No populated grad_norm columns found. Enable parameters.log_loss_gradient_cosines "
            "and rerun training with the current logging implementation."
        )
        return

    df_norm = df[df[norm_cols].notna().any(axis=1)]
    x_col = "epoch" if "epoch" in df_norm.columns else "step"
    norms = df_norm.set_index(x_col)[norm_cols].ffill()
    norms.columns = [col.replace("grad_norm/", "") for col in norms.columns]
    shares = norms.div(norms.sum(axis=1), axis=0)

    # One colour per term in both panels, paired with a dash pattern so the lines
    # stay separable without relying on hue.
    palette = colormaps["tab10"].colors
    dashes = ("-", "--", "-.", ":")
    styles = {
        name: (palette[index % len(palette)], dashes[index % len(dashes)])
        for index, name in enumerate(norms.columns)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for name in norms.columns:
        color, dash = styles[name]
        axes[0].plot(norms.index, norms[name], label=name, color=color, linestyle=dash)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(x_col.capitalize())
    axes[0].set_ylabel(r"$\|w\,\nabla_{\theta} L\|$")
    axes[0].set_title("Weighted gradient magnitude per loss term")
    axes[0].legend(fontsize=8)

    axes[1].stackplot(
        shares.index,
        *(shares[name] for name in shares.columns),
        labels=list(shares.columns),
        colors=[styles[name][0] for name in shares.columns],
    )
    if shares.index.min() < shares.index.max():
        axes[1].set_xlim(shares.index.min(), shares.index.max())
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel(x_col.capitalize())
    axes[1].set_ylabel("Share of total gradient")
    axes[1].set_title("Gradient budget: which term steers the update")
    # Outside the axes and top-down, so the legend matches the stacking order
    # instead of covering the bands it describes.
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles[::-1], labels[::-1], fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5)
    )
    plt.show()

    latest = shares.iloc[-1].sort_values(ascending=False)
    print(f"Gradient budget at {x_col} {shares.index[-1]}:")
    for name, share in latest.items():
        print(f"  {name:>15}: {100 * share:5.1f}%   |w*grad| = {norms.iloc[-1][name]:.6g}")
    return fig
