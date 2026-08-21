import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

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
        top=0.9,
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
        valid = truth_counts != 0
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

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
        ax.set_title(observable["label"], fontsize=14, pad=7)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if share_axes:
            ax.label_outer()

        rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.3)
        rax.plot(
            bin_centers[valid],
            pred_counts[valid] / truth_counts[valid],
            color="black",
            marker="o",
            linestyle="none",
            markersize=3,
            label=f"{pred_label}/{truth_label}",
        )
        rax.set_ylim(0.5, 1.5)
        rax.tick_params(axis="both", labelsize=10)
        rax.grid(axis="y", linestyle="--", alpha=0.25)

    handles, labels = hist_axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        frameon=False,
        fontsize=12,
    )
    fig.supxlabel(r"Observable [rad/$\pi$]", fontsize=12)
    fig.supylabel("Events", fontsize=12)
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
        hist2d_kwargs = {"bins": [bins, bins], "cmap": "viridis"}
        if observable["log"]:
            hist2d_kwargs["norm"] = LogNorm(vmin=1, vmax=observable["vmax"])
        else:
            hist2d_kwargs.update(vmin=1, vmax=observable["vmax"])

        image = ax.hist2d(pred, truth, **hist2d_kwargs)[3]
        ax.plot(
            [bins[0], bins[-1]],
            [bins[0], bins[-1]],
            color="gainsboro",
            linestyle="--",
            linewidth=1.3,
        )
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bins[0], bins[-1])
        ax.set_title(f"{observable['label']}  RMSE = {rmse:.2f}", fontsize=13, pad=7)
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

    txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=2, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.set_ylim(0, 1.15 * max(pred_counts.max(), truth_counts.max(), 1))
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="major", pad=12)

    # Bottom panel: Pred/True ratio
    pred_counts = pred_counts.astype(float)
    truth_counts = truth_counts.astype(float)
    bin_centers = 0.5 * (bins_edges[1:] + bins_edges[:-1])

    valid = (pred_counts > 0) & (truth_counts > 0)
    ratio = np.full_like(pred_counts, np.nan, dtype=float)
    ratio_err = np.full_like(pred_counts, np.nan, dtype=float)

    ratio[valid] = pred_counts[valid] / truth_counts[valid]
    # Propagated Poisson uncertainty for r = N_pred / N_true:
    # sigma_r = r * sqrt(1/N_pred + 1/N_true)
    ratio_err[valid] = ratio[valid] * np.sqrt(1.0 / pred_counts[valid] + 1.0 / truth_counts[valid])

    # Reference uncertainty band from denominator (truth) statistics around unity
    truth_rel_err = np.full_like(truth_counts, np.nan, dtype=float)
    truth_mask = truth_counts > 0
    truth_rel_err[truth_mask] = 1.0 / np.sqrt(truth_counts[truth_mask])
    band_low = 1.0 - truth_rel_err
    band_high = 1.0 + truth_rel_err
    rax.fill_between(
        bin_centers, band_low, band_high, step="mid", color="gray", alpha=0.25, linewidth=0
    )

    y_min, y_max = ratio_ylim
    span = y_max - y_min

    in_view = valid & (ratio >= y_min) & (ratio <= y_max)
    overflow_high = valid & (ratio > y_max)
    overflow_low = valid & (ratio < y_min)

    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.3)
    rax.errorbar(
        bin_centers[in_view],
        ratio[in_view],
        yerr=ratio_err[in_view],
        fmt="o",
        color="black",
        markersize=3,
        linewidth=1,
        capsize=0,
    )

    # Draw arrows at the panel edge for overflow points
    y_top = y_max
    y_top_from = y_max - 0.15 * span
    y_bot = y_min
    y_bot_from = y_min + 0.15 * span

    for x in bin_centers[overflow_high]:
        rax.annotate(
            "",
            xy=(x, y_top),
            xytext=(x, y_top_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )

    for x in bin_centers[overflow_low]:
        rax.annotate(
            "",
            xy=(x, y_bot),
            xytext=(x, y_bot_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )

    rax.set_xlabel(name + " [" + unit + "]", loc="right")
    rax.set_ylabel(ratio_text)
    rax.set_ylim(*ratio_ylim)
    rax.grid(axis="y", linestyle="--", alpha=0.35)
    rax.tick_params(axis="both", which="major", pad=10)

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
    offset=0.5,
    savepath=None,
):
    err = 0.2
    cor_mask = np.abs(_rel_err_func(pred, truth)) <= err  # set 20% relative error cut
    fig, ax = plt.subplots()
    if log:
        norm = LogNorm(vmin=1, vmax=vmax)
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", norm=norm)
    else:
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", vmin=1, vmax=vmax)

    ax.plot(bins_edges, bins_edges, color="gainsboro", linestyle="--")

    ax.set_xlabel(f"{xlabel} [{unit}]")
    ax.set_ylabel(f"{ylabel} [{unit}]")
    ax.set_title(f"{name}" + f" (RMSE: {_rmse(pred, truth):.2f})", loc="right")
    print(f"Rel err < 20%: {100*np.sum(cor_mask)/len(truth):.2f} %")

    txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=2, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.tick_params(axis="both", which="major", pad=10)

    ticks = np.linspace(bins_edges[0], bins_edges[-1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    x_min, x_max = bins_edges[0], bins_edges[-1]
    y_min, y_max = bins_edges[0], bins_edges[-1]
    x_margin = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + x_margin)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    ax.set_aspect("equal", adjustable="box")  # Make plot square
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
    if log:
        norm = LogNorm(vmin=1, vmax=vmax)
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", norm=norm)
    else:
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", vmin=1, vmax=vmax)
    ax.set_xlabel(rf"$\Delta_\text{{res}}${name_pos} [{unit}]")
    ax.set_ylabel(rf"$\Delta_\text{{res}}${name_neg} [{unit}]")
    txt = hep.atlas.label("   " + ATLAS_LABEL_TEXT, data=True, loc=0, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.tick_params(axis="both", which="major", pad=10)
    ticks = np.linspace(bins_edges[0], bins_edges[-1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    ax.set_aspect("equal", adjustable="box")  # Make plot square
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


LOSS_COMPONENTS = [
    ("huber", r"Std $W$ 4-vec Huber"),
    ("higgs_mass", r"$m_H$ Huber"),
    ("alpha_mmd", r"$\alpha$ MMD"),
    ("mass_mmd", r"Joint $m_W$ MMD"),
    ("w_mass_huber", r"$m_W$ Huber"),
    ("angular_mmd", "Angular MMD"),
    ("dmet", r"$\Delta \mathrm{MET}$"),
]

DEFAULT_LOSS_WEIGHTS = {
    "huber": 1.0,
    "higgs_mass": 0.0,
    "alpha_mmd": 0.0,
    "mass_mmd": 0.0,
    "w_mass_huber": 0.0,
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
    print("Best loss values at best epoch (min val_loss):")
    print(
        f"  total: loss={_fmt_value(_value_at(data['totals']['train'], best_epoch))} "
        f"val_loss={_fmt_value(_value_at(data['totals']['val'], best_epoch))}"
    )
    for name, label in LOSS_COMPONENTS:
        if name not in data["raw"]["val"]:
            continue
        train = _fmt_value(_value_at(data["raw"]["train"][name], best_epoch))
        val = _fmt_value(_value_at(data["raw"]["val"][name], best_epoch))
        weighted_train = _fmt_value(_value_at(data["contributions"]["train"][name], best_epoch))
        weighted_val = _fmt_value(_value_at(data["contributions"]["val"][name], best_epoch))
        print(
            f"  {label}: train={train} val={val} weighted_train={weighted_train} weighted_val={weighted_val}"
        )
    fig_raw, raw_axes = plt.subplots(2, 4, figsize=(16, 8.5), sharex=True, layout="constrained")
    fig_weighted, weighted_axes = plt.subplots(
        2, 4, figsize=(16, 8.5), sharex=True, layout="constrained"
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

        plt.figure(figsize=(16, max(8, 0.35 * len(pair_grad_cols))))
        plt.imshow(
            heatmap, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
        )
        plt.colorbar(label="Gradient cosine")
        plt.yticks(range(len(heatmap.index)), heatmap.index)
        tick_positions = np.linspace(
            0, len(heatmap.columns) - 1, min(10, len(heatmap.columns)), dtype=int
        )
        plt.xticks(tick_positions, heatmap.columns[tick_positions])
        plt.xlabel(x_col.capitalize())
        plt.ylabel("Loss pair")
        plt.title("Pairwise Similarity Over Training")
        plt.tight_layout()
        plt.show()

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
            plt.figure(figsize=(16, max(8, 0.35 * len(total_heatmap))))
            plt.imshow(
                total_heatmap,
                aspect="auto",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            plt.colorbar(label="Gradient cosine")
            plt.yticks(range(len(total_heatmap.index)), total_heatmap.index)
            tick_positions = np.linspace(
                0, len(total_heatmap.columns) - 1, min(10, len(total_heatmap.columns)), dtype=int
            )
            plt.xticks(tick_positions, total_heatmap.columns[tick_positions])
            plt.xlabel(x_col.capitalize())
            plt.ylabel("Loss")
            plt.title("Loss-vs-Total Similarity Over Training")
            plt.tight_layout()
            plt.show()
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
            plt.figure(figsize=(16, max(8, 0.35 * len(rest_heatmap))))
            plt.imshow(
                rest_heatmap,
                aspect="auto",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            plt.colorbar(label="Gradient cosine")
            plt.yticks(range(len(rest_heatmap.index)), rest_heatmap.index)
            tick_positions = np.linspace(
                0, len(rest_heatmap.columns) - 1, min(10, len(rest_heatmap.columns)), dtype=int
            )
            plt.xticks(tick_positions, rest_heatmap.columns[tick_positions])
            plt.xlabel(x_col.capitalize())
            plt.ylabel("Loss")
            plt.title("Loss-vs-Rest Similarity Over Training")
            plt.tight_layout()
            plt.show()
    else:
        print("No populated loss-vs-rest grad_cos metrics found.")

    print(
        f"Visualized {len(pair_grad_cols)} pairwise, {len(total_grad_cols)} total, "
        f"and {len(rest_grad_cols)} rest grad_cos metrics across {len(df_grad)} logged rows."
    )


# ---------------------------------------------------------------------------
# Disabled diagnostics moved from notebooks/visualize.ipynb.
# Kept as comments: re-enable by turning the blocks below into live functions.
# ---------------------------------------------------------------------------

# ===== input-gaussianization (notebook cell) =====
# # Diagnose the production neural-input representation without changing raw inputs.
# from data.preprocessing import neural_input_features_numpy
# from matplotlib.lines import Line2D
# from scipy import stats as scipy_stats


# feature_names = [
#     "lep+ px", "lep+ py", "lep+ pz", "lep+ log1p(E)",
#     "lep- px", "lep- py", "lep- pz", "lep- log1p(E)",
#     "jet0 px", "jet0 py", "jet0 pz", "jet0 log1p(E)",
#     "jet1 px", "jet1 py", "jet1 pz", "jet1 log1p(E)",
#     "MET px", "MET py", "m_ll", "deta_ll",
#     "dphi_ll",
# ]

# test_candidate = neural_input_features_numpy(train_features)
# checkpoint_mean = model.model.norm.mean.detach().cpu().numpy()
# checkpoint_std = model.model.norm.std.detach().cpu().numpy()
# # if checkpoint_mean.shape != (21,) or checkpoint_std.shape != (21,):
#     # raise ValueError("Checkpoint normalization must contain 21 means and scales")
# if not np.isfinite(checkpoint_mean).all() or not np.isfinite(checkpoint_std).all():
#     raise ValueError("Checkpoint normalization statistics must be finite")
# if np.any(checkpoint_std <= 0.0):
#     raise ValueError("Checkpoint normalization scales must be positive")
# features_z = (test_candidate - checkpoint_mean) / checkpoint_std

# # if features_z.shape[1] != 21 or not np.isfinite(features_z).all():
# #     raise ValueError("Candidate standardized features must be finite with 21 columns")

# z_limits = (-6, 6)
# z_edges = np.linspace(*z_limits, 61)
# all_events = np.ones(len(train_features), dtype=bool)
# jet0_padding = (train_features[:, 8:12] == 0.0).all(axis=1)
# jet1_padding = (train_features[:, 12:16] == 0.0).all(axis=1)
# jet0_present = ~jet0_padding
# jet1_present = ~jet1_padding
# jet_present_masks = {**{i: jet0_present for i in range(8, 12)},
#                      **{i: jet1_present for i in range(12, 16)}}
# jet_padding_masks = {**{i: jet0_padding for i in range(8, 12)},
#                      **{i: jet1_padding for i in range(12, 16)}}
# fig, axes = plt.subplots(4, 6, figsize=(20, 12), constrained_layout=True)

# summary_rows = []
# for ax in axes.flat[len(feature_names):]:
#     fig.delaxes(ax)
# for i, ax in enumerate(axes.flat[:len(feature_names)]):
#     present_mask = jet_present_masks.get(i, all_events)
#     padding_mask = jet_padding_masks.get(i)
#     values = features_z[present_mask, i]
#     plot_limits = z_limits
#     if padding_mask is not None:
#         padding_location = -checkpoint_mean[i] / checkpoint_std[i]
#         padding_margin = (z_edges[1] - z_edges[0]) / 2.0
#         plot_limits = (min(z_limits[0], padding_location - padding_margin),
#                        max(z_limits[1], padding_location + padding_margin))
#     plot_edges = np.linspace(*plot_limits, len(z_edges))
#     if len(values):
#         event_weights = np.full(len(values), 1.0 / len(train_features))
#         present_label = "Present fraction / bin" if padding_mask is not None else "Test fraction / bin"
#         ax.hist(values, bins=plot_edges, weights=event_weights, histtype="step",
#                 linewidth=1.5, color="black", label=present_label)
#     if padding_mask is not None:
#         padding_values = features_z[padding_mask, i]
#         if len(padding_values):
#             padding_weights = np.full(len(padding_values), 1.0 / len(train_features))
#             ax.hist(padding_values, bins=plot_edges, weights=padding_weights, histtype="step",
#                     linewidth=1.5, color="tab:blue", label="Padding fraction / bin")
#     if len(values):
#         distribution_summary = {
#             "z_mean": values.mean(),
#             "z_std": values.std(),
#             "skew": scipy_stats.skew(values),
#             "kurtosis": scipy_stats.kurtosis(values),
#             "|z|>2 [%]": 100.0 * np.mean(np.abs(values) > 2),
#             "KS p-value": scipy_stats.kstest(values, "norm").pvalue,
#         }
#     else:
#         distribution_summary = {
#             metric: np.nan
#             for metric in ("z_mean", "z_std", "skew", "kurtosis", "|z|>2 [%]", "KS p-value")
#         }
#     bin_width = plot_edges[1] - plot_edges[0]
#     reference_scale = len(values) / len(train_features) if padding_mask is not None else 1.0
#     reference_grid = np.linspace(*plot_limits, 400)
#     ax.plot(reference_grid, scipy_stats.norm.pdf(reference_grid) * bin_width * reference_scale,
#             color="tab:red", linewidth=1.8,
#             label=r"$\mathcal{N}(0,1)$")
#     ax.set_xlim(plot_limits)
#     ax.set_xlabel("checkpoint z", fontsize=8)
#     ax.set_yscale("log")
#     ax.set_title(f"[{i}] {feature_names[i]}", fontsize=11)
#     ax.tick_params(axis="both", labelsize=8)
#     n_present = len(values)
#     n_padding = np.count_nonzero(padding_mask) if padding_mask is not None else 0
#     summary_rows.append({
#         "feature": feature_names[i],
#         "n_events": n_present,
#         "n_present": n_present,
#         "n_padding": n_padding,
#         "padding_fraction": n_padding / len(train_features),
#         **distribution_summary,
#         "checkpoint scale": checkpoint_std[i],
#     })

# legend_handles = [
#     Line2D([], [], color="black", linewidth=1.5, label="Present fraction / bin"),
#     Line2D([], [], color="tab:blue", linewidth=1.5, label="Padding fraction / bin"),
#     Line2D([], [], color="tab:red", linewidth=1.8, label=r"$\mathcal{N}(0,1)$"),
# ]
# fig.legend(handles=legend_handles, loc="outside upper right", frameon=False)
# fig.suptitle("Checkpoint-standardized test inputs", fontsize=16,
#              fontweight="semibold")
# plt.show()

# summary_df = pd.DataFrame(summary_rows).set_index("feature")
# well_conditioned = (
#     (summary_df["z_mean"].abs() < 0.05)
#     & ((summary_df["z_std"] - 1.0).abs() < 0.1)
# )
# print(f"{well_conditioned.sum()} / {len(summary_df)} features are well-conditioned "
#       f"(|z mean| < 0.05 and |z std - 1| < 0.1).")
# display(summary_df.round(4))


# ===== truth-angle-feature-correlation (notebook cell) =====
# # Truth-angle conditioning feature diagnostic
# aligned_inputs = np.asarray(train_features)[np.asarray(angular_valid, dtype=bool)]
# aligned_truth_angles = true_ang.detach().cpu().numpy() if hasattr(true_ang, "detach") else np.asarray(true_ang)
# if aligned_inputs.ndim != 2 or aligned_inputs.shape[1] != 21:
# #     raise ValueError(f"Expected raw inputs with shape (events, 21), got {aligned_inputs.shape}")
# # if aligned_truth_angles.ndim != 2 or aligned_truth_angles.shape[1] != 8:
#     raise ValueError(f"Expected truth angles with shape (events, 8), got {aligned_truth_angles.shape}")
# if len(aligned_inputs) != len(aligned_truth_angles):
#     raise ValueError("Aligned reconstructed features and truth angles have different row counts")

# def _wrapped_delta_phi(phi_a, phi_b):
#     return np.arctan2(np.sin(phi_a - phi_b), np.cos(phi_a - phi_b))

# def _pt_eta_phi(px, py, pz):
#     pt_value = np.hypot(px, py)
#     eta_value = np.arcsinh(np.divide(pz, pt_value, out=np.full_like(pt_value, np.nan), where=pt_value > 0))
#     return pt_value, eta_value, np.arctan2(py, px)

# lplus_px, lplus_py, lplus_pz, lplus_energy = aligned_inputs[:, 0:4].T
# lminus_px, lminus_py, lminus_pz, lminus_energy = aligned_inputs[:, 4:8].T
# met_px, met_py = aligned_inputs[:, 16:18].T
# pt_lplus, eta_lplus, phi_lplus = _pt_eta_phi(lplus_px, lplus_py, lplus_pz)
# pt_lminus, eta_lminus, phi_lminus = _pt_eta_phi(lminus_px, lminus_py, lminus_pz)
# met, _, phi_met = _pt_eta_phi(met_px, met_py, np.zeros_like(met_px))
# ll_px, ll_py = lplus_px + lminus_px, lplus_py + lminus_py
# ll_pz, ll_energy = lplus_pz + lminus_pz, lplus_energy + lminus_energy
# pt_ll, eta_ll, phi_ll = _pt_eta_phi(ll_px, ll_py, ll_pz)
# m_ll_sq = ll_energy**2 - ll_px**2 - ll_py**2 - ll_pz**2
# m_ll = np.where(m_ll_sq >= -1.0e-6, np.sqrt(np.clip(m_ll_sq, 0.0, None)), np.nan)
# deta_ll = eta_lplus - eta_lminus
# dphi_ll = _wrapped_delta_phi(phi_lplus, phi_lminus)
# dphi_lplus_met = _wrapped_delta_phi(phi_lplus, phi_met)
# dphi_lminus_met = _wrapped_delta_phi(phi_lminus, phi_met)
# dphi_ll_met = _wrapped_delta_phi(phi_ll, phi_met)
# dR_ll = np.hypot(deta_ll, dphi_ll)
# mT_lplus_met = np.sqrt(np.clip(2.0 * pt_lplus * met * (1.0 - np.cos(dphi_lplus_met)), 0.0, None))
# mT_lminus_met = np.sqrt(np.clip(2.0 * pt_lminus * met * (1.0 - np.cos(dphi_lminus_met)), 0.0, None))
# et_ll = np.sqrt(m_ll**2 + pt_ll**2)
# mT_ll_met = np.sqrt(np.clip((et_ll + met)**2 - (ll_px + met_px)**2 - (ll_py + met_py)**2, 0.0, None))

# # Values paired with True indicate periodic quantities requiring circular treatment.
# hl_angle_features = {
#     "pt_lplus": (pt_lplus, False), "eta_lplus": (eta_lplus, False), "phi_lplus": (phi_lplus, True),
#     "pt_lminus": (pt_lminus, False), "eta_lminus": (eta_lminus, False), "phi_lminus": (phi_lminus, True),
#     "MET": (met, False), "phi_MET": (phi_met, True),
#     "pt_ll": (pt_ll, False), "eta_ll": (eta_ll, False), "phi_ll": (phi_ll, True),
#     "m_ll": (m_ll, False), "deta_ll": (deta_ll, False), "dphi_ll": (dphi_ll, True),
#     "dphi_lplus_MET": (dphi_lplus_met, True), "dphi_lminus_MET": (dphi_lminus_met, True),
#     "dphi_ll_MET": (dphi_ll_met, True),
#     # "mT_lplus_met": (mT_lplus_met, False), "dR_ll": (dR_ll, False),
#     # "mT_lminus_met": (mT_lminus_met, False), "mT_ll_met": (mT_ll_met, False),
# }
# truth_angle_targets = {
#     "theta+": (aligned_truth_angles[:, 0], False), "phi+": (aligned_truth_angles[:, 1], True),
#     "theta-": (aligned_truth_angles[:, 2], False), "phi-": (aligned_truth_angles[:, 3], True),
#     "sum theta": (aligned_truth_angles[:, 4], False), "diff theta": (aligned_truth_angles[:, 5], False),
#     "sum phi": (aligned_truth_angles[:, 6], True), "diff phi": (aligned_truth_angles[:, 7], True),
# }

# def _association_strength(feature, target, feature_periodic=False, target_periodic=False):
#     feature, target = np.asarray(feature), np.asarray(target)
#     finite = np.isfinite(feature) & np.isfinite(target)
#     if finite.sum() < 3:
#         return np.nan
#     feature, target = feature[finite], target[finite]
#     feature_components = np.column_stack((np.sin(feature), np.cos(feature))) if feature_periodic else feature[:, None]
#     target_components = np.column_stack((np.sin(target), np.cos(target))) if target_periodic else target[:, None]
#     feature_components = feature_components[:, np.std(feature_components, axis=0) > np.finfo(float).eps]
#     target_components = target_components[:, np.std(target_components, axis=0) > np.finfo(float).eps]
#     if feature_components.shape[1] == 0 or target_components.shape[1] == 0:
#         return np.nan
#     if len(feature) <= feature_components.shape[1] + target_components.shape[1]:
#         return np.nan
#     joined = np.column_stack((feature_components, target_components))
#     correlation = np.corrcoef(joined, rowvar=False)
#     feature_dim = feature_components.shape[1]
#     cross = correlation[:feature_dim, feature_dim:]
#     if not feature_periodic and not target_periodic:
#         return float(np.clip(cross[0, 0], -1.0, 1.0))
#     feature_corr = correlation[:feature_dim, :feature_dim]
#     target_corr = correlation[feature_dim:, feature_dim:]
#     canonical_sq = np.linalg.eigvals(np.linalg.pinv(feature_corr) @ cross @ np.linalg.pinv(target_corr) @ cross.T)
#     return float(np.sqrt(np.clip(np.max(np.real(canonical_sq)), 0.0, 1.0)))

# feature_target_association = np.array([
#     [_association_strength(feature, target, feature_periodic, target_periodic)
#      for target, target_periodic in truth_angle_targets.values()]
#     for feature, feature_periodic in hl_angle_features.values()
# ])
# feature_target_association_fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
# image = ax.imshow(feature_target_association, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
# ax.set_xticks(range(len(truth_angle_targets)), labels=list(truth_angle_targets), rotation=35, ha="right")
# ax.set_yticks(range(len(hl_angle_features)), labels=list(hl_angle_features))
# # ax.set_title("Reconstructed feature association with truth rest-frame angles")
# for row in range(feature_target_association.shape[0]):
#     for column in range(feature_target_association.shape[1]):
#         value = feature_target_association[row, column]
#         ax.text(column, row, "--" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=10, color="white" if np.isfinite(value) and abs(value) > 0.55 else "black")
# # feature_target_association_fig.colorbar(image, ax=ax, label="Association coefficient")
# # feature_target_association_fig.text(0.5, 0.005, "Signed Pearson r for linear-linear cells; nonnegative circular-aware strength when either quantity is phi-like.", ha="center", fontsize=9)
# plt.show()

# print("Strongest reconstructed features by truth target:")
# feature_names = list(hl_angle_features)
# for column, target_name in enumerate(truth_angle_targets):
#     values = feature_target_association[:, column]
#     ranked = [index for index in np.argsort(np.nan_to_num(np.abs(values), nan=-1.0))[::-1] if np.isfinite(values[index])][:5]
#     summary = ", ".join(f"{feature_names[index]} ({values[index]:+.3f})" for index in ranked)
#     print(f"  {target_name:>10}: {summary}")
