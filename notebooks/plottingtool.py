import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
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


def plot_angular_1d_grid(observables, title, share_axes=False):
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
            label="Pred",
        )
        ax.hist(
            truth,
            bins=bins,
            linewidth=2,
            color="blue",
            histtype="step",
            label="True",
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
            label="Pred/True",
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

    fig.supxlabel(r"Pred [rad/$\pi$]", fontsize=12)
    fig.supylabel(r"True [rad/$\pi$]", fontsize=12)
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
