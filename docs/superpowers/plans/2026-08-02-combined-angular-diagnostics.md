# Combined Angular Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one runnable notebook cell that renders the twelve angular observables as three slide-friendly figures containing paired 1D and 2D panels.

**Architecture:** Keep the existing plotting utilities unchanged. Add a local `plot_angular_grid` helper to `notebooks/visualize.ipynb`, describe each plot with a small dictionary, and invoke the helper once for each of the three approved observable groups.

**Tech Stack:** Python, NumPy, Matplotlib, Jupyter notebook JSON, nbformat

## Global Constraints

- Modify only `notebooks/visualize.ipynb`; do not change `notebooks/plottingtool.py`.
- Generate exactly three figures with four observable rows and two columns each.
- Draw 1D Pred/True overlays without ratio panels and 2D Pred-vs-True histograms.
- Express all angular axes in `rad/$\pi$` by dividing values by `np.pi`.
- Preserve the existing bin ranges and logarithmic versus linear 2D scaling.
- Filter non-finite pairs and reject shape mismatches or observables with no finite pairs.
- Do not execute the full data-dependent notebook during automated verification.

---

### Task 1: Add the Combined Angular Plot Cell

**Files:**
- Modify: `notebooks/visualize.ipynb`, immediately after the `# diff of theta_phi` code cell
- Test: `notebooks/visualize.ipynb` notebook structure and Python syntax

**Interfaces:**
- Consumes: the twelve existing `*_pred` and `*_true` NumPy arrays created by the four preceding angular-feature cells
- Produces: `plot_angular_grid(observables, title, shared_colorbar=False) -> tuple[matplotlib.figure.Figure, numpy.ndarray]`, plus `angular_fig`, `mixed_sum_fig`, and `mixed_diff_fig`

- [ ] **Step 1: Verify that no combined helper already exists**

Run:

```bash
rg -n 'plot_angular_grid|angular_fig|mixed_sum_fig|mixed_diff_fig' notebooks/visualize.ipynb
```

Expected: no matches.

- [ ] **Step 2: Add one code cell after the mixed theta-phi difference cell**

Insert a code cell with empty outputs and this source:

```python
from matplotlib.colors import LogNorm


def plot_angular_grid(observables, title, shared_colorbar=False):
    fig, axes = plt.subplots(
        len(observables),
        2,
        figsize=(14, 3.4 * len(observables)),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )

    shared_mappable = None
    for row, observable in enumerate(observables):
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
        pred = pred[finite]
        truth = truth[finite]
        bins = observable["bins"]

        hist_ax, corr_ax = axes[row]
        hist_ax.hist(pred, bins=bins, linewidth=2, color="red", histtype="step", label="Pred")
        hist_ax.hist(truth, bins=bins, linewidth=2, color="blue", histtype="step", label="True")
        hist_ax.set_xlim(bins[0], bins[-1])
        hist_ax.set_ylabel("Events")
        hist_ax.set_xlabel(rf"{observable['label']} [rad/$\pi$]")
        hist_ax.grid(axis="y", linestyle="--", alpha=0.25)

        hist2d_kwargs = {
            "bins": [bins, bins],
            "cmap": "viridis",
        }
        if observable["log"]:
            hist2d_kwargs["norm"] = LogNorm(vmin=1, vmax=observable["vmax"])
        else:
            hist2d_kwargs.update(vmin=1, vmax=observable["vmax"])

        image = corr_ax.hist2d(pred, truth, **hist2d_kwargs)[3]
        corr_ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color="gainsboro", linestyle="--")
        corr_ax.set_xlim(bins[0], bins[-1])
        corr_ax.set_ylim(bins[0], bins[-1])
        corr_ax.set_xlabel(r"Pred [rad/$\pi$]")
        corr_ax.set_ylabel(r"True [rad/$\pi$]")
        corr_ax.set_title(observable["label"], loc="right")
        corr_ax.set_aspect("equal", adjustable="box")

        if shared_colorbar:
            shared_mappable = image
        else:
            fig.colorbar(image, ax=corr_ax, label="Events")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False)
    fig.suptitle(title, fontsize=18)
    if shared_colorbar:
        fig.colorbar(shared_mappable, ax=axes[:, 1], label="Events", shrink=0.9)
    plt.show()
    return fig, axes


angular_fig, angular_axes = plot_angular_grid(
    [
        {
            "pred": sum_theta_pred,
            "truth": sum_theta_true,
            "label": r"$\sum_{+-}\theta^*_{\ell}$",
            "bins": np.linspace(0, 2, 51),
            "log": True,
            "vmax": 8e2,
        },
        {
            "pred": diff_theta_pred,
            "truth": diff_theta_true,
            "label": r"$\Delta_{+-}\theta^*_{\ell}$",
            "bins": np.linspace(-1, 1, 61),
            "log": True,
            "vmax": 8e2,
        },
        {
            "pred": sum_phi_pred,
            "truth": sum_phi_true,
            "label": r"$\sum_{+-}\phi^*_{\ell}$",
            "bins": np.linspace(-1, 1, 61),
            "log": False,
            "vmax": 3e2,
        },
        {
            "pred": diff_phi_pred,
            "truth": diff_phi_true,
            "label": r"$\Delta_{+-}\phi^*_{\ell}$",
            "bins": np.linspace(-1, 1, 61),
            "log": False,
            "vmax": 3e2,
        },
    ],
    "Angular sums and differences",
)

mixed_sum_fig, mixed_sum_axes = plot_angular_grid(
    [
        {"pred": sum_theta_pos_phi_pos_pred, "truth": sum_theta_pos_phi_pos_true, "label": r"$\sum_{++}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": sum_theta_pos_phi_neg_pred, "truth": sum_theta_pos_phi_neg_true, "label": r"$\sum_{+-}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": sum_theta_neg_phi_pos_pred, "truth": sum_theta_neg_phi_pos_true, "label": r"$\sum_{-+}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": sum_theta_neg_phi_neg_pred, "truth": sum_theta_neg_phi_neg_true, "label": r"$\sum_{--}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    ],
    "Mixed angular sums",
    shared_colorbar=True,
)

mixed_diff_fig, mixed_diff_axes = plot_angular_grid(
    [
        {"pred": diff_theta_pos_phi_pos_pred, "truth": diff_theta_pos_phi_pos_true, "label": r"$\Delta_{++}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": diff_theta_pos_phi_neg_pred, "truth": diff_theta_pos_phi_neg_true, "label": r"$\Delta_{+-}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": diff_theta_neg_phi_pos_pred, "truth": diff_theta_neg_phi_pos_true, "label": r"$\Delta_{-+}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
        {"pred": diff_theta_neg_phi_neg_pred, "truth": diff_theta_neg_phi_neg_true, "label": r"$\Delta_{--}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    ],
    "Mixed angular differences",
    shared_colorbar=True,
)
```

- [ ] **Step 3: Validate notebook JSON and code-cell syntax**

Run:

```bash
python - <<'PY'
from pathlib import Path

import nbformat

path = Path("notebooks/visualize.ipynb")
notebook = nbformat.read(path, as_version=4)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{path}:cell-{index}", "exec")

matches = [cell for cell in notebook.cells if "def plot_angular_grid" in cell.source]
assert len(matches) == 1, f"expected one combined plotting cell, found {len(matches)}"
source = matches[0].source
assert source.count("plot_angular_grid(") == 4
assert all(name in source for name in ("angular_fig", "mixed_sum_fig", "mixed_diff_fig"))
print("Notebook JSON, code syntax, and combined plotting cell validated")
PY
```

Expected: `Notebook JSON, code syntax, and combined plotting cell validated`.

- [ ] **Step 4: Inspect the focused diff**

Run:

```bash
git diff -- notebooks/visualize.ipynb
```

Expected: one added code cell after the mixed theta-phi difference cell, with no changes to existing cell outputs or sources.

- [ ] **Step 5: Manually verify the rendered figures**

Run the new cell after the four existing angular calculation cells. Confirm that it displays exactly three figures, each with four rows; every row has a Pred/True distribution and a square prediction-versus-truth panel; the first figure has per-panel colorbars; and each mixed figure has one shared logarithmic colorbar.
