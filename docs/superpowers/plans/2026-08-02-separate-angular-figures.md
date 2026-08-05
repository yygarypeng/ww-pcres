# Separate Angular Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crowded paired-panel output with six readable 2x2 figures, separating 1D distributions from 2D correlations for each angular observable group.

**Architecture:** Keep all logic in the existing combined notebook cell. Use one shared input-preparation helper plus dedicated 1D and 2D figure helpers, then define each observable group once and pass it to both plotting helpers.

**Tech Stack:** Python, NumPy, Matplotlib, Jupyter notebook JSON, nbformat

## Global Constraints

- Modify only the existing combined plotting cell in `notebooks/visualize.ipynb`; preserve all other cells and existing user changes.
- Clear the replaced cell's stale outputs and set its execution count to `null`.
- Generate exactly six figures: separate 1D and 2D figures for each of three observable groups.
- Every figure must use a readable 2x2 grid with one observable per panel.
- Keep Pred red, True blue, `viridis` correlations, existing bin ranges, and existing linear/logarithmic normalization.
- Express angular axes in `rad/$\pi$`, omit ratio panels, and keep the 2D equality line.
- Use one shared legend per 1D figure, per-panel colorbars for mixed-normalization theta/phi correlations, and one shared colorbar for each mixed correlation figure.
- Validate matching shapes, filter paired non-finite values, and reject observables with no finite pairs.
- Do not execute the full data-dependent notebook; verify the isolated cell with synthetic arrays and a noninteractive backend.
- Do not commit unless the user explicitly requests a commit.

---

### Task 1: Replace the Crowded Combined Plot Cell

**Files:**
- Modify: `notebooks/visualize.ipynb`, cell containing `def plot_angular_grid`
- Test: isolated notebook cell structure, syntax, and synthetic rendering

**Interfaces:**
- Consumes: the twelve existing angular `*_pred` and `*_true` arrays
- Produces: `_prepare_angular_data`, `plot_angular_1d_grid`, `plot_angular_2d_grid`, six figure variables, and six `(2, 2)` axes arrays

- [ ] **Step 1: Verify the old crowded layout is present**

Run a read-only `nbformat` check asserting there is exactly one cell containing `def plot_angular_grid`, that it calls `plt.subplots(len(observables), 2, ...)`, and that the cell currently has three figure calls.

Expected: PASS before editing.

- [ ] **Step 2: Replace only that cell's source and clear its execution state**

Use this exact Python source in the existing cell:

```python
from matplotlib.colors import LogNorm


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


def plot_angular_1d_grid(observables, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, observable in zip(axes.flat, observables):
        pred, truth = _prepare_angular_data(observable)
        bins = observable["bins"]
        ax.hist(pred, bins=bins, linewidth=2.2, color="red", histtype="step", label="Pred")
        ax.hist(truth, bins=bins, linewidth=2.2, color="blue", histtype="step", label="True")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(observable["label"], fontsize=17, pad=10)
        ax.set_xlabel(r"Observable [rad/$\pi$]", fontsize=14)
        ax.set_ylabel("Events", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False, fontsize=14)
    fig.suptitle(title, fontsize=20, fontweight="semibold")
    plt.show()
    return fig, axes


def plot_angular_2d_grid(observables, title, shared_colorbar=False):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    shared_mappable = None
    for ax, observable in zip(axes.flat, observables):
        pred, truth = _prepare_angular_data(observable)
        bins = observable["bins"]
        hist2d_kwargs = {"bins": [bins, bins], "cmap": "viridis"}
        if observable["log"]:
            hist2d_kwargs["norm"] = LogNorm(vmin=1, vmax=observable["vmax"])
        else:
            hist2d_kwargs.update(vmin=1, vmax=observable["vmax"])

        image = ax.hist2d(pred, truth, **hist2d_kwargs)[3]
        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color="gainsboro", linestyle="--", linewidth=1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bins[0], bins[-1])
        ax.set_title(observable["label"], fontsize=17, pad=10)
        ax.set_xlabel(r"Pred [rad/$\pi$]", fontsize=14)
        ax.set_ylabel(r"True [rad/$\pi$]", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_aspect("equal", adjustable="box")

        if shared_colorbar:
            shared_mappable = image
        else:
            colorbar = fig.colorbar(image, ax=ax, label="Events", pad=0.02)
            colorbar.ax.tick_params(labelsize=11)
            colorbar.set_label("Events", fontsize=13)

    fig.suptitle(title, fontsize=20, fontweight="semibold")
    if shared_colorbar:
        colorbar = fig.colorbar(shared_mappable, ax=axes.ravel().tolist(), label="Events", shrink=0.88, pad=0.02)
        colorbar.ax.tick_params(labelsize=11)
        colorbar.set_label("Events", fontsize=13)
    plt.show()
    return fig, axes


angular_observables = [
    {"pred": sum_theta_pred, "truth": sum_theta_true, "label": r"$\sum_{+-}\theta^*_{\ell}$", "bins": np.linspace(0, 2, 51), "log": True, "vmax": 8e2},
    {"pred": diff_theta_pred, "truth": diff_theta_true, "label": r"$\Delta_{+-}\theta^*_{\ell}$", "bins": np.linspace(-1, 1, 61), "log": True, "vmax": 8e2},
    {"pred": sum_phi_pred, "truth": sum_phi_true, "label": r"$\sum_{+-}\phi^*_{\ell}$", "bins": np.linspace(-1, 1, 61), "log": False, "vmax": 3e2},
    {"pred": diff_phi_pred, "truth": diff_phi_true, "label": r"$\Delta_{+-}\phi^*_{\ell}$", "bins": np.linspace(-1, 1, 61), "log": False, "vmax": 3e2},
]

mixed_sum_observables = [
    {"pred": sum_theta_pos_phi_pos_pred, "truth": sum_theta_pos_phi_pos_true, "label": r"$\sum_{++}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": sum_theta_pos_phi_neg_pred, "truth": sum_theta_pos_phi_neg_true, "label": r"$\sum_{+-}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": sum_theta_neg_phi_pos_pred, "truth": sum_theta_neg_phi_pos_true, "label": r"$\sum_{-+}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": sum_theta_neg_phi_neg_pred, "truth": sum_theta_neg_phi_neg_true, "label": r"$\sum_{--}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
]

mixed_diff_observables = [
    {"pred": diff_theta_pos_phi_pos_pred, "truth": diff_theta_pos_phi_pos_true, "label": r"$\Delta_{++}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": diff_theta_pos_phi_neg_pred, "truth": diff_theta_pos_phi_neg_true, "label": r"$\Delta_{+-}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": diff_theta_neg_phi_pos_pred, "truth": diff_theta_neg_phi_pos_true, "label": r"$\Delta_{-+}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
    {"pred": diff_theta_neg_phi_neg_pred, "truth": diff_theta_neg_phi_neg_true, "label": r"$\Delta_{--}\theta^*\phi^*$", "bins": np.linspace(0, 1, 51), "log": True, "vmax": 8e2},
]

angular_1d_fig, angular_1d_axes = plot_angular_1d_grid(angular_observables, "Angular sums and differences: 1D distributions")
angular_2d_fig, angular_2d_axes = plot_angular_2d_grid(angular_observables, "Angular sums and differences: 2D correlations")
mixed_sum_1d_fig, mixed_sum_1d_axes = plot_angular_1d_grid(mixed_sum_observables, "Mixed angular sums: 1D distributions")
mixed_sum_2d_fig, mixed_sum_2d_axes = plot_angular_2d_grid(mixed_sum_observables, "Mixed angular sums: 2D correlations", shared_colorbar=True)
mixed_diff_1d_fig, mixed_diff_1d_axes = plot_angular_1d_grid(mixed_diff_observables, "Mixed angular differences: 1D distributions")
mixed_diff_2d_fig, mixed_diff_2d_axes = plot_angular_2d_grid(mixed_diff_observables, "Mixed angular differences: 2D correlations", shared_colorbar=True)
```

- [ ] **Step 3: Validate notebook isolation and Python syntax**

Load the notebook with `nbformat`, compile every code cell, and assert:

```python
matches = [cell for cell in notebook.cells if "def plot_angular_1d_grid" in cell.source]
assert len(matches) == 1
cell = matches[0]
assert cell.execution_count is None
assert cell.outputs == []
assert "def plot_angular_grid" not in cell.source
assert cell.source.count("plot_angular_1d_grid(") == 4
assert cell.source.count("plot_angular_2d_grid(") == 4
```

Expected: all assertions pass.

- [ ] **Step 4: Render the isolated cell with synthetic arrays**

Use Matplotlib's `Agg` backend, provide finite synthetic arrays for every consumed name, execute only the replacement cell, and assert:

```python
figure_axes = (
    ("angular_1d_fig", "angular_1d_axes"),
    ("angular_2d_fig", "angular_2d_axes"),
    ("mixed_sum_1d_fig", "mixed_sum_1d_axes"),
    ("mixed_sum_2d_fig", "mixed_sum_2d_axes"),
    ("mixed_diff_1d_fig", "mixed_diff_1d_axes"),
    ("mixed_diff_2d_fig", "mixed_diff_2d_axes"),
)
assert len(plt.get_fignums()) == 6
for figure_name, axes_name in figure_axes:
    assert namespace[figure_name] is not None
    assert namespace[axes_name].shape == (2, 2)
```

Expected: six figures render without warnings and all six axes arrays have shape `(2, 2)`.

- [ ] **Step 5: Run repository verification and inspect the focused notebook state**

Run `pytest`, `git diff --check -- notebooks/visualize.ipynb`, and a focused `nbformat` inspection of the replacement cell.

Expected: 42 tests pass with the existing ONNX tracer warning; no whitespace errors; only the intended cell has the revised source and cleared output.
