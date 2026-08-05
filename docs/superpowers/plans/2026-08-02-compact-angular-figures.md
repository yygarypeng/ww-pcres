# Compact Angular Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the six angular diagnostic figures more compact by removing repeated axis text, sharing compatible axes, and adding normalized RMSE values to every 2D panel.

**Architecture:** Modify only the existing angular plotting cell. Keep the observable definitions and data preparation unchanged; refine both plotting helpers and mark the mixed-group calls as shareable.

**Tech Stack:** Python, NumPy, Matplotlib, Jupyter notebook JSON, nbformat

## Global Constraints

- Modify only notebook cell ID `7b632f5e` in `notebooks/visualize.ipynb`; preserve every other cell and existing user change.
- Preserve six separate 2x2 figures and clear stale outputs after editing.
- Use figure-level axis labels instead of repeated panel labels.
- Share x/y axes and hide interior tick labels only for the two mixed theta-phi groups.
- Keep the non-shared theta/phi panel ticks because their ranges differ.
- Show `RMSE = x.xx` in every 2D title, computed from finite `pred / np.pi` and `truth / np.pi` values.
- Preserve observable mappings, bins, colors, normalization, equality lines, legends, and colorbar behavior.
- Verify with synthetic arrays and Matplotlib's noninteractive `Agg` backend; do not execute the full notebook.
- Do not commit unless explicitly requested.

---

### Task 1: Compact Axes and Add RMSE

**Files:**
- Modify: `notebooks/visualize.ipynb`, cell ID `7b632f5e`
- Test: cell structure, source syntax, and isolated synthetic rendering

**Interfaces:**
- Preserves: `_prepare_angular_data(observable)` and all observable lists and figure variables
- Updates: `plot_angular_1d_grid(observables, title, share_axes=False)` and `plot_angular_2d_grid(observables, title, shared_colorbar=False, share_axes=False)`

- [ ] **Step 1: Run a failing compact-layout check**

Load cell ID `7b632f5e` with `nbformat` and assert that both helper signatures contain `share_axes=False`, the source contains `fig.supxlabel`, and it contains `RMSE = {rmse:.2f}`.

Expected: FAIL before implementation because those features are absent.

- [ ] **Step 2: Replace the two plotting helpers**

Keep `_prepare_angular_data` unchanged and replace the 1D and 2D helper definitions with:

```python
def plot_angular_1d_grid(observables, title, share_axes=False):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 6.8),
        sharex=share_axes,
        sharey=share_axes,
        constrained_layout=True,
    )
    for ax, observable in zip(axes.flat, observables):
        pred, truth = _prepare_angular_data(observable)
        bins = observable["bins"]
        ax.hist(pred, bins=bins, linewidth=2, color="red", histtype="step", label="Pred")
        ax.hist(truth, bins=bins, linewidth=2, color="blue", histtype="step", label="True")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(observable["label"], fontsize=14, pad=7)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if share_axes:
            ax.label_outer()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False, fontsize=12)
    fig.supxlabel(r"Observable [rad/$\pi$]", fontsize=12)
    fig.supylabel("Events", fontsize=12)
    fig.suptitle(title, fontsize=17, fontweight="semibold")
    plt.show()
    return fig, axes


def plot_angular_2d_grid(observables, title, shared_colorbar=False, share_axes=False):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 7.6),
        sharex=share_axes,
        sharey=share_axes,
        constrained_layout=True,
    )
    shared_mappable = None
    for ax, observable in zip(axes.flat, observables):
        pred, truth = _prepare_angular_data(observable)
        bins = observable["bins"]
        rmse = np.sqrt(np.mean((pred - truth) ** 2))
        hist2d_kwargs = {"bins": [bins, bins], "cmap": "viridis"}
        if observable["log"]:
            hist2d_kwargs["norm"] = LogNorm(vmin=1, vmax=observable["vmax"])
        else:
            hist2d_kwargs.update(vmin=1, vmax=observable["vmax"])

        image = ax.hist2d(pred, truth, **hist2d_kwargs)[3]
        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color="gainsboro", linestyle="--", linewidth=1.3)
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
        colorbar = fig.colorbar(shared_mappable, ax=axes.ravel().tolist(), label="Events", shrink=0.88, pad=0.02)
        colorbar.ax.tick_params(labelsize=10)
        colorbar.set_label("Events", fontsize=11)
    plt.show()
    return fig, axes
```

- [ ] **Step 3: Enable sharing only for mixed-group calls**

Leave the two angular calls non-shared. Replace the four mixed-group calls with:

```python
mixed_sum_1d_fig, mixed_sum_1d_axes = plot_angular_1d_grid(mixed_sum_observables, "Mixed angular sums: 1D distributions", share_axes=True)
mixed_sum_2d_fig, mixed_sum_2d_axes = plot_angular_2d_grid(mixed_sum_observables, "Mixed angular sums: 2D correlations", shared_colorbar=True, share_axes=True)
mixed_diff_1d_fig, mixed_diff_1d_axes = plot_angular_1d_grid(mixed_diff_observables, "Mixed angular differences: 1D distributions", share_axes=True)
mixed_diff_2d_fig, mixed_diff_2d_axes = plot_angular_2d_grid(mixed_diff_observables, "Mixed angular differences: 2D correlations", shared_colorbar=True, share_axes=True)
```

- [ ] **Step 4: Clear stale cell execution state and validate structure**

Set the target cell's `execution_count` to `null` and `outputs` to `[]`. Compile every notebook code cell and assert the target contains both updated signatures, two `supxlabel` calls, two `supylabel` calls, four `share_axes=True` call arguments, and `RMSE = {rmse:.2f}`.

Expected: all assertions pass and no other cell is modified.

- [ ] **Step 5: Render synthetic figures and verify compact behavior**

Execute only the target cell with finite synthetic arrays under `Agg`. Assert:

```python
assert len(plt.get_fignums()) == 6
assert all(namespace[name].shape == (2, 2) for name in axes_variable_names)
assert all("RMSE = " in ax.get_title() for name in two_d_axes_names for ax in namespace[name].flat)
assert all(ax.get_title().rsplit("RMSE = ", 1)[1].count(".") == 1 for name in two_d_axes_names for ax in namespace[name].flat)
assert namespace["mixed_sum_2d_axes"][0, 0].get_shared_x_axes().joined(
    namespace["mixed_sum_2d_axes"][0, 0], namespace["mixed_sum_2d_axes"][1, 1]
)
assert not namespace["angular_2d_axes"][0, 0].get_shared_x_axes().joined(
    namespace["angular_2d_axes"][0, 0], namespace["angular_2d_axes"][1, 1]
)
```

Also draw all canvases and require zero Matplotlib warnings.

- [ ] **Step 6: Run repository checks**

Run `pytest` and `git diff --check -- notebooks/visualize.ipynb`.

Expected: 42 tests pass with the existing ONNX tracer warning and no notebook whitespace errors.
