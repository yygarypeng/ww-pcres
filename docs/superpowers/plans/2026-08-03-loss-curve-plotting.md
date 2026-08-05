# Loss-Curve Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale loss plotting in `notebooks/visualize.ipynb` with tested raw, weighted-total, and weighted-contribution diagnostics that match the current model and logger.

**Architecture:** Keep the feature notebook-local. The loss-curve cell will expose small data-preparation helpers plus one `plot_loss_curves(metrics_path, cfg)` entry point, then invoke it for `LOG_DIR`; tests will load that cell from notebook JSON and execute it against synthetic sparse `CSVLogger` data.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, Jupyter notebook JSON, pytest

## Global Constraints

- Current component keys are exactly `huber`, `higgs_mass`, `kinematic_loss_mmd`, `w_mass_huber`, `angular_loss_mmd`, and `dmet`.
- Checkpoint selection is exactly minimum `val_loss`, matching `train/train.py`.
- Preserve epoch 0 and align every series by explicit epoch values.
- Use linear y-scales because MMD estimates may be negative.
- Support both configured static weights and per-epoch adaptive `loss_weight/<name>` values.
- Do not change model losses, metric logging, callback monitoring, or gradient-cosine plots.
- Do not add a reusable application plotting module.

---

### Task 1: Tested Loss-Curve Notebook Cell

**Files:**
- Create: `tests/test_loss_curve_notebook.py`
- Modify: `notebooks/visualize.ipynb` cell `861fd17d` (rename the cell id to `loss-curves`)

**Interfaces:**
- Consumes: a `pandas.DataFrame` shaped like Lightning `CSVLogger` output, `cfg["parameters"]["loss_weights"]`, and `cfg["parameters"]["adaptive_loss_weights"]`
- Produces: `_metric_series(df, column) -> pandas.Series`, `_epoch_weights(df, name, epochs, initial_weight, adaptive) -> pandas.Series`, `_prepare_loss_plot_data(df, cfg) -> dict`, and `plot_loss_curves(metrics_path, cfg) -> dict | None`
- `plot_loss_curves` returns diagnostics containing `best_epoch`, `final_epoch`, `raw`, `weights`, `contributions`, `mismatches`, and `figures`; this return value exists for verification while notebook users consume the displayed figures and text.

- [ ] **Step 1: Add a notebook-cell loader and failing static-data test**

Create `tests/test_loss_curve_notebook.py` with the loader and static test below. The loader executes the complete cell with `LOG_DIR=None`, preventing its normal filesystem invocation, and returns the notebook-local functions for direct testing.

```python
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "visualize.ipynb"


def load_loss_cell():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "loss-curves")
    namespace = {
        "LOG_DIR": None,
        "np": np,
        "pd": pd,
        "plt": plt,
    }
    exec("".join(cell["source"]), namespace)
    return namespace


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
    namespace = load_loss_cell()
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

    data = namespace["_prepare_loss_plot_data"](df, cfg)

    assert data["best_epoch"] == 1
    assert data["final_epoch"] == 2
    assert data["raw"]["train"]["huber"].index.tolist() == [0, 1, 2]
    pd.testing.assert_series_equal(
        data["contributions"]["train"]["huber"],
        pd.Series([6.0, 4.0, 2.0], index=pd.Index([0, 1, 2], name="epoch")),
    )
    assert data["mismatches"] == []
```

- [ ] **Step 2: Run the static-data test and confirm the intended failure**

Run: `pytest tests/test_loss_curve_notebook.py::test_static_data_keeps_epoch_zero_and_reconstructs_totals -v`

Expected: FAIL because no notebook cell has id `loss-curves` yet.

- [ ] **Step 3: Replace the stale cell with metric metadata and data preparation**

In `notebooks/visualize.ipynb`, rename cell `861fd17d` to `loss-curves`, clear its outputs, set `execution_count` to `null`, and replace its source. Start the source with the exact metadata and helpers below:

```python
LOSS_COMPONENTS = [
    ("huber", "Standardized Four-Vector Huber"),
    ("higgs_mass", "Higgs Mass"),
    ("kinematic_loss_mmd", "Joint Kinematic MMD"),
    ("w_mass_huber", r"$W$ Mass Huber"),
    ("angular_loss_mmd", "Angular MMD"),
    ("dmet", r"$\Delta \mathrm{MET}$"),
]


def _metric_series(df, column):
    if "epoch" not in df.columns or column not in df.columns:
        return pd.Series(dtype=float, name=column)
    values = df.loc[df[column].notna() & df["epoch"].notna(), ["epoch", column]]
    values = values.drop_duplicates("epoch", keep="last").sort_values("epoch")
    return values.set_index("epoch")[column]


def _epoch_weights(df, name, epochs, initial_weight, adaptive):
    result = pd.Series(float(initial_weight), index=epochs, dtype=float, name=name)
    if not adaptive:
        return result
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
    configured_weights = params.get("loss_weights", {})
    adaptive = bool(params.get("adaptive_loss_weights", False))
    raw = {"train": {}, "val": {}}
    weights = {}
    contributions = {"train": {}, "val": {}}
    unavailable = []

    all_epochs = totals["train"].index.union(totals["val"].index).sort_values()
    for name, label in LOSS_COMPONENTS:
        train = _metric_series(df, f"{name}_loss")
        val = _metric_series(df, f"val_{name}_loss")
        if train.empty or val.empty:
            unavailable.append(label)
            continue
        raw["train"][name] = train
        raw["val"][name] = val
        weights[name] = _epoch_weights(
            df,
            name,
            all_epochs,
            configured_weights.get(name, 0.0),
            adaptive,
        )
        contributions["train"][name] = train * weights[name].reindex(train.index)
        contributions["val"][name] = val * weights[name].reindex(val.index)

    mismatches = []
    for stage in ("train", "val"):
        if not contributions[stage]:
            continue
        reconstructed = pd.concat(contributions[stage], axis=1).sum(axis=1, min_count=1)
        comparison = pd.concat([reconstructed.rename("reconstructed"), totals[stage]], axis=1).dropna()
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
```

Do not add plotting yet. Add a temporary `plot_loss_curves` that reads the CSV, calls `_prepare_loss_plot_data`, and returns the dictionary, plus the normal `LOG_DIR` invocation:

```python
def plot_loss_curves(metrics_path, cfg):
    print(f"Using metrics file: {metrics_path}")
    if not metrics_path.exists():
        print(f"No metrics.csv found at {metrics_path}")
        return None
    try:
        return _prepare_loss_plot_data(pd.read_csv(metrics_path), cfg)
    except (ValueError, pd.errors.EmptyDataError) as error:
        print(f"Cannot plot losses: {error}")
        return None


if "LOG_DIR" not in globals():
    log_dirs = sorted(
        (RUN_DIR / "logs").glob("version_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    LOG_DIR = log_dirs[0] if log_dirs else None

if LOG_DIR is None:
    print("No logs found.")
else:
    loss_diagnostics = plot_loss_curves(LOG_DIR / "metrics.csv", cfg)
```

- [ ] **Step 4: Run the static-data test and confirm it passes**

Run: `pytest tests/test_loss_curve_notebook.py::test_static_data_keeps_epoch_zero_and_reconstructs_totals -v`

Expected: PASS.

- [ ] **Step 5: Add failing adaptive-weight and plotting tests**

Append these tests to `tests/test_loss_curve_notebook.py`:

```python
def test_adaptive_weights_use_last_logged_value_per_epoch_and_forward_fill():
    namespace = load_loss_cell()
    df = sparse_metrics(
        epochs=[0, 1, 2],
        component_values={"huber": ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])},
        weights={"huber": 1.0},
    )
    weight_rows = pd.DataFrame(
        [
            {"epoch": 0, "loss_weight/huber": 1.5},
            {"epoch": 0, "loss_weight/huber": 2.0},
            {"epoch": 2, "loss_weight/huber": 4.0},
        ]
    )
    df = pd.concat([df, weight_rows], ignore_index=True)
    # Make logged totals match adaptive contributions for exact-reconstruction checks.
    df.loc[df["loss"].notna(), "loss"] = [4.0, 4.0, 8.0]
    df.loc[df["val_loss"].notna(), "val_loss"] = [6.0, 6.0, 12.0]
    cfg = {
        "parameters": {
            "loss_weights": {"huber": 1.0},
            "adaptive_loss_weights": True,
        }
    }

    data = namespace["_prepare_loss_plot_data"](df, cfg)

    pd.testing.assert_series_equal(
        data["weights"]["huber"],
        pd.Series([2.0, 2.0, 4.0], index=pd.Index([0, 1, 2], name="epoch"), name="huber"),
    )
    assert data["mismatches"] == []


def test_plot_loss_curves_builds_three_figures_and_warns_on_mismatch(tmp_path, capsys):
    namespace = load_loss_cell()
    weights = {"huber": 2.0}
    df = sparse_metrics(
        epochs=[0, 1],
        component_values={"huber": ([2.0, 1.0], [2.5, 1.5])},
        weights=weights,
    )
    df.loc[df["val_loss"].notna(), "val_loss"] += 1.0
    metrics_path = tmp_path / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    cfg = {"parameters": {"loss_weights": weights, "adaptive_loss_weights": False}}

    diagnostics = namespace["plot_loss_curves"](metrics_path, cfg)

    assert diagnostics["mismatches"] == ["val"]
    assert [len(figure.axes) for figure in diagnostics["figures"]] == [1, 6, 2]
    output = capsys.readouterr().out
    assert "could not be reconstructed exactly for: validation" in output
    plt.close("all")


def test_plot_loss_curves_reports_missing_csv(tmp_path, capsys):
    namespace = load_loss_cell()

    result = namespace["plot_loss_curves"](tmp_path / "missing.csv", {"parameters": {}})

    assert result is None
    assert "No metrics.csv found" in capsys.readouterr().out


def test_plot_loss_curves_reports_empty_csv(tmp_path, capsys):
    namespace = load_loss_cell()
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("")

    result = namespace["plot_loss_curves"](metrics_path, {"parameters": {}})

    assert result is None
    assert "Cannot plot losses" in capsys.readouterr().out
```

- [ ] **Step 6: Run the new tests and confirm the plotting test fails**

Run: `pytest tests/test_loss_curve_notebook.py -v`

Expected: adaptive, missing-file, and empty-file tests pass; plotting test fails with missing `figures` because rendering has not been implemented.

- [ ] **Step 7: Implement the three figures and epoch-aligned summary**

Extend `plot_loss_curves` after `_prepare_loss_plot_data` returns. Use `plt.subplots` to create exactly:

```python
fig_total, ax_total = plt.subplots(figsize=(10.5, 4.2), layout="constrained")
fig_raw, raw_axes = plt.subplots(2, 3, figsize=(15.6, 8.0), sharex=True, layout="constrained")
fig_contrib, contrib_axes = plt.subplots(1, 2, figsize=(15.6, 4.6), sharex=True, sharey=True, layout="constrained")
```

Implement rendering with these exact rules:

- Plot train as `tab:blue` and validation as `tab:red` in the total and raw figures.
- Draw `ax.axvline(best_epoch, color="0.35", linestyle="--", linewidth=1)` on every visible axis.
- Mark `(best_epoch, val_loss)` only in the total panel.
- Give every raw component its own y-scale; do not call `sharey=True` for the raw grid.
- Keep all six raw axes present. For unavailable pairs, turn that axis off.
- Use `ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)` on total and visible raw axes.
- Use one color per component from `plt.get_cmap("tab10")` in both contribution panels.
- Label contribution panels `Training` and `Validation`, share their y-axis, and allow negative values on the linear scale.
- Add figure-level legends so raw axes do not repeat identical legends.

After plotting, print unavailable labels, mismatch warnings, and a summary selected by exact epochs. Add this local formatter to avoid mixing epochs:

```python
def _value_at(series, epoch):
    return None if epoch is None or epoch not in series.index else float(series.loc[epoch])
```

Print the monitor and totals in this form:

```python
print("Monitor metric: val_loss (min)")
print(f"Best epoch: {best_epoch:g}; val_loss={data['totals']['val'].loc[best_epoch]:.6g}")
if final_epoch is not None:
    train_final = _value_at(data["totals"]["train"], final_epoch)
    val_final = _value_at(data["totals"]["val"], final_epoch)
    print(f"Final common epoch: {final_epoch:g}; loss={train_final:.6g}; val_loss={val_final:.6g}")
```

For each available component, print raw train/validation values and weighted train/validation contributions at `best_epoch` and `final_epoch`, skipping a stage value only when that exact epoch is absent. Use the display label from `LOSS_COMPONENTS`.

Print diagnostics with exact phrases tested above:

```python
if data["unavailable"]:
    print("Unavailable train/validation pairs: " + ", ".join(data["unavailable"]))
if data["mismatches"]:
    display_stages = ["validation" if stage == "val" else stage for stage in data["mismatches"]]
    print(
        "Warning: weighted contributions could not be reconstructed exactly for: "
        + ", ".join(display_stages)
        + ". Logger timing or aggregation may differ from epoch reconstruction."
    )
```

Before returning, call `plt.show()` once and set:

```python
data["figures"] = [fig_total, fig_raw, fig_contrib]
return data
```

- [ ] **Step 8: Run notebook-cell tests and confirm all cases pass**

Run: `pytest tests/test_loss_curve_notebook.py -v`

Expected: 5 tests pass.

- [ ] **Step 9: Run focused regression tests for model logging assumptions**

Run: `pytest tests/test_loss_curve_notebook.py tests/test_model_loss.py tests/test_train_overrides.py -v`

Expected: all tests pass, confirming the notebook metadata still matches current loss keys and training configuration behavior.

- [ ] **Step 10: Validate notebook JSON and inspect the edited cell**

Run:

```bash
python -m json.tool notebooks/visualize.ipynb >/dev/null
python -c 'import json; from pathlib import Path; n=json.loads(Path("notebooks/visualize.ipynb").read_text()); c=next(c for c in n["cells"] if c.get("id")=="loss-curves"); assert c["execution_count"] is None; assert c["outputs"] == []; print("loss-curves cell valid")'
```

Expected: `loss-curves cell valid`.

- [ ] **Step 11: Review the final diff for notebook-output noise and unrelated changes**

Run: `git diff --stat -- notebooks/visualize.ipynb tests/test_loss_curve_notebook.py && git diff -- tests/test_loss_curve_notebook.py`

Expected: only one notebook cell is changed, its stale embedded output is removed, and the new test file contains the five specified tests. Do not revert or stage unrelated worktree changes.

- [ ] **Step 12: Commit only the implementation files**

```bash
git add notebooks/visualize.ipynb tests/test_loss_curve_notebook.py
git commit -m "Fix loss curve diagnostics"
```

Expected: one commit containing only the notebook-cell replacement and its tests.
