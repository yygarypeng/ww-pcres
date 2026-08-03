# Loss-Curve Plotting Design

## Goal

Replace the stale loss-plotting cell in `notebooks/visualize.ipynb` with a notebook-local diagnostic that reflects the losses currently logged by `model/model.py`. The output must make the overall optimization objective, each raw component, and each component's weighted influence understandable for both static and adaptive loss weighting.

## Current Logging Semantics

`LightningWBoson._log_losses` records epoch-aggregated metrics through Lightning's `CSVLogger`:

- Weighted total: `loss` and `val_loss`
- Standardized four-vector Huber: `huber_loss` and `val_huber_loss`
- Higgs mass: `higgs_mass_loss` and `val_higgs_mass_loss`
- Joint kinematic MMD: `kinematic_loss_mmd_loss` and `val_kinematic_loss_mmd_loss`
- W-mass Huber: `w_mass_huber_loss` and `val_w_mass_huber_loss`
- Angular MMD: `angular_loss_mmd_loss` and `val_angular_loss_mmd_loss`
- Delta-MET: `dmet_loss` and `val_dmet_loss`

The total is a weighted sum of enabled raw components. It is not necessarily adaptive, so the plot must call it "Weighted Total Loss," not "Adaptive Total Loss."

Training callbacks in `train/train.py` select checkpoints by minimizing `val_loss`. The notebook must use that exact metric and mode instead of reading the removed `trainer.monitor_metric` and `trainer.monitor_mode` configuration keys.

## Data Preparation

The cell will read the selected run's `metrics.csv` and treat it as a sparse metric table. Each plotted metric will be extracted independently by selecting non-null `epoch` and metric values, sorting by epoch, and resolving duplicate epoch entries by keeping the last logged value. This avoids coupling component availability to rows populated for another metric.

Epoch 0 is valid data and will not be skipped. Train and validation series will be aligned by their explicit epoch values rather than row positions.

The six component definitions and display labels will be declared in one concise metadata list. A component is plotted only when both its train and validation columns contain data. Missing pairs will be named in a message. A missing or empty CSV, absent epoch column, or absent populated `val_loss` will produce a clear diagnostic instead of a plotting exception or an invented fallback monitor.

## Weight Reconstruction

For static runs, each component's configured value from `cfg["parameters"]["loss_weights"]` is used for every epoch.

For adaptive runs, `model.py` logs `loss_weight/<component>` on each training step. The plotting cell will construct one weight per epoch by selecting the last non-null logged weight in that epoch. Missing epoch weights will be forward-filled after seeding from the configured initial value. Components without a logged adaptive series retain their configured weight.

The weighted contribution for a component and epoch is:

```text
raw epoch loss * epoch loss weight
```

The same epoch weight will be associated with train and validation raw metrics. For each stage, the cell will compare the sum of reconstructed contributions against the logged total on common epochs. If they differ beyond a small numerical tolerance, it will warn that logger/hook timing or aggregation prevents exact reconstruction; it will not silently describe the reconstruction as exact.

## Figures

The notebook cell will produce three figures using the existing Matplotlib and ATLAS styling.

### Weighted Total

A single wide panel plots `loss` and `val_loss`. A dashed vertical line marks the epoch with minimum `val_loss`, and a marker identifies that validation value. This panel communicates checkpoint selection and overfitting behavior directly.

### Raw Components

A 2-by-3 grid plots train and validation curves for:

- Standardized four-vector Huber
- Higgs mass
- Joint kinematic MMD
- W-mass Huber
- Angular MMD
- Delta-MET

Each panel has independent y-limits because the raw losses have unrelated scales. Axes use linear scaling and scientific tick formatting. Linear scaling is intentional because finite-sample MMD estimates can be negative, making a log scale invalid.

### Weighted Contributions

Two side-by-side panels show all available weighted component contributions, one for training and one for validation. A component has the same color in both panels, and the panels share their y-scale. Separating stages avoids a crowded twelve-line overlay while preserving direct magnitude comparison.

All figures share epoch as the x-axis, show the full recorded range, and mark the selected best epoch consistently. Legends are figure-level or placed only where needed to avoid repeating identical labels in every component panel.

## Text Summary

The cell will report:

- The metrics file used
- The selected monitor (`val_loss`, minimized)
- The best epoch and its total train/validation values when available
- The final common train/validation epoch and total values
- Raw component values and weighted contributions at the best and final common epochs
- Any unavailable component pairs
- Any contribution reconstruction mismatch

Values will be selected by epoch, not by independently taking the last non-null value of each column. This prevents a summary from combining values from different epochs.

## Verification

The plotting logic will be exercised with temporary synthetic CSV data before the notebook is considered complete.

Static-weight verification will check:

- Epoch 0 is retained.
- The minimum `val_loss` epoch is selected.
- Raw components are aligned by epoch.
- Configured weights produce the expected contributions.
- Reconstructed contributions sum to the logged total.

Adaptive-weight verification will check:

- Repeated step-level weights collapse to the last value for each epoch.
- Missing weights use the configured initial value or forward-filled prior value.
- Train and validation contributions use the intended epoch weight.
- A deliberate total mismatch emits a warning.

Plotting verification will cover one and multiple available components, missing optional component columns, and creation of all three figure layouts without requiring an interactive display.

## Scope

This change is limited to the loss-curve cell in `notebooks/visualize.ipynb`. It will not alter model loss definitions, metric logging, callback monitoring, or add a reusable plotting API. Existing gradient-cosine diagnostics remain separate.
