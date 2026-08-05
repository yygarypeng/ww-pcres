# Combined Angular Diagnostics Design

## Goal

Add one self-contained code cell to `notebooks/visualize.ipynb` that combines the existing angular diagnostic plots into slide-friendly figures. The user will execute the cell after the cells that define the prediction and truth arrays.

## Scope

The cell will generate six figures, separating one-dimensional distributions from two-dimensional correlations for each observable group:

1. Theta and phi sums and differences, 1D.
2. Theta and phi sums and differences, 2D.
3. Mixed theta-phi sums, 1D.
4. Mixed theta-phi sums, 2D.
5. Mixed theta-phi differences, 1D.
6. Mixed theta-phi differences, 2D.

Each figure uses a balanced 2x2 grid with one observable per panel. The 1D figures overlay the predicted and true distributions; the 2D figures show predicted-versus-true histograms. Pred/True ratio panels are intentionally omitted to keep labels and data readable on slides.

## Implementation

The notebook cell will define separate local 1D and 2D plotting helpers rather than modify `notebooks/plottingtool.py`. Both helpers will accept the same list of observable specifications containing the predicted values, true values, display label, bin edges, and two-dimensional histogram scaling.

The helpers will:

- Divide angular values by `np.pi` before plotting.
- Draw prediction as a red step histogram and truth as a blue step histogram in each 1D panel.
- Draw the two-dimensional histograms with the existing `viridis` color map in a separate figure.
- Draw a diagonal prediction-equals-truth reference line in each 2D panel.
- Use logarithmic normalization for the theta and mixed theta-phi panels, while preserving linear normalization for the phi sum and difference panels.
- Put one shared legend at the top of each 1D figure. The theta/phi 2D figure will use per-panel colorbars because its panels use different normalization. Each mixed 2D figure will use one shared logarithmic colorbar because all four panels use the same normalization.
- Use one figure-level x label and one figure-level y label instead of repeating axis labels in every panel. Label all angular axes in `rad/$\pi$`.
- Share x and y scales in the mixed theta-phi figures and hide their interior tick labels. Do not share scales in the theta/phi figures because theta sum uses `[0, 2]` while the other observables use `[-1, 1]`; retain each panel's tick values there.
- Include `RMSE = x.xx` in every 2D panel title. Compute RMSE after dividing prediction and truth by `np.pi`, so it uses the plotted `rad/$\pi$` unit, and format it to exactly two decimal places.
- Use constrained layout and compact slide-oriented figure sizes, with readable panel titles, ticks, legends, colorbars, and spacing at presentation-export size.
- Return both figures and their axes for each observable group while displaying them in the notebook.

## Observable Groups

The first pair of figures contains theta sum, theta difference, phi sum, and phi difference. The second pair contains the four charge combinations of mixed theta-phi sums. The third pair contains the corresponding four mixed theta-phi differences. Every figure arranges its four observables in a 2x2 grid.

The cell will preserve the bin ranges already used in the notebook:

- Theta sum: `[0, 2]`.
- Theta difference: `[-1, 1]`.
- Phi sum and difference: `[-1, 1]`.
- Mixed theta-phi sums and differences: `[0, 1]`.

## Error Handling

Shared input preparation will validate that every observable specification has matching prediction and truth shapes. Non-finite value pairs will be removed before histogramming. An observable with no finite pairs will raise a clear `ValueError` rather than render an empty or misleading panel.

## Verification

Because the new block depends on arrays created during the interactive notebook workflow, verification will check notebook syntax and structure without executing the full data-dependent notebook, then exercise the isolated cell with synthetic arrays using a noninteractive backend. The synthetic check will confirm six 2x2 figures, figure-level labels, shared mixed-group axes, hidden interior mixed-group tick labels, and two-decimal normalized RMSE titles. The cell should also be manually run after the prerequisite cells to confirm that the compact figures remain readable with production distributions.
