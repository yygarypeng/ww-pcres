# Task 3 Report: Fixed Plots And Gradient Table

## Status

DONE

## RED

Command:

```bash
pytest -q tests/test_angular_diagnostics.py
```

The first attempt encountered the repository's intermittent native Python import segmentation
fault before collection and therefore was not accepted as RED evidence. A fresh serial run gave
the expected missing-interface failure:

```text
ImportError: cannot import name 'gradient_diagnostic_rows' from
'scripts.diagnose_angular_degradation'
1 error in 1.71s
```

This established that the plot-saving and gradient-table interfaces did not exist.

## GREEN

Focused command after implementation:

```bash
pytest -q tests/test_angular_diagnostics.py
```

Output summary:

```text
20 passed in 2.10s
```

Combined diagnostics and plotting command:

```bash
pytest -q tests/test_angular_diagnostics.py tests/test_angular_plotting.py
```

Output summary:

```text
37 passed in 2.67s
```

Final verification command:

```bash
pytest -q && ruff check . && black --check scripts/diagnose_angular_degradation.py tests/test_angular_diagnostics.py && git diff --check
```

Output summary:

```text
221 passed, 2 warnings, 42 subtests passed in 4.90s
All checks passed!
2 files would be left unchanged.
```

The two warnings are the existing ONNX tracer warnings.

## Changed Files

- `scripts/diagnose_angular_degradation.py`
- `tests/test_angular_diagnostics.py`
- `.superpowers/sdd/2026-08-21-angular-degradation-diagnosis/task-3-report.md`

## Implementation Details

- Forces the noninteractive Matplotlib `Agg` backend before importing pyplot or plotting helpers.
- Recreates the notebook's six semantic figure families from the fixed 4,096-event checkpoint
  angles and manifest-derived theta/phi bins.
- Passes angular event values in radians to helpers, while supplying bin edges in the helpers'
  documented rad/pi coordinate system.
- Saves deterministic epoch-prefixed PNG names under `<metrics-stem>_plots/` by default and closes
  every figure in a `finally` block. `--plot-dir` provides an explicit override.
- Writes `<metrics-stem>_gradients.csv` by default, with `--gradient-output` as an override.
- Selects the manifest's exact `gradient_indices`, runs each loaded checkpoint in `eval()` mode,
  and reports every raw loss returned by `_compute_batch_losses`.
- Reports raw value, effective weight, `abs(weight) * ||grad(raw loss)||_2`, and cosine against raw
  W+ and W- Huber gradients for every row.
- Adds W+ and W- Huber reference rows at half the effective main Huber weight. Their means preserve
  `main huber = 0.5 * (huber_wplus + huber_wminus)` because both use the model's standardized
  four-vector residual definition.
- Uses `torch.autograd.grad(..., allow_unused=True)` and does not invoke backward or mutate `.grad`.

## Self-Review

- Checked fixed filename ordering, file creation, figure cleanup, persisted-bin routing, radians
  passed as event values, and the six notebook-compatible plot call configurations.
- Hand-checked gradient norms and cosine values with a two-parameter model, including a negative
  effective weight to verify the required absolute weighting.
- Checked that the persisted 512 indices, rather than a fresh sample, reach `_compute_batch_losses`.
- Checked the charge Huber decomposition and the target effective reference weight of 25 when the
  main Huber weight is 50.
- Reviewed output paths, unsafe checkpoint loading posture, trainable-parameter filtering, unused
  gradient handling, and figure cleanup on save failures.
- No blocking findings or safe removal candidates were identified.

## Commit

Implementation commit: `37fefe4` (`Add angular plots and gradient diagnostics`).

This report is committed as a task-metadata follow-up to that implementation commit.

## Concerns

- No production checkpoint/data artifacts were supplied, so actual GPU memory and six-figure render
  time per checkpoint were not measured end to end.
- Checkpoints are loaded separately for fixed-panel inference and 512-event gradient evaluation.
  This avoids retaining the 4,096-event inference model/graph and keeps the existing evaluator
  minimally changed, at the cost of one additional trusted checkpoint load per epoch.
- The first RED attempt hit the same class of intermittent native import segmentation fault noted
  in Task 2; all subsequent focused and full serial verification runs completed normally.
