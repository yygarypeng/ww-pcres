# Constrained hyper-parameter sweep

Finds a training configuration with an unbiased W momentum whose angular and
inter-angular distributions are no worse than
`configs/untuned_kfold_config.yaml`. If no tuned configuration passes every
constraint, the untuned configuration stays selected.

The sweep never reads the test group, never writes into `paths.saved_path`, and
never overwrites a source configuration file.

## Quick start

```bash
./sweep/run_sweep.sh              # validate, then run the whole study in the background
tail -f sweep/outputs/sweep.log
./sweep/run_sweep.sh --report     # print selection.json once the study has finished
```

Re-running the same command resumes finished baselines and trials. A second
driver is refused while the PID in `sweep.pid` is alive, and `sweep.log` is
appended to, so a crash traceback survives the resume. Stop a run with
`kill $(cat sweep/outputs/sweep.pid)`; the driver ignores `SIGINT` and `SIGHUP`
and outlives its terminal.

Foreground commands that never train:

- `--validate` checks the base config, search space, data path, and resume database.
- `--dry-run` prints sampled points from the search space.
- `--report` prints `selection.json`.

A second study needs its own output directory, and the launcher needs it twice:
`SWEEP_OUTPUT_DIR` for the log and PID file, `--output-dir` for the study. The
second study, whose winner is `configs/kfold_config.yaml`, ran as:

```bash
SWEEP_OUTPUT_DIR=sweep/outputs_v2 ./sweep/run_sweep.sh \
    --space sweep/spaces/constrained_v2.yaml \
    --output-dir sweep/outputs_v2 --study-name constrained-sweep-v2
```

## Monitoring

`sweep.monitor` restarts the driver if it dies before `selection.json` exists.
It gives up after `--max-restarts`, or when two restarts in a row die within two
minutes, and flags a live driver that has written nothing for `--stall-minutes`.

```bash
( trap '' INT HUP; exec setsid python -m sweep.monitor --interval 60 ) \
    < /dev/null &>> sweep/outputs/monitor.out &
```

For a non-default study, pass its output directory with `--output-dir` and its
launcher arguments with `--launcher-arg=...`.

`sweep.progress` prints a read-only snapshot of a running study (best feasible
trial, which constraints reject trials, and axes the best trials are pressed
against) and writes `figure/progress.png`:

```bash
python -m sweep.progress --output-dir sweep/outputs_v2 \
    --study-name constrained-sweep-v2 --space sweep/spaces/constrained_v2.yaml
```

## What the study does

1. **Baseline calibration.** Trains the untuned configuration with three seeds
   on fold 0. Each distribution and resolution limit is the baseline mean plus
   `--threshold-sigma` sample standard deviations. Each W momentum bias limit is
   `--threshold-sigma` standard deviations of the signed baseline bias, measured
   against zero, so a biased baseline fails its own bias limits by design.
   Resuming with different thresholds is refused; use a new `--study-name`.
2. **Constrained search.** A TPE study with the limits as trial constraints.
   Feasible trials are ranked by the sum of `bias`, `angular_1d`, and
   `interangular_1d`. The search stops on convergence, after three consecutive
   failed trials, or when the time budget runs out.
3. **Confirmation.** The top feasible candidates are retrained with three seeds
   on `--confirm-folds` (fold 1 by default), next to paired untuned baselines
   that set that fold's limits. A candidate must be feasible on every fold and seed.
4. **Selection.** A candidate wins only by beating its paired confirmation
   baselines. The result is written to `selected_config.yaml` (the winner or the
   untouched base config) and `selection.json`.

## The fixed ruler

Every model is scored on metrics that do not depend on its training loss, MMD
kernel, or bandwidths:

- **`bias`**: RMS of |mean residual| / resolution over the six W momentum
  directions `px_sum`, `py_sum`, `px_diff`, `py_diff`, `w0_pz`, `w1_pz`. The W
  energies are reported as `bias_energy` but neither ranked nor constrained,
  because the median-seeking L1 loss sits about 22 GeV low on them by construction.
- **`angular_1d`**: mean total-variation distance over `theta0`, `phi0`,
  `theta1`, `phi1`.
- **`interangular_1d`**: mean total-variation distance over `sum_theta`,
  `diff_theta`, `sum_phi`, `diff_phi`.

Two-dimensional angular joints are reported only. Early stopping watches the
composite score; checkpoint selection adds the constraint penalty, so the kept
epoch is feasible whenever training reached one.

## Search spaces

- `spaces/constrained.yaml` (default): four-vector loss shape (`l1`, `huber`,
  `rmse`), the `fourvec_bias` penalty behind an on/off gate, angular MMD, W-mass
  MMD, and `dmet` weights, angular MMD kernel and bandwidths, learning rate, and
  weight decay.
- `spaces/constrained_v2.yaml`: the second study, which widens the axes the first
  study's best trials were pressed against, freezes the settled ones, and adds
  the `w_fourvec` weight.

Batch size and model shape stay fixed: batch size changes the MMD estimator,
and the model shape is part of the exported ONNX contract.

## Budget

One launch is sized to `--budget-hours` (20 by default). Confirmation time is
reserved from the measured baseline wall clock, and the rest becomes the search
timeout; if the budget cannot cover confirmation the workflow stops and keeps
the finished baselines. At 36.5 s/epoch the defaults give three baselines
(1.8 h), about 20 search trials of 60 epochs (12 h), and nine confirmation runs
(5.5 h). `--budget-hours 0` removes the cap.

Sixty epochs is a screen: production runs peak much later, so retrain a winner
at full length and re-measure its bias before trusting it.

`--full-epochs` retries failed and pruned configurations and runs new trials for
all `--epochs` without pruning or early stopping. The mode is saved in the study
and persists on resume.

## Resource use

- One GPU trial at a time on CPUs `0-9,12-31`, skipping the defective `10-11`.
- A short throughput benchmark picks the dataloader worker count. Workers use
  `spawn` without persistence to avoid inheriting CUDA state and leaking pipes,
  and fall back to the main process if `/dev/shm` cannot hold the tensors.
- Fold arrays are cached across trials, and only the train and validation
  groups are opened. Pruning waits for a warmup of a third of `--epochs`.

## Outputs

```text
sweep/outputs/
  study.db               Optuna study, resumable
  sweep.log, sweep.pid   driver log (appended per launch) and live PID
  monitor.json, .log     latest monitor poll, and one JSON line per poll
  resources.json         measured dataloader rows/s per worker count
  thresholds.json        baseline-derived feasibility limits
  baseline/              search-fold baselines, one directory per seed
  trials/                per-trial config, checkpoint, and metrics.json
  confirmation/          paired baseline and candidate runs per fold
  selected_config.yaml   the winner, or the untouched base config
  selection.json         status, scores, and fingerprints
```

## Using the winner

`selected_config.yaml` inherits `paths.saved_path` from the base config, and
training deletes `<saved_path>/fold<i>` before each fold. Copy it into
`configs/`, set `paths.saved_path` to a new directory with `meta` in its name
(for example `fold_meta_ggF_v3`), and train that copy with
`train/k_fold_train.py`.

`configs/kfold_config.yaml`, which `train/run_k_fold_train.sh` trains, holds
the adopted winner of the second study.

## Verification without training

```bash
python -m pytest -q tests/test_sweep*.py
bash -n sweep/run_sweep.sh
python -m sweep.optimize --validate
python -m sweep.optimize --dry-run
```

The CUDA runtime tests in `tests/test_sweep_runtime.py` and
`tests/test_sweep_resources.py` are skipped without a GPU.
