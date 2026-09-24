# Constrained hyper-parameter sweep

Searches for a training configuration whose W momentum bias is compatible with
zero and whose angular and inter-angular distributions and W momentum resolution
are no worse than those of `configs/untuned_kfold_config.yaml`, all judged
against the spread of repeated untuned runs. If no candidate passes every
constraint and beats the paired untuned baselines, the untuned configuration
stays selected. The sweep never reads the test group, never writes into
`paths.saved_path`, and never overwrites a source configuration file.

## Running

```bash
./sweep/run_sweep.sh              # validate, then run the whole study in the background
tail -f sweep/outputs/sweep.log
./sweep/run_sweep.sh --report     # print selection.json once the study has finished
```

Re-running the same command reuses finished baselines, trials, and confirmation
runs, and skips the search once it has converged. A second driver is refused
while the process in `sweep.pid` (the last launched driver) is alive, and
`sweep.log` is appended to. The driver ignores `SIGINT` and `SIGHUP` and
outlives its terminal; stop it with `kill $(cat sweep/outputs/sweep.pid)`.

Foreground commands that never train: `--validate` parses the base config and
checks the search space, data path, and resume database; `--dry-run` prints
three sampled points; `--report` prints `selection.json`.

A second study needs its own output directory, given twice: `SWEEP_OUTPUT_DIR`
for the log and PID file, `--output-dir` for the study. The study behind
`configs/kfold_config.yaml` ran from the repository root as:

```bash
SWEEP_OUTPUT_DIR=sweep/outputs_v2 ./sweep/run_sweep.sh \
    --space sweep/spaces/constrained_v2.yaml --output-dir sweep/outputs_v2 \
    --study-name constrained-sweep-v2 --budget-hours 10 --top-candidates 1
```

## Monitoring

`sweep.monitor` restarts the driver if it dies before `selection.json` exists,
and gives up after `--max-restarts`, or when two restarts in a row die within
two minutes. It flags a live driver that has written nothing for
`--stall-minutes`.

```bash
( trap '' INT HUP; exec setsid taskset -c 0-9,12-31 python -m sweep.monitor --interval 60 ) \
    < /dev/null &>> sweep/outputs/monitor.out &
```

For another study, give the monitor its `--output-dir` and pass the study's
other launcher arguments with `--launcher-arg=...`, for example
`--launcher-arg=--space --launcher-arg=sweep/spaces/constrained_v2.yaml`.

`sweep.progress` prints a read-only snapshot of a study (best feasible trial,
the constraints that reject trials, and axes the best trials are pressed
against) and writes `figure/progress.png` and `.pdf` in its output directory:

```bash
python -m sweep.progress --output-dir sweep/outputs_v2 \
    --study-name constrained-sweep-v2 --space sweep/spaces/constrained_v2.yaml
```

## What the study does

1. **Baselines.** The untuned configuration is trained with three seeds on
   fold 0. Each distribution and resolution limit is the baseline mean plus
   `--threshold-sigma` (default 2) sample standard deviations; each W momentum
   bias limit is `--threshold-sigma` standard deviations of the signed baseline
   bias, measured against zero. Resuming with different thresholds is refused.
2. **Search.** A TPE study with the limits as trial constraints ranks feasible
   trials by the sum of `bias`, `angular_1d`, and `interangular_1d`. It stops
   when the best feasible score has not improved for `--patience` completed
   trials (checked once `--min-trials` have completed), when
   `--no-feasible-limit` trials complete with none feasible, after three
   consecutive failed trials, or when the time budget runs out.
3. **Confirmation.** The best `--top-candidates` feasible trials are retrained
   with three seeds on `--confirm-folds` (fold 1 by default), next to paired
   untuned baselines that set that fold's limits. A candidate must be feasible
   on every fold and seed.
4. **Selection.** A candidate wins only by beating its paired baselines. The
   result goes to `selected_config.yaml` (the winner or the untouched base
   config) and `selection.json`.

## The fixed ruler

Metrics that no swept setting can change:

- **`bias`**: RMS of |mean residual| / resolution over the six W momentum
  directions `px_sum`, `py_sum`, `px_diff`, `py_diff`, `w0_pz`, `w1_pz`. The W
  energies are reported as `bias_energy` but neither ranked nor constrained.
- **`angular_1d`**: mean total-variation distance over `theta0`, `phi0`,
  `theta1`, `phi1`.
- **`interangular_1d`**: the same over `sum_theta`, `diff_theta`, `sum_phi`,
  `diff_phi`.

Two-dimensional angular joints are reported only. Early stopping watches the sum
of the three. For search trials and confirmation candidates, checkpoint
selection adds the constraint penalty, so the kept epoch is feasible whenever
one was reached; baselines keep their best-sum epoch.

## Search spaces

- `spaces/constrained.yaml` (default): four-vector loss (`l1`, `huber`, `rmse`,
  with a Huber delta for `huber`), the `fourvec_bias` penalty behind an on/off
  gate, the angular MMD, W-mass MMD, and `dmet` weights, the angular MMD kernel
  and bandwidths, learning rate, and weight decay.
- `spaces/constrained_v2.yaml`: adds the `w_fourvec` weight, moves the W-mass
  MMD range up and the learning-rate and weight-decay ranges down, narrows the
  `fourvec_bias`, angular MMD, and `dmet` ranges, and keeps the bias penalty
  on. Trials use the base config's L1 loss and IMQ kernel with bandwidths
  `[0.6, 1.2, 2.4]`.

Neither space varies batch size or model shape.

## Budget

One launch is sized to `--budget-hours` (20 by default). After the baselines,
confirmation time is reserved (confirmation runs × mean baseline wall time
× 1.1) and the rest becomes the search timeout; Optuna finishes the trial in
flight when it expires. If nothing is left, the workflow stops and keeps the
finished baselines. The defaults give nine confirmation runs (one fold, three
seeds, the baseline plus two candidates). The fold-0 baselines took 36.6 s of
wall time per trained epoch, about 37 minutes for all 60 epochs.
`--budget-hours 0` removes the cap.

Sixty epochs is a screen: the untuned `260921` folds reached their best
validation loss at epochs 76 to 129, so retrain a winner at full length and
re-measure its bias before trusting it.

`--full-epochs` retries failed and pruned configurations and runs new trials for
all `--epochs` without pruning or early stopping; the mode persists on resume.

## Resources

- The launcher pins the driver to CPUs `0-9,12-31` and runs one GPU trial at a
  time.
- A short throughput benchmark picks the dataloader worker count. Workers use
  `spawn` without persistence and fall back to the main process if `/dev/shm`
  cannot hold the tensors.
- Fold arrays are cached across trials; only the train and validation groups
  are read.
- Pruning starts after `--startup-trials` trials, and within a trial after
  `--prune-warmup` epochs (default: a third of `--epochs`, at least 5).

## Outputs

```text
sweep/outputs/
  study.db               Optuna study, resumable
  sweep.log, sweep.pid   driver log (appended per launch), last driver PID
  monitor.json, .log     latest monitor poll, one JSON line per poll
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
`configs/`, set `paths.saved_path` to a new directory with `meta` in its name,
and train the copy. `configs/kfold_config.yaml` holds the adopted winner of the
second study.

## Verification without training

```bash
python -m pytest -q tests/test_sweep*.py
bash -n sweep/run_sweep.sh
python -m sweep.optimize --validate
python -m sweep.optimize --dry-run
```

The CUDA tests in `tests/test_sweep_runtime.py` and
`tests/test_sweep_resources.py` are skipped without a GPU.
