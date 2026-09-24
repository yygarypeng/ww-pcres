# Physics-constrained Residual Regressor

PyTorch Lightning regressor for reconstructing the two W-boson four-vectors in
`H -> WW* -> lnu lnu` from leptons, missing transverse momentum, and jets.

## Setup

```bash
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
```

Set `paths.data_path` to the input HDF5 file and `paths.saved_path` to a new run
directory. Adjust the remaining training, data-selection, loss, and MMD settings
in `configs/config.yaml` as needed.

## Data

The HDF5 file must contain the three pre-split top-level groups `ggF_train`,
`ggF_val`, and `ggF_test`. Each group must contain:

| Group | Fields |
| --- | --- |
| `pos_lep`, `neg_lep` | `px`, `py`, `pz`, `energy` |
| `jets` | `px`, `py`, `pz`, `energy`, each shaped with at least two jet slots |
| `met` | `px`, `py` |
| `truth_pos_w`, `truth_neg_w` | `px`, `py`, `pz`, `energy`, `m` |
| `event` | `eventNumber`, read only by `train/k_fold_train.py` |

The groups are read as-is. The optional `data.max_events_per_category` caps the
events read from each group; any other `data:` key is rejected. The loader drops
events with invalid or non-finite kinematics and events with a dilepton mass of
125 GeV or more.

## Training

Run one model:

```bash
python train/train.py --config configs/config.yaml
```

Add `--wandb` for Weights & Biases logging or `--gpu {0,1}` to pick a GPU.
`./train/run_train.sh` runs the same command in the background with CPU affinity
`0-9,12-15` (Linux `taskset`) and writes its output to `record.log`.

Cross-fitting trains one model per fold:

```bash
./train/run_k_fold_train.sh              # every fold of configs/kfold_config.yaml
./train/run_k_fold_train.sh --fold 0     # one fold
python train/k_fold_train.py --config configs/untuned_kfold_config.yaml --fold 0
```

`configs/kfold_config.yaml` is the sweep-tuned configuration and
`configs/untuned_kfold_config.yaml` the untuned baseline. The launcher runs in the
background with W&B logging and appends to `record_k_fold.log`. It checks the
config, data file, and GPU first, and refuses to overwrite an existing fold
unless given `--overwrite`. Folds run one after another, because a single fold
saturates the GPU.

`parameters.folds` sets the fold count (8 in the shipped configs). Training pools
the train and validation groups, and fold `i` holds out the events with
`eventNumber % folds == i`; every fold is scored on the full test group.
Consumers therefore select the model for an event by `eventNumber % folds`, and
fold membership survives regenerating or refiltering the HDF5. Folds differ
slightly in size, because HWWFrames already splits by `eventNumber % 100`.

Each run writes checkpoints and Lightning CSV logs to `paths.saved_path`
(`paths.saved_path/fold<i>` for folds), deleting that directory first, so use a
new path to keep earlier runs. Keep `meta` in the directory name (for example
`fold_meta_ggF_v3`) so run outputs stay out of Git.

## Hyper-parameter Sweeps

`sweep/` runs a constrained Optuna search on one fold, scoring each trial on a
fixed ruler: W momentum bias in units of resolution, and total-variation
distances on the lepton decay angles and their sums and differences.

```bash
./sweep/run_sweep.sh             # validate, then run in the background
./sweep/run_sweep.sh --report    # print selection.json once finished
```

Outputs land under `sweep/outputs/`; nothing in `sweep/` writes to
`paths.saved_path` or reads the test group. See `sweep/README.md`.

## Analysis And Export

- `notebooks/visualize.ipynb` plots a run's metrics and predictions and saves each
  figure as a PDF under `paths.saved_path/figure/`.
- `python scripts/save_pcres_io.py --config configs/config.yaml` reloads a
  checkpoint, exports aligned test arrays, and creates parity/residual plots;
  see `docs/pcres_io.md` for its output schema.
- `python scripts/evaluate_higgs_constraint.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  evaluates checkpoint Higgs-mass constraints.
- `python scripts/evaluate_mmd_bandwidths.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  evaluates each configured MMD feature bandwidth.
- `onnx/README.md` documents checkpoint export and PyTorch/ONNX Runtime parity
  checks. Run its commands from `onnx/`.
- `docs/checkpoint_manual.md` is the consumer-facing contract for the exported
  models: input columns, fold selection by `eventNumber`, and a minimal ONNX
  Runtime client.
- `physics/ohbboosting.py` provides optional ROOT-based visualization checks and
  requires ROOT.

## Verification

```bash
ruff check .
ruff format --check .
pytest
```

## License

BSD-3-Clause. See `LICENSE`.
