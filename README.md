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
| `event` | `eventNumber`, read only by `train/k_fold_train.py` and `sweep/` |

The groups are read as-is. The optional `data.max_events_per_category` caps the
events read from each group; any other `data:` key is rejected. The loader drops
rows with non-finite values, invalid input energies, or invalid truth W
kinematics, and rows with a dilepton mass of 125 GeV or more.

## Training

Run one model:

```bash
python train/train.py --config configs/config.yaml
```

`--wandb` enables Weights & Biases logging and `--gpu {0,1}` picks a GPU.
`./train/run_train.sh` runs the same command in the background on CPUs
`0-9,12-15` and writes `record.log`.

Cross-fitting trains one model per fold:

```bash
./train/run_k_fold_train.sh              # every fold of configs/kfold_config.yaml
./train/run_k_fold_train.sh --fold 0     # one fold
```

The launcher runs in the background with W&B logging, appends to
`record_k_fold.log`, checks the config, data file, and GPU first, and refuses to
retrain an existing fold unless given `--overwrite`. `train/k_fold_train.py`
takes the same `--config` and `--fold` arguments without those checks. Folds run
one after another. `configs/kfold_config.yaml` is the sweep-tuned configuration
and `configs/untuned_kfold_config.yaml` the untuned baseline.

`parameters.folds` sets the fold count: 8 in both k-fold configs, 2 when absent.
Training pools the train and validation groups; fold `i` validates on the events
with `eventNumber % folds == i` and trains on the rest, and every fold is tested
on the same test group. Consumers select the model for an event by
`eventNumber % folds`, and fold membership survives regenerating or refiltering
the HDF5.

In the v6.1 ggF file, `eventNumber % 100` is 0-9 for test, 10-29 for validation,
and 30-99 for training. Since 4 divides both 8 and 100, `eventNumber % 8` fixes
`eventNumber % 4`, and the 90 pooled residues split 22/22/23/23 across it. Folds
2, 3, 6, and 7 therefore validate on about 4.5% more events (160.7k to 161.2k,
against 153.6k to 154.7k), and every fold tests on 140,565.

Each run writes checkpoints and Lightning CSV logs to `paths.saved_path`
(`paths.saved_path/fold<i>` per fold) and deletes that directory first, so use a
new path to keep earlier runs. Keep `meta` in the directory name (for example
`fold_meta_ggF_v3`) so outputs stay out of Git.

## Hyper-parameter Sweeps

`sweep/` runs a constrained Optuna search on one fold, scoring each trial on a
fixed ruler: W momentum bias in units of resolution, and total-variation
distances on the lepton decay angles and their sums and differences.

```bash
./sweep/run_sweep.sh             # validate, then run in the background
./sweep/run_sweep.sh --report    # print selection.json once finished
```

Outputs land under `sweep/outputs/` by default; nothing in `sweep/` writes to
`paths.saved_path` or reads the test group. See `sweep/README.md`.

## Analysis And Export

- `notebooks/visualize.ipynb` plots a run's metrics and predictions and saves each
  figure as a PDF under `paths.saved_path/figure/`.
- `python scripts/save_pcres_io.py --config configs/config.yaml` runs the newest
  checkpoint under `paths.saved_path` (by modification time, usually `last.ckpt`)
  on the test group, saves inputs, outputs, and targets to `pcres_io.npz`, and
  writes parity and residual plots; `docs/pcres_io.md` describes the file.
- `python scripts/evaluate_higgs_constraint.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  reports Higgs-mass metrics for each distinct checkpoint under
  `--checkpoint-dir`, on the `ggF_val` group unless `--split` says otherwise.
- `python scripts/evaluate_mmd_bandwidths.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  reports, for the same checkpoints and split, the MMD at each configured
  bandwidth of the alpha, mass, and angular features.
- `onnx/README.md` covers ONNX export and the PyTorch/ONNX Runtime parity check.
- `docs/checkpoint_manual.md` documents the published `260921` exports for
  consumers: input columns, fold selection by `eventNumber`, and a minimal ONNX
  Runtime client.
- `python -m physics.torchBoost` checks the PyTorch rest-frame decay angles
  against the ROOT reference in `physics/ohbboosting.py` when ROOT is installed,
  and writes `torchboost_theta_phi_compare.png` to the current directory.

## Verification

```bash
ruff check .
ruff format --check .
pytest
```

## License

BSD-3-Clause. See `LICENSE`.
