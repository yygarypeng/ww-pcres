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

Those three groups are the whole splitting policy and are read as-is. The
`data:` config section accepts one optional key, `max_events_per_category`,
which caps the events read from each group; any other key there is rejected
rather than silently ignored. Cross-fitting on top of this split is provided by
`train/k_fold_train.py` (see Training below).

The loader removes invalid/non-finite kinematics and events with measured
dilepton mass at least 125 GeV before training. Fold assignments described below
are unaffected by that filtering, because they are keyed on `eventNumber`.

## Training

Run one model:

```bash
python train/train.py --config configs/config.yaml
```

Add `--wandb` for Weights & Biases logging or `--gpu {0,1}` to select a physical
GPU. The launcher requires Linux `taskset`, hard-codes CPU affinity
`0-9,12-15`, backgrounds the process, and redirects stdout/stderr to `record.log`:

```bash
./train/run_train.sh
```

Run every cross-fitting fold, or just one of them:

```bash
./train/run_k_fold_train.sh
python train/k_fold_train.py --config configs/kfold_config.yaml
python train/k_fold_train.py --config configs/kfold_config.yaml --fold 0
```

Folds always run one after another: a single fold already holds the GPU at 100%
and peaks near 9.5 GiB of this card's 16 GiB, so there is nothing for a
concurrent fold to reclaim and no room to hold it.

`parameters.folds` sets the fold count; the config ships with 8 folds. Training
concatenates the pre-split train and validation groups and cuts them into N
residue classes of the HWWFrames `eventNumber`: fold `i` holds out the events
with `eventNumber % N == i` and trains on the rest, while the complete test group
remains held out and is scored by every fold. Downstream code therefore selects
the model for an event with the same `eventNumber % N` it was held out by, and
fold membership survives regenerating or refiltering the HDF5. The residue
classes are not exactly equal in size, because HWWFrames already splits
train/validation/test by `eventNumber % 100` and 8 does not divide 100: on the
`v6.1` ggF merged file the validation folds range from 153.6k to 161.2k rows,
against 1.099M to 1.107M training rows and the same 140,565 test rows for every
fold.

Fold outputs go to `paths.saved_path/fold<i>`.

Training writes checkpoints and Lightning CSV logs below `paths.saved_path`.
Every fresh run deletes its entire target output directory after data and model
setup, so use a new path to preserve existing runs. This applies separately to
each fold directory.

## Analysis And Export

- `notebooks/visualize.ipynb` visualizes Lightning metrics.
- `python scripts/save_pcres_io.py --config configs/config.yaml` reloads a
  checkpoint, exports aligned test arrays, and creates parity/residual plots;
  see `docs/pcres_io.md` for its output schema.
- `python scripts/evaluate_higgs_constraint.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  evaluates checkpoint Higgs-mass constraints.
- `python scripts/evaluate_mmd_bandwidths.py --checkpoint-dir outputs/run --data-path /path/to/training_data.h5`
  evaluates each configured MMD feature bandwidth.
- `onnx/README.md` documents checkpoint export and PyTorch/ONNX Runtime parity
  checks. Run its commands from `onnx/`.
- `physics/ohbboosting.py` provides optional ROOT-based visualization checks and
  requires ROOT.

## Verification

```bash
ruff check .
pytest
```

## License

BSD-3-Clause. See `LICENSE`.
