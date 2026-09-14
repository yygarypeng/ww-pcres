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

The HDF5 file must contain pre-split top-level categories such as `ggF_train`,
`ggF_val`, and `ggF_test`. Each selected category must contain:

| Group | Fields |
| --- | --- |
| `pos_lep`, `neg_lep` | `px`, `py`, `pz`, `energy` |
| `jets` | `px`, `py`, `pz`, `energy`, each shaped with at least two jet slots |
| `met` | `px`, `py` |
| `truth_pos_w`, `truth_neg_w` | `px`, `py`, `pz`, `energy`, `m` |

`data.categories: null` selects the three default `ggF` categories. A list such
as `[ggF, VBF]` selects those stems for every split. Split-specific
`train_categories`, `val_categories`, and `test_categories` override it;
`max_events_per_category` optionally limits each category.

The loader removes invalid/non-finite kinematics and events with measured
dilepton mass at least 125 GeV before training. Fold assignments described
below use row indices after this filtering.

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

Run both cross-fitting folds sequentially (the default), concurrently on one
selected GPU, or individually:

```bash
python train/two_fold_train.py --config configs/config.yaml
python train/two_fold_train.py --config configs/config.yaml --parallel --gpu 0
python train/two_fold_train.py --config configs/config.yaml --fold 0
python train/two_fold_train.py --config configs/config.yaml --fold 1
```

Fold 0 trains and validates on odd filtered rows and tests on even rows; fold 1
does the reverse. Their outputs go to `paths.saved_path/fold0` and
`paths.saved_path/fold1`.

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
