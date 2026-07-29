# Physics-constrained Residual Regressor (ww-pcres)

The physics goal of this neural network (NN) is to construct spin-correlation-sensitive variable components $\theta^\ast_{\ell^\pm}$, $\phi^\ast_{\ell^\pm}$ in the associated $W$ boson rest frames; specifically, we aim to reconstruct the spin-correlation parameters via the $H \to WW^\ast \to \ell\nu\ell\nu$ decay channel.
To better `correlation' between truth labels and predictions, ie, event-wise errors, a deterministic model is implemented; however, a vanilla DNN might collapse/average out the physical patterns.
Therefore, we try to design a structure that can be informed by physics constraints, such as $m_{W^\pm}$ spectrum and Higgs mass, etc.
As for details, please see our paper[will link in here soon]!

Technically, this is a PyTorch Lightning regressor for reconstructing the two $W$ bosons' four-vectors in $H \to WW^\ast \to \ell\nu\ell\nu$. 
The model predicts neutrino momenta from leptons, MET, and jets event features, then builds W four-vectors with physics-aware customized layers and losses.

## Layout

- `configs/`: example, local, and sanity YAML configs.
- `data/`: HDF5 loading, feature construction, dataset splitting, and dataloaders.
- `model/`: Lightning module, neural-network layers, and losses.
- `physics/`: kinematics helpers and W-rest-frame boost utilities.
- `train/`: training entry points and launcher.
- `scripts/`: standalone utility scripts.
- `notebooks/`: exploratory notebooks and notebook plotting helpers.
- `onnx/`: ONNX export and validation utilities.
- `docs/`: notes for generated files and project workflows.
- `archive/`: old experiments kept for reference only.

Generated outputs belong under `outputs/` or W&B/Lightning output folders and are ignored by git.

## Install

```bash
pip install -r requirements.txt
```

`onnx` and `onnxruntime` are installed by `requirements.txt` for the supported ONNX workflow.

`physics/ohbboosting.py` additionally requires ROOT and is only needed for ROOT-based visualization checks.

## Configure

Local config is intentionally ignored by git:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Edit:

```yaml
paths:
  saved_path: "outputs/run"
  data_path: "/path/to/training_data.h5"
```

Data selection uses named, pre-split HDF5 categories:

```yaml
data:
  categories: null
  max_events_per_category: null
```

`categories: null` selects `ggF_train`, `ggF_val`, and `ggF_test` by default. Use category stems such as `categories: [ggF, VBF]` to select the corresponding train, validation, and test categories for each stem. For direct control, set `train_categories`, `val_categories`, and `test_categories` to lists of full HDF5 category names instead; these split-specific settings take precedence over `categories`.

`max_events_per_category` optionally caps the number of events read from each selected category in each split. `null` loads every event.

## Train

```bash
python train/train.py --config configs/config.yaml
```

With Weights & Biases:

```bash
python train/train.py --config configs/config.yaml --wandb
```

The launcher uses the same entry point:

```bash
./train/run_train.sh
```

Outputs are written under `paths.saved_path`. Training deletes that output directory before a fresh run, after data and model setup have succeeded.

Adaptive loss weights, when enabled, are updated once at the end of each training epoch using the first training batch from that epoch. The cosine metrics are logged as `grad_cos/{loss}__total`.

## Data

The HDF5 file should contain top-level categories such as `ggF_train`, `ggF_val`, `ggF_test`, or `VBF_train`. Each selected category needs these groups and fields:

- `pos_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `neg_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `met`: `px`, `py`, `pt`, `phi`
- `jets`: `px`, `py`, `pz`, `energy`
- `truth_pos_w`: `px`, `py`, `pz`, `energy`, `m`
- `truth_neg_w`: `px`, `py`, `pz`, `energy`, `m`

The loader builds 22 input features and 10 targets. Target columns contain each W boson's `(px, py, pz, energy)` in GeV followed by the two truth W masses. Each truth W must be finite and timelike, have a nonnegative stored mass, and agree with $E^2-|p|^2$ within `1e-6 + 1e-6` times the sum-of-squares scale; the combined W pair must also be timelike. Input standardization is fitted on the training split only.

## ONNX

ONNX export is a supported workflow. See `onnx/README.md`.

## License

BSD-3-Clause. See `LICENSE`.
