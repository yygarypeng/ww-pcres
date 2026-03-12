# hww_pcres_regressor

Physics-constrained residual regressor for reconstructing the two $W$ boson four-vectors from dilepton, MET, jet, and angular observables in $H \to WW$ events.

## Repository overview

The main training workflow is now config-driven:

- `train.py`: primary training entry point.
- `config.yaml`: training hyperparameters and filesystem paths.
- `load_data.py`: HDF5 loading and feature construction.
- `data_module.py`: PyTorch Lightning data module.
- `model.py`: Lightning model definition.
- `two_fold_train.py`: older two-fold training script with hardcoded parameters.
- `onnx/`: ONNX export and runtime validation scripts.

## Configuration

Before training, update the paths in `config.yaml`:

```yaml
paths:
  saved_path: "/path/to/output_dir"
  data_path: "/path/to/training_data.h5"
```

Key parameters are also defined in `config.yaml`, including:

- `batch_size`
- `epochs`
- `learning_rate`
- `warmup_epochs`
- `d_model`
- `n_heads`
- `num_blocks`
- `loss_weights`

Important: `train.py` deletes the entire directory specified by `saved_path` before starting a fresh training run. Point `saved_path` to a dedicated output directory, not a shared location.

## Training

Run standard training with:

```bash
python train.py
```

Enable Weights & Biases logging with:

```bash
python train.py --wandb
```

Outputs are written under `saved_path`, including:

- Lightning checkpoints
- CSV logs
- optional Weights & Biases run metadata

The trainer uses GPU automatically when CUDA is available; otherwise it falls back to CPU.

## Two-fold training

`two_fold_train.py` is still available for the older fold-based workflow:

```bash
python two_fold_train.py
python two_fold_train.py --wandb
```

Unlike `train.py`, this script does not read `config.yaml`. Its paths and hyperparameters are defined directly inside the file.

## ONNX export

The ONNX utilities live in `onnx/`:

- `convert_to_onnx.py`: exports a checkpoint to ONNX.
- `onnxruntime_check.py`: compares ONNX Runtime output against the PyTorch checkpoint.

Typical usage:

```bash
cd onnx
python convert_to_onnx.py
python onnxruntime_check.py
```

These scripts currently assume:

- checkpoints exist under `../hww_pcres_regressor_kfold/<fold>/...`
- the selected `fold` matches the exported checkpoint
- the hardcoded input dimension in the script matches the trained model

If you trained with `train.py` or changed the feature set, update the ONNX scripts accordingly before exporting.

## Data expectations

`load_data.py` expects an HDF5 file with grouped particle records, including at least:

- positive and negative lepton kinematics
- MET features
- jet features
- truth $W^+$ and $W^-$ targets

The loader constructs the training feature matrix and target tensor, removes non-finite rows, and computes standardization statistics used by the model.

## Dependencies

The code imports the following Python packages:

- `torch`
- `pytorch_lightning`
- `numpy`
- `pyyaml`
- `h5py`
- `scikit-learn`
- `wandb` (optional, only when using `--wandb`)

## Author

Yuan-Yen Peng (`ypeng@cern.ch`), NTHU group.
