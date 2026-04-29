# hww_pcres_regressor

PyTorch Lightning regressor for reconstructing the two W-boson four-vectors in
H -> WW events. The model predicts neutrino momenta from leptons, MET, jets,
and high-level event features, then builds W four-vectors with physics-aware
layers and losses.

## Layout

- `train.py`: main training script.
- `config.yaml`: local run configuration.
- `config.example.yaml`: template config for new users.
- `load_data.py`: HDF5 loading and feature construction.
- `data_module.py`: dataset split and dataloaders.
- `model.py`, `layers.py`, `losses.py`, `physics.py`, `torchBoost.py`: model and physics code.
- `plottingtool.py`, `ohbboosting.py`, `visualize.ipynb`: visualization and ROOT-based angular comparison helpers.
- `onnx/`: supported ONNX export and validation workflow.
- `archive/`: old experiments kept for reference, not the main workflow.

## Install

```bash
pip install -r requirements.txt
```

Optional:

```bash
pip install wandb onnx onnxruntime
```

`ohbboosting.py` additionally requires ROOT and is only needed for ROOT-based visualization checks.

## Configure

```bash
cp config.example.yaml config.yaml
```

Edit:

```yaml
paths:
  saved_path: "/path/to/output_dir"
  data_path: "/path/to/training_data.h5"
```

Category selection is intentionally simple. By default, all top-level HDF5 categories are loaded,
concatenated, and then randomly split by `val_frac` and `test_frac`:

```yaml
data:
  categories: null              # or ["ggF_train", "ggF_val", "ggF_test"]
  val_frac: 0.05
  test_frac: 0.01
```

Use a dedicated `saved_path`. Training deletes that output directory before a fresh run, after data
and model setup have succeeded.

## Train

```bash
python train.py --config config.yaml
```

With Weights & Biases:

```bash
python train.py --config config.yaml --wandb
```

Outputs are written under `paths.saved_path`.

## Data

The HDF5 file should contain top-level categories such as `ggF_train`, `ggF_val`, `ggF_test`, or
`VBF_train`. The selected categories are loaded together before splitting. Each selected category
needs these groups and fields:

- `pos_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `neg_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `met`: `px`, `py`, `pt`, `phi`
- `jets`: `px`, `py`, `pz`, `energy`
- `truth_pos_w`: `px`, `py`, `pz`, `energy`, `m`
- `truth_neg_w`: `px`, `py`, `pz`, `energy`, `m`

The loader builds 26 input features and 10 targets. Non-finite rows are removed. Input
standardization is fitted on the training split only.

## ONNX

ONNX export is a supported public workflow. See `onnx/README.md`.

## License

BSD-3-Clause. See `LICENSE`.

## Author

Yuan-Yen Peng (`ypeng@cern.ch`), NTHU group.
