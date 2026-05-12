# hww_pcres_regressor

PyTorch Lightning regressor for reconstructing the two W-boson four-vectors in
H -> WW events. The model predicts neutrino momenta from leptons, MET, jets,
and high-level event features, then builds W four-vectors with physics-aware
layers and losses.

## Layout

- `configs/`: example, local, and sanity YAML configs.
- `data/`: HDF5 loading, feature construction, dataset splitting, and dataloaders.
- `model/`: Lightning module, neural-network layers, and losses.
- `physics/`: kinematics helpers and W-rest-frame boost utilities.
- `train/`: training entry points and launcher.
- `sweep/`: W&B Sweep config, trial wrapper, and launcher.
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

Optional ONNX tools:

```bash
pip install onnx onnxruntime
```

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

Data selection supports all categories, explicit categories, or capped sanity subsets:

```yaml
data:
  categories: null
  val_frac: 0.05
  test_frac: 0.01
  max_events_per_category: null
```

Trainer batch limits are optional and useful for quick smoke tests:

```yaml
trainer:
  limit_train_batches: 2
  limit_val_batches: 1
  limit_test_batches: 0
```

## Sanity Check

Run a one-epoch, two-batch sanity check with a small HDF5 subset:

```bash
python -m train.main --config configs/sanity.yaml
```

This writes to `outputs/sanity/`, which is ignored by git.

## Train

```bash
python -m train.main --config configs/config.yaml
```

With Weights & Biases:

```bash
python -m train.main --config configs/config.yaml --wandb
```

The launcher uses the same entry point:

```bash
./train/run_train.sh
```

Outputs are written under `paths.saved_path`. Training deletes that output directory before a fresh run, after data and model setup have succeeded.

## W&B Sweep

`sweep/sweep.yaml` defines the W&B hyperparameter search. `sweep/sweep.py` runs one sweep trial by loading `configs/config.yaml`, applying sampled W&B parameters, writing generated configs under `wandb_sweep_runs/configs/`, writing model outputs under `wandb_sweep_runs/runs/`, and calling the training workflow.

Create the sweep:

```bash
wandb sweep sweep/sweep.yaml
```

Run the W&B agent command printed by W&B, or start an agent in the background:

```bash
SWEEP_ID=ENTITY/pcres-sweep/SWEEP_ID ./sweep/run_sweep.sh
```

Watch and stop the background agent:

```bash
tail -f sweep.log
kill "$(cat sweep.pid)"
```

## Data

The HDF5 file should contain top-level categories such as `ggF_train`, `ggF_val`, `ggF_test`, or `VBF_train`. Each selected category needs these groups and fields:

- `pos_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `neg_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `met`: `px`, `py`, `pt`, `phi`
- `jets`: `px`, `py`, `pz`, `energy`
- `truth_pos_w`: `px`, `py`, `pz`, `energy`, `m`
- `truth_neg_w`: `px`, `py`, `pz`, `energy`, `m`

The loader builds 26 input features and 10 targets. Non-finite rows are removed. Input standardization is fitted on the training split only.

## ONNX

ONNX export is a supported workflow. See `onnx/README.md`.

## License

BSD-3-Clause. See `LICENSE`.
