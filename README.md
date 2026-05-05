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

## Parameter Scan

`scan.py` expands the `scan.parameters` grid in the YAML config, writes one generated config per
point, gives every run a unique `paths.saved_path`, and launches concurrent `train.py` processes.
By default, scan runs use W&B and run test evaluation after training so `test_loss` and each
`test_*_loss` metric are uploaded.

The default scan varies these loss weights:

```yaml
scan:
  strategy: random               # sample combinations from the full grid
  seed: 114                      # reproducible random sampling
  num_samples: 8                 # run 8 total combinations
  max_parallel: 8                # run up to 8 jobs at the same time
  max_retries: 1                 # retry failed jobs once
  gpus: "0"                      # use "0,1,2" to round-robin across GPUs
  parameters:
    parameters.loss_weights.dinu_pt: [0.003, 0.01, 0.03]
    parameters.loss_weights.angular_loss_mmd: [3.0, 10.0, 30.0]
    parameters.loss_weights.higgs_mass: [1.0, 3.0, 5.0]
```

Run 8 sampled combinations on GPU 0 in the background:

```bash
./run_scan.sh
```

`run_scan.sh` stops the previous local `scans/latest` launcher if it is still running, replaces that
workspace, writes the launcher log to `scans/latest/run_scan.log`, writes the PID to
`scans/latest/run_scan.pid`, and uploads W&B runs to the `scan-pcres` project. The launcher resolves
paths relative to the repository, so it can be called from another working directory.

Run in the foreground instead:

```bash
BACKGROUND=0 ./run_scan.sh
```

Generate configs without launching training:

```bash
DRY_RUN=1 ./run_scan.sh
```

Run without W&B:

```bash
WANDB_MODE=disabled ./run_scan.sh
```

Common scan overrides:

```bash
EPOCHS=128 MAX_PARALLEL=4 MAX_RETRIES=1 GPU_IDS=0,1 ./run_scan.sh
```

Run `scan.py` directly:

```bash
python scan.py --config config.yaml --max-parallel 8 --gpus 0 --wandb-project scan-pcres
```

For a quick scouting scan, override epochs without editing the base config:

```bash
python scan.py --config config.yaml --max-parallel 8 --gpus 0 --override parameters.epochs=128
```

Generated configs, logs, and per-run outputs are written under `scans/<timestamp>/`. Use W&B to
filter by flattened config keys such as `parameters.loss_weights.dinu_pt` and sort by `test_loss`
or a specific `test_*_loss` metric.

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
