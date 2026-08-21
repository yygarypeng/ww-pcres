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

### Local MMD configuration

The local MMD losses use a product of two independently configured kernel mixtures:

- an output-feature kernel for `alpha_mmd`, joint charge-ordered `mass_mmd`, or `angular_mmd`;
- a condition kernel over standardized `m_ll` and `deta_ll` plus unstandardized `dphi_ll`.

Feature and condition bandwidth lists form a normalized Cartesian-product mixture. Adding another bandwidth therefore changes kernel coverage without mechanically rescaling the loss. The mass loss applies `asinh(m_W^2 / 80.4^2)` and fixed robust statistics fitted on the training truth split. Angular features use `2 * theta / pi - 1` together with `sin(phi)` and `cos(phi)`, producing six features in [-1, 1] while preserving phi periodicity.

Configure the two sides separately under the top-level `mmd` section; see `configs/config.example.yaml` for the supported keys. Unsupported loss names are rejected instead of being ignored.

Set `mmd.local: true` (the default) to multiply each output-feature kernel by the condition kernel over the three high-level features. Set `mmd.local: false` to use global MMD over output features only. Global mode still passes the high-level features into the neural network; it only removes them from MMD conditioning.

Set `parameters.angular_mmd_ramp_epochs` to ramp the angular MMD weight over $R$ epochs. At epoch $e$, its effective weight is $w_{\mathrm{angular}}[1 - \cos(\pi \min(e / R, 1))] / 2$, reaching the configured $w_{\mathrm{angular}}$ at epoch $R$; setting $R$ to zero applies the full configured weight immediately. Other loss weights are unaffected. Early stopping starts checking `val_loss` at epoch $R$, so the ramp-up phase cannot stop training prematurely; with $R$ set to zero it checks from epoch 0 as usual.

### Visualization and inference check

`notebooks/visualize.ipynb` reads the component losses, effective weights, and gradient-cosine columns from the Lightning `metrics.csv`. Its loss panels include the separate alpha and joint-mass MMD histories.

To verify checkpoint reload and export aligned test-set arrays plus quick parity/residual plots:

```bash
python scripts/save_pcres_io.py --config configs/config.yaml
```

See `docs/pcres_io.md` for the output schema.

## Data

The HDF5 file should contain top-level categories such as `ggF_train`, `ggF_val`, `ggF_test`, or `VBF_train`. Each selected category needs these groups and fields:

- `pos_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `neg_lep`: `px`, `py`, `pz`, `energy`, `pt`, `eta`, `phi`
- `met`: `px`, `py`
- `jets`: `px`, `py`, `pz`, `energy`
- `truth_pos_w`: `px`, `py`, `pz`, `energy`, `m`
- `truth_neg_w`: `px`, `py`, `pz`, `energy`, `m`

The loader's public input contains 21 raw columns, ordered as positive-lepton `(px, py, pz, E)`, negative-lepton `(px, py, pz, E)`, jet 0 `(px, py, pz, E)`, jet 1 `(px, py, pz, E)`, MET `(px, py)`, then `m_ll`, `deta_ll`, and `dphi_ll`. Lepton energies must be finite and strictly positive. Each missing jet must be an exact-zero four-vector, while each present jet must have finite, strictly positive energy. Negative or non-finite jet energies and nonzero jet four-vectors with exactly zero energy are invalid.

Only the neural aggregation path converts these raw inputs to 21 features, in this order: positive-lepton `(px, py, pz, log1p(E))`, negative-lepton `(px, py, pz, log1p(E))`, jet 0 `(px, py, pz, log1p(E))`, jet 1 `(px, py, pz, log1p(E))`, MET `(px, py)`, `m_ll`, `deta_ll`, and `dphi_ll`. Non-angular statistics are fitted on the training split only; each jet slot uses only events where that raw jet is present, with mean zero and scale one if no training event contains the slot. The `dphi_ll` feature keeps fixed mean zero and scale one. Padded jets remain in event arrays and are excluded by attention masks.

The MMD condition path is separate from neural aggregation and constructs three condition features directly from raw inputs: `m_ll`, `deta_ll`, and `dphi_ll`. Only `m_ll` and `deta_ll` are standardized; `dphi_ll` remains unchanged by using mean zero and scale one.

The loader also builds 10 targets. Target columns contain each W boson's `(px, py, pz, energy)` in GeV followed by the two truth W masses. Each truth W must be finite and timelike, have a nonnegative stored mass, and agree with $E^2-|p|^2$ within `1e-6 + 1e-6` times the sum-of-squares scale; the combined W pair must also be timelike.

## ONNX

ONNX export is a supported workflow. See `onnx/README.md`.

## License

BSD-3-Clause. See `LICENSE`.
