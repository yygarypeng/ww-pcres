# PCRES Checkpoint Manual

This manual defines the data contract for the accompanying PCRES `last.ckpt`.
The model reconstructs W+ and W- four-vectors from two charged leptons, up to two
jets, and missing transverse momentum (MET).

It was trained on ggF samples from the QE `v6.1` dataset, using the
different-flavor (DF) lepton selection and only the default TCPT selection.

## Checkpoint

| Property | Value |
| --- | --- |
| File | `last.ckpt` |
| Size | ~50.5 MB |
| PyTorch Lightning version | 2.6.1 |
| Input | `(n_events, 18)` `float32` |
| Output | `(n_events, 8)` `float32` |

This is a full Lightning training checkpoint, not a weights-only file. It includes
optimizer and training state and may contain source-machine paths. Load it only
from a trusted source because Python checkpoint deserialization may execute code.

## Input Data

Each row is one event. All momenta and energies are in GeV; four-vectors use
Cartesian order `(px, py, pz, E)`.

Supply the **raw values listed below**. Do not apply logarithms or standardization
before inference. The model contains its own preprocessing and the normalization
statistics stored in this checkpoint.

| Index | Input name | Description | Internal value | Mean | Standard deviation |
| ---: | --- | --- | --- | ---: | ---: |
| 0 | `pos_lep_px` | Positive-lepton x momentum [GeV] | `px` | 0.009317 | 30.055290 |
| 1 | `pos_lep_py` | Positive-lepton y momentum [GeV] | `py` | -0.063729 | 30.025602 |
| 2 | `pos_lep_pz` | Positive-lepton z momentum [GeV] | `pz` | 0.212582 | 74.908890 |
| 3 | `pos_lep_energy` | Positive-lepton energy [GeV] | `log1p(E)` | 3.894312 | 0.703577 |
| 4 | `neg_lep_px` | Negative-lepton x momentum [GeV] | `px` | -0.001047 | 30.036331 |
| 5 | `neg_lep_py` | Negative-lepton y momentum [GeV] | `py` | -0.070254 | 30.021790 |
| 6 | `neg_lep_pz` | Negative-lepton z momentum [GeV] | `pz` | 0.081111 | 75.030495 |
| 7 | `neg_lep_energy` | Negative-lepton energy [GeV] | `log1p(E)` | 3.895314 | 0.704038 |
| 8 | `jet0_px` | First stored jet x momentum [GeV] | `px` | -0.165215 | 72.403885 |
| 9 | `jet0_py` | First stored jet y momentum [GeV] | `py` | 0.300912 | 72.464081 |
| 10 | `jet0_pz` | First stored jet z momentum [GeV] | `pz` | 0.838567 | 261.438446 |
| 11 | `jet0_energy` | First stored jet energy [GeV] | `log1p(E)` | 4.847825 | 0.880103 |
| 12 | `jet1_px` | Second stored jet x momentum [GeV] | `px` | -0.037926 | 38.703732 |
| 13 | `jet1_py` | Second stored jet y momentum [GeV] | `py` | 0.045813 | 38.727287 |
| 14 | `jet1_pz` | Second stored jet z momentum [GeV] | `pz` | 0.374476 | 153.653900 |
| 15 | `jet1_energy` | Second stored jet energy [GeV] | `log1p(E)` | 4.463601 | 0.764432 |
| 16 | `met_px` | Missing transverse x momentum [GeV] | `px` | 0.175328 | 47.747604 |
| 17 | `met_py` | Missing transverse y momentum [GeV] | `py` | -0.127430 | 47.749397 |

The model calculates

```python
standardized_value = (internal_value - mean) / standard_deviation
```

Statistics were fitted on the training split. For each jet slot, padded rows were
excluded when fitting that jet's four statistics.

Jet order follows the source data; the model does not sort jets. Represent an
absent jet as exactly `[0, 0, 0, 0]`. The model masks that raw zero vector. Do not
replace it with standardized values, NaNs, or another sentinel.

All values must be finite. Lepton energies must be positive. Each jet must be
either exact-zero padding or have positive energy. Validate these rules before
inference; the direct PyTorch path does not enforce all of them.

## Target Labels

Targets are needed for training or evaluation, not inference. Their shape is
`(n_events, 10)` with dtype `float32`. They are unnormalized truth values in GeV.

| Index | Target name | Meaning |
| ---: | --- | --- |
| 0 | `w_pos_px` | Truth W+ x momentum [GeV] |
| 1 | `w_pos_py` | Truth W+ y momentum [GeV] |
| 2 | `w_pos_pz` | Truth W+ z momentum [GeV] |
| 3 | `w_pos_energy` | Truth W+ energy [GeV] |
| 4 | `w_neg_px` | Truth W- x momentum [GeV] |
| 5 | `w_neg_py` | Truth W- y momentum [GeV] |
| 6 | `w_neg_pz` | Truth W- z momentum [GeV] |
| 7 | `w_neg_energy` | Truth W- energy [GeV] |
| 8 | `w_pos_mass` | Truth W+ invariant mass [GeV] |
| 9 | `w_neg_mass` | Truth W- invariant mass [GeV] |

W+ is matched to the positive input lepton; W- is matched to the negative input
lepton. Columns 0-7 are the regression references. Columns 8-9 provide truth
masses for training losses and validation but are not direct model outputs.

Valid labels have finite components, positive W energies, nonnegative masses,
timelike W four-vectors, and a timelike combined W+W- four-vector. Each stored mass
must agree numerically with its four-vector invariant mass.

Feature and target names are not embedded in the checkpoint. This manual records
the ordering defined by its training pipeline.

## Model Output

Inference returns a `torch.float32` tensor with shape `(n_events, 8)`:

| Index | Output name | Meaning |
| ---: | --- | --- |
| 0 | `w_pos_px` | Reconstructed W+ x momentum [GeV] |
| 1 | `w_pos_py` | Reconstructed W+ y momentum [GeV] |
| 2 | `w_pos_pz` | Reconstructed W+ z momentum [GeV] |
| 3 | `w_pos_energy` | Reconstructed W+ energy [GeV] |
| 4 | `w_neg_px` | Reconstructed W- x momentum [GeV] |
| 5 | `w_neg_py` | Reconstructed W- y momentum [GeV] |
| 6 | `w_neg_pz` | Reconstructed W- z momentum [GeV] |
| 7 | `w_neg_energy` | Reconstructed W- energy [GeV] |

Compare `outputs` with `targets[:, :8]`. The eight components are not independent:
each W is its charge-matched input lepton plus a reconstructed massless neutrino.
The two neutrino transverse momenta sum to `MET - predicted_dMET`.

## Minimal Inference Example

Use the matching model implementation. Set `checkpoint_path` to the file location:

```python
import torch

from model.model import LightningWBoson


def predict(raw_inputs, checkpoint_path):
    # weights_only=False is required for this trusted Lightning checkpoint.
    model = LightningWBoson.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    ).eval()

    inputs = torch.as_tensor(raw_inputs, dtype=torch.float32)
    if inputs.ndim != 2 or inputs.shape[1] != 18:
        raise ValueError(f"Expected (N, 18), got {tuple(inputs.shape)}")

    with torch.inference_mode():
        return model(inputs)  # (N, 8)
```

Keep this manual and the matching model source with the checkpoint. The checkpoint
stores the architecture and normalization statistics, but not feature names,
target names, jet-ordering policy, or a schema version.
