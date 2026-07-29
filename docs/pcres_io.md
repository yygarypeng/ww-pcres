# PCRES IO File

`pcres_io.npz` stores test-set inference results saved by `scripts/save_pcres_io.py`.

## Workflow

Run `python scripts/save_pcres_io.py --config <config-path>` to evaluate the latest checkpoint under `paths.saved_path` on the configured pre-split test categories. The script writes `pcres_io.npz` and diagnostic plots under `paths.saved_path`. The archive contains the test split only; its row count depends on the selected categories, available events, and `max_events_per_category` setting.

## Contents

- `inputs`: input features passed to the model, shape `(n_events, 22)`.
- `outputs`: model predictions, shape `(n_events, 8)`.
- `targets`: true target values, shape `(n_events, 10)`.
- `checkpoint`: checkpoint path used for inference.
- `config`: config file path used to build the datamodule/model setup.

Each row corresponds to one test event, so `inputs[i]`, `outputs[i]`, and `targets[i]` refer to the same sample.

## Input Columns

| Index | Name |
| --- | --- |
| 0 | `pos_lep_px` |
| 1 | `pos_lep_py` |
| 2 | `pos_lep_pz` |
| 3 | `pos_lep_energy` |
| 4 | `neg_lep_px` |
| 5 | `neg_lep_py` |
| 6 | `neg_lep_pz` |
| 7 | `neg_lep_energy` |
| 8 | `jet0_px` |
| 9 | `jet0_py` |
| 10 | `jet0_pz` |
| 11 | `jet0_energy` |
| 12 | `jet1_px` |
| 13 | `jet1_py` |
| 14 | `jet1_pz` |
| 15 | `jet1_energy` |
| 16 | `met_px` |
| 17 | `met_py` |
| 18 | `m_ll` |
| 19 | `deta_ll` |
| 20 | `dphi_ll` |
| 21 | `dphi_llmet` |

## Output Columns

| Index | Name |
| --- | --- |
| 0 | `w_pos_px` |
| 1 | `w_pos_py` |
| 2 | `w_pos_pz` |
| 3 | `w_pos_energy` [GeV] |
| 4 | `w_neg_px` |
| 5 | `w_neg_py` |
| 6 | `w_neg_pz` |
| 7 | `w_neg_energy` [GeV] |

## Target Columns

| Index | Name |
| --- | --- |
| 0 | `w_pos_px` |
| 1 | `w_pos_py` |
| 2 | `w_pos_pz` |
| 3 | `w_pos_energy` [GeV] |
| 4 | `w_neg_px` |
| 5 | `w_neg_py` |
| 6 | `w_neg_pz` |
| 7 | `w_neg_energy` [GeV] |
| 8 | `w_pos_mass` |
| 9 | `w_neg_mass` |

Compare `outputs` with `targets[:, :8]`. Momentum, energy, and mass values are in GeV. The final two target columns are truth W masses kept for reference/checking and are not directly predicted by the model.

## Reading The File

```python
import numpy as np

data = np.load("pcres_io.npz")

inputs = data["inputs"]
outputs = data["outputs"]
targets = data["targets"]

print(inputs.shape, outputs.shape, targets.shape)
print(data["checkpoint"])
print(data["config"])
```

## Check Plots

`scripts/save_pcres_io.py` also writes two quick diagnostic figures next to the `.npz` file:

- `pcres_io_parity.png`: model outputs vs targets.
- `pcres_io_residuals.png`: output minus target distributions.
