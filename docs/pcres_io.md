# PCRES IO File

`pcres_io.npz` stores test-set inference results saved by `scripts/save_pcres_io.py`.

## Dataset Split

The `scripts/save_pcres_io.py` run used the original pre-split HDF5 categories:

| Split | Events |
| --- | ---: |
| Train (`ggF_train`) | 263,735 |
| Validation (`ggF_val`) | 75,569 |
| Test (`ggF_test`) | 37,681 |

`pcres_io.npz` contains the test split only, so `inputs`, `outputs`, and
`targets` each have 37,681 rows for that run. The file was written to
`/root/work/ww-pcres/hww_pcres_regressor_nofold/pcres_io.npz`.

## Contents

- `inputs`: input features passed to the model, shape `(n_events, 26)`.
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
| 19 | `dilep_pt` |
| 20 | `met_pt` |
| 21 | `deta_ll` |
| 22 | `dphi_llmet` |
| 23 | `dphi_pos_lep_met` |
| 24 | `dphi_neg_lep_met` |
| 25 | `dphi_ll` |

## Output Columns

| Index | Name |
| --- | --- |
| 0 | `w_pos_px` |
| 1 | `w_pos_py` |
| 2 | `w_pos_pz` |
| 3 | `w_pos_log_energy` |
| 4 | `w_neg_px` |
| 5 | `w_neg_py` |
| 6 | `w_neg_pz` |
| 7 | `w_neg_log_energy` |

## Target Columns

| Index | Name |
| --- | --- |
| 0 | `w_pos_px` |
| 1 | `w_pos_py` |
| 2 | `w_pos_pz` |
| 3 | `w_pos_log_energy` |
| 4 | `w_neg_px` |
| 5 | `w_neg_py` |
| 6 | `w_neg_pz` |
| 7 | `w_neg_log_energy` |
| 8 | `w_pos_mass` |
| 9 | `w_neg_mass` |

Compare `outputs` with `targets[:, :8]`. The final two target columns are truth W masses kept for reference/checking and are not directly predicted by the model.

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
