The configured model uses a 216-dimensional representation, 8 attention heads,
and 3 attention blocks with 2-head trunk outputs.  It has ~2.83 M trainable
parameters on the QE `v6.1` dataset with only DF and TCPT default selections.

Model artifacts: hww_regressor_fold0.onnx ... hww_regressor_fold7.onnx  
Source run: 8-fold cross-fitting under `fold_meta/fold{0..7}`, each exported
from its own best-validation checkpoint.
ONNX version: opset 11; IR 6

| Fold | Source checkpoint | val loss | test loss |
| ---- | ----------------- | -------- | --------- |
| 0 | `fold_meta/fold0/logs/version_0/checkpoints/reg-epoch=110-val_loss=10.00.ckpt` | 10.001 | 10.029 |
| 1 | `fold_meta/fold1/logs/version_0/checkpoints/reg-epoch=107-val_loss=10.06.ckpt` | 10.064 | 10.035 |
| 2 | `fold_meta/fold2/logs/version_0/checkpoints/reg-epoch=151-val_loss=10.03.ckpt` | 10.025 | 10.028 |
| 3 | `fold_meta/fold3/logs/version_0/checkpoints/reg-epoch=131-val_loss=10.08.ckpt` | 10.083 | 10.055 |
| 4 | `fold_meta/fold4/logs/version_0/checkpoints/reg-epoch=114-val_loss=10.06.ckpt` | 10.059 | 10.034 |
| 5 | `fold_meta/fold5/logs/version_0/checkpoints/reg-epoch=117-val_loss=10.07.ckpt` | 10.072 | 10.040 |
| 6 | `fold_meta/fold6/logs/version_0/checkpoints/reg-epoch=90-val_loss=10.07.ckpt` | 10.067 | 10.024 |
| 7 | `fold_meta/fold7/logs/version_0/checkpoints/reg-epoch=99-val_loss=10.03.ckpt` | 10.031 | 10.037 |

- Input: a float32 tensor with shape `(batch_size, 18)`.
  - l+: `(px, py, pz, energy)`
  - l-: `(px, py, pz, energy)`
  - Leading jet: `(px, py, pz, energy)`
  - Subleading jet: `(px, py, pz, energy)`
  - MET: `(MET_px, MET_py)`
- Output: a float32 tensor with shape `(batch_size, 8)`.
  - W+: `(px, py, pz, energy)`
  - W-: `(px, py, pz, energy)`

Energy inputs are transformed with `log1p` and all inputs are standardized
using statistics from the training data internally.  **Do not transform them again.**
Just use the RAW leptons, jets and MET as they are provided in the input tensor.
Each standardization is the one fitted on that fold's own training rows, so the
folds are not interchangeable.

> Note: these eight models were trained with the row-index fold assignment, where
> fold `i` holds out the pooled train+validation rows with `row_index % 8 == i`.
> That membership is **not** reproducible from `eventNumber`, so selecting a model
> with `eventNumber % 8` does not reconstruct the held-out set and does not give
> unbiased cross-fitting. Retrain before relying on an `eventNumber`-based split.
>
> They were also trained on a merged file that mixed ggF and VBF, which was
> labelled ggF by mistake. The current training input
> (`mc20_qe_v61_recotruth_ggF_merged.h5`) is ggF only, so these models were
> fitted on a different sample as well as a different split.

Every export was compared against its PyTorch checkpoint with
`onnx/onnxruntime_check.py`; all eight agree within `atol=1e-3, rtol=3e-3`.

Repo: https://gitlab.cern.ch/atlas-physics/higp/nresmultileptons/hww/qe/lvlv/ml-models/hww_pcres_regressor  
Date: 2026-09-19 (training started; fold 7 and the ONNX export finished 2026-09-20)
