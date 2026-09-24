The configured model uses a 216-dimensional representation, 8 attention heads, and 3 attention blocks with 2-head trunk outputs.  It has ~2.83 M trainable parameters on the QE `v6.1` dataset with only DF and TCPT default selections.

The architecture is the same as `260921`; the training config is the sweep-tuned `configs/kfold_config.yaml` (trial 5 of the constrained v2 Optuna sweep), which reweights the losses (W four-vector 0.56, W mass MMD 14.3, angular MMD 106.5, dmet 0.105), adds the four-vector mean-residual bias penalty (8.93), and lowers the weight decay to 2.3e-6.

Model artifacts: hww_regressor_fold0.onnx ... hww_regressor_fold7.onnx  
Source run: 8-fold cross-fitting under `fold_meta_ggF_tuned_v2/fold{0..7}`, each exported from its own best-val_loss checkpoint (`reg-epoch=<epoch>-val_loss=*.ckpt`).  
ONNX version: opset 11; IR 6  
Client contract: `docs/checkpoint_manual.md`

Scored with the ONNX models on the shared test split, per fold and averaged over the reported angular observables (theta* and phi* per lepton charge, plus their sums and differences).  `w_fourvec_rmse` is the RMSE over the eight W four-vector components against truth in GeV; the angular columns are in radians, with `ks` the two-sample KS statistic and `emd` the Wasserstein distance between the predicted and truth distributions.

| Fold | epoch | w_fourvec_rmse | angular_rmse | angular_ks | angular_emd |
| ---- | ----- | -------------- | ------------ | ---------- | ----------- |
| fold0 | 184 | 57.7935 | 1.5006 | 0.0056 | 0.0086 |
| fold1 | 115 | 58.1283 | 1.5032 | 0.0054 | 0.0091 |
| fold2 | 114 | 58.1790 | 1.5033 | 0.0052 | 0.0092 |
| fold3 | 105 | 57.9366 | 1.5026 | 0.0056 | 0.0095 |
| fold4 | 144 | 58.1269 | 1.5027 | 0.0051 | 0.0088 |
| fold5 | 110 | 58.1668 | 1.5049 | 0.0051 | 0.0091 |
| fold6 | 106 | 58.0475 | 1.5022 | 0.0053 | 0.0096 |
| fold7 | 146 | 58.0510 | 1.5012 | 0.0052 | 0.0089 |

Against the untuned `260921` export, the angular KS and EMD are lower on every fold, the per-event angular RMSE is ~0.006 rad higher, and the W four-vector RMSE is unchanged.

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
using statistics from the training data internally.  **Do not transform them again.** Just use the RAW leptons, jets and MET as they are provided in the input tensor. Each standardization is the one fitted on that fold's own training rows, so the folds are not interchangeable.

Fold membership follows the ATLAS convention on the HWWFrames `eventNumber`: fold `i` was validated on the pooled train+validation (from Danning's HDF5) events with `eventNumber % 8 == i` and trained on the rest, so a consumer selects the model for an event with `eventNumber % 8` and always evaluates it on data the model never fitted.  The shared test split is untouched by the rotation and is the same 140565 events for every fold testing.

Every export was compared against its PyTorch checkpoint with
`onnx/onnxruntime_check.py`; all eight agree within `atol=1e-3, rtol=3e-3`.
On the 140565 test events, against the checkpoint in full fp32, all eight also
agree within that tolerance, with a largest absolute difference of 0.016 GeV.

Repo: https://gitlab.cern.ch/atlas-physics/higp/nresmultileptons/hww/qe/lvlv/ml-models/hww_pcres_regressor  
Date: 2026-09-25 (training started 2026-09-24)
