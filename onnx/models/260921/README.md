The configured model uses a 216-dimensional representation, 8 attention heads, and 3 attention blocks with 2-head trunk outputs.  It has ~2.83 M trainable parameters on the QE `v6.1` dataset with only DF and TCPT default selections.

ONNX version: opset 11; IR 6

Scored on the shared test split, per fold and averaged over the reported angular observables (theta* and phi* per lepton charge, plus their sums and differences).  `w_fourvec_rmse` is the RMSE over the eight W four-vector components against truth in GeV; the angular columns are in radians, with `ks` the two-sample KS statistic and `emd` the Wasserstein distance between the predicted and truth distributions.

| Fold | epoch | w_fourvec_rmse | angular_rmse | angular_ks | angular_emd |
| ---- | ----- | -------------- | ------------ | ---------- | ----------- |
| fold0 | 125 | 57.9900 | 1.4963 | 0.0073 | 0.0116 |
| fold1 | 99 | 58.0455 | 1.4950 | 0.0077 | 0.0130 |
| fold2 | 129 | 57.8515 | 1.4953 | 0.0061 | 0.0111 |
| fold3 | 76 | 58.1103 | 1.4967 | 0.0074 | 0.0122 |
| fold4 | 106 | 57.9278 | 1.4980 | 0.0067 | 0.0118 |
| fold5 | 110 | 57.9722 | 1.4961 | 0.0067 | 0.0122 |
| fold6 | 120 | 57.8670 | 1.4978 | 0.0064 | 0.0113 |
| fold7 | 118 | 58.0830 | 1.4961 | 0.0068 | 0.0111 |


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
using statistics from the training data internally.  **Do NOT transformed again!** Just use the RAW leptons, jets and MET as they are provided in the input tensor. Each standardization is the one fitted on that fold's own training rows, so the folds are not interchangeable.

Fold membership follows the ATLAS convention on the HWWFrames `eventNumber`: fold `i` was validated on the pooled train+validation (from Danning's HDF5) events with `eventNumber % 8 == i` and trained on the rest, so a consumer selects the model for an event with `eventNumber % 8` and always evaluates it on data the model never fitted.  The shared test split is untouched by the rotation and is the same 140565 events for every fold testing.

Every export was compared against its PyTorch checkpoint with
`onnx/onnxruntime_check.py`; all eight agree within `atol=1e-3, rtol=3e-3`.

Repo: https://gitlab.cern.ch/atlas-physics/higp/nresmultileptons/hww/qe/lvlv/ml-models/hww_pcres_regressor  
Date: 2026-09-21
