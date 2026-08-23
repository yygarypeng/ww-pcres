The configured model uses a 192-dimensional representation, 6 attention
heads, and 5 attention blocks with 2-head trunk outputs.  Better constraints
on ptvv through new structural designs. It has ~4.08 M trainable parameters
on the QE `v6.1` dataset with only DF and TCPT default selections.

Model artifacts: hww_regressor_fold0.onnx; hww_regressor_fold1.onnx  
Source checkpoint: meta/logs/version_0/checkpoints/reg-epoch=216-val_loss=8.18.ckpt  
Fold 1 is an identical copy of fold 0 because training uses one fold.
ONNX version: opset 11; IR 6

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
using statistics from the training data internally.  **Do NOT transformed again!**
Just use the RAW leptons, jets and MET as they are provided in the input tensor.

> Note: Training currently uses one fold only. The same model is used for
> both even (fold0) and odd (fold1) `eventNumber` selections.  Robustness, and tunings
> are needed to investigate!

Repo: https://gitlab.cern.ch/atlas-physics/higp/nresmultileptons/hww/qe/lvlv/ml-models/hww_pcres_regressor  
Date: 2026-08-21
