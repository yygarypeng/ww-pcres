# ONNX Export

Export the latest checkpoint under `paths.saved_path`:

```bash
cd onnx
python convert_to_onnx.py --config ../configs/config.yaml --output hww_pcres_regressor.onnx
```

Compare ONNX Runtime output with the PyTorch checkpoint:

```bash
python onnxruntime_check.py --config ../configs/config.yaml --onnx hww_pcres_regressor.onnx
```

Use a specific checkpoint if needed:

```bash
python convert_to_onnx.py --config ../configs/config.yaml --checkpoint /path/to/model.ckpt --output hww_pcres_regressor.onnx
python onnxruntime_check.py --config ../configs/config.yaml --checkpoint /path/to/model.ckpt --onnx hww_pcres_regressor.onnx
```

The exported model uses a dynamic batch dimension and the public raw 22-column input contract. Columns are positive-lepton and negative-lepton four-vectors, two jet four-vectors, MET `(px, py)`, then `m_ll`, `deta_ll`, `dphi_ll`, and `dphi_llmet`. Lepton energies must be finite and strictly positive. Finite negative jet energy is an accepted absent-jet sentinel whose complete four-vector is canonicalized to zero. Present jets have finite, strictly positive energy; non-finite jet energy and nonzero momentum with exactly zero energy remain invalid.

The ONNX graph embeds the same 22-to-24 neural transform as PyTorch: each energy becomes `log1p(E)`, and each raw delta-phi becomes an unstandardized sine/cosine pair. The internal order is positive-lepton `(px, py, pz, log1p(E))`, negative-lepton `(px, py, pz, log1p(E))`, jet 0 `(px, py, pz, log1p(E))`, jet 1 `(px, py, pz, log1p(E))`, MET `(px, py)`, `m_ll`, `deta_ll`, `sin(dphi_ll)`, `cos(dphi_ll)`, `sin(dphi_llmet)`, `cos(dphi_llmet)`. Training statistics are fitted per jet slot using only present jets, and the sine/cosine features use fixed mean zero and scale one. Callers must continue to provide raw 22-column data; do not pre-transform it.

Checkpoints from the previous input-preprocessing schema are incompatible and require retraining; partial weight migration is not supported. Preserve existing checkpoints and ONNX binaries rather than overwriting them. Since fresh training deletes its configured run directory, use a new `paths.saved_path` when retraining and export a new ONNX output path.
