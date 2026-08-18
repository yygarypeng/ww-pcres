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

The exported model uses a dynamic batch dimension and the public raw 21-column input contract. Columns are positive-lepton and negative-lepton four-vectors, two jet four-vectors, MET `(px, py)`, then `m_ll`, `deta_ll`, and `dphi_ll`. Lepton energies must be finite and strictly positive. Each missing jet must be an exact-zero four-vector, while each present jet must have finite, strictly positive energy. Negative or non-finite jet energies and nonzero jet four-vectors with exactly zero energy are invalid.

The ONNX graph embeds the same 21-to-21 neural transform as PyTorch: each energy becomes `log1p(E)`, and raw `dphi_ll` is passed through unchanged. The internal order is positive-lepton `(px, py, pz, log1p(E))`, negative-lepton `(px, py, pz, log1p(E))`, jet 0 `(px, py, pz, log1p(E))`, jet 1 `(px, py, pz, log1p(E))`, MET `(px, py)`, `m_ll`, `deta_ll`, and `dphi_ll`. Training statistics are fitted per jet slot using only present jets, and `dphi_ll` uses fixed mean zero and scale one. Callers must continue to provide raw 21-column data; do not pre-transform it.

The current input-preprocessing schema version is 3. Checkpoints from earlier input-preprocessing schemas are incompatible and require retraining; partial weight migration is not supported. Preserve existing checkpoints and ONNX binaries rather than overwriting them. Since fresh training deletes its configured run directory, use a new `paths.saved_path` when retraining and export to a new ONNX output path.
