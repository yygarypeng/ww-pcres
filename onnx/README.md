# ONNX Export

Export the newest checkpoint by modification time under `paths.saved_path`:

```bash
cd onnx
python convert_to_onnx.py --config ../configs/config.yaml --output models/hww_pcres_regressor.onnx
```

Compare ONNX Runtime output with the PyTorch checkpoint:

```bash
python onnxruntime_check.py --config ../configs/config.yaml --onnx models/hww_pcres_regressor.onnx
```

Use a specific checkpoint if needed:

```bash
python convert_to_onnx.py --config ../configs/config.yaml --checkpoint /path/to/model.ckpt --output models/hww_pcres_regressor.onnx
python onnxruntime_check.py --config ../configs/config.yaml --checkpoint /path/to/model.ckpt --onnx models/hww_pcres_regressor.onnx
```

Store exported models under `onnx/models/`. The repository-wide `*.onnx` ignore rule keeps these binaries local while this README preserves the directory convention.

The exported model uses a dynamic batch dimension and the 18-column raw input contract. The model constructor rejects any other input width, so a checkpoint from an earlier contract must be retrained rather than exported. The columns are positive-lepton and negative-lepton four-vectors, two jet four-vectors, and MET `(px, py)`. Lepton energies must be finite and strictly positive. Each missing jet must be an exact-zero four-vector, while each present jet must have finite, strictly positive energy. Negative or non-finite jet energies and nonzero jet four-vectors with exactly zero energy are invalid.

The ONNX graph embeds the same neural transform as PyTorch: each energy becomes `log1p(E)`. The internal order is positive-lepton `(px, py, pz, log1p(E))`, negative-lepton `(px, py, pz, log1p(E))`, jet 0 `(px, py, pz, log1p(E))`, jet 1 `(px, py, pz, log1p(E))`, and MET `(px, py)`. Callers must provide raw 18-column data; do not pre-transform it.

The current input-preprocessing schema version is 3. Checkpoints from earlier input-preprocessing schemas are incompatible and require retraining; partial weight migration is not supported. Preserve existing checkpoints and ONNX binaries rather than overwriting them. Since fresh training deletes its configured run directory, use a new `paths.saved_path` when retraining and export to a new ONNX output path.
