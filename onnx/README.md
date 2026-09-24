# ONNX Export

Run these commands from `onnx/`. Without `--checkpoint`, the newest checkpoint
under `paths.saved_path` is exported; for a cross-fitting run, export each fold's
checkpoint explicitly.

```bash
cd onnx
python convert_to_onnx.py --config ../configs/config.yaml --output models/hww_pcres_regressor.onnx
python convert_to_onnx.py --config ../configs/config.yaml --checkpoint /path/to/model.ckpt --output models/hww_pcres_regressor.onnx
```

Compare ONNX Runtime with the PyTorch checkpoint (add the same `--checkpoint`):

```bash
python onnxruntime_check.py --config ../configs/config.yaml --onnx models/hww_pcres_regressor.onnx
```

Keep exported models under `onnx/models/<date>/` with a README describing them;
the `*.onnx` binaries themselves are gitignored.

## Input contract

The model takes raw float32 inputs of shape `(batch, 18)` with a dynamic batch
dimension: positive-lepton, negative-lepton, leading-jet, and subleading-jet
`(px, py, pz, energy)`, then MET `(px, py)`. Do not pre-transform them; the graph
applies `log1p` to the energies and the training standardization internally.

- Lepton energies must be finite and strictly positive.
- A missing jet is an exact-zero four-vector; a present jet needs a finite,
  strictly positive energy.

The model rejects any other input width, so checkpoints from an earlier input
contract must be retrained rather than exported. Training deletes its run
directory, so retrain into a new `paths.saved_path` and export to a new path.
