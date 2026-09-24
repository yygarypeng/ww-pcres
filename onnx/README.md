# ONNX Export

Run these commands from `onnx/`. Without `--checkpoint`, the scripts use the
most recently modified checkpoint under `paths.saved_path`, usually `last.ckpt`
(the final epoch, not the best); for a cross-fitting run, pass each fold's
checkpoint explicitly.

```bash
cd onnx
python convert_to_onnx.py --config ../configs/config.yaml \
    --checkpoint /path/to/model.ckpt --output models/<date>/hww_regressor_fold0.onnx
python onnxruntime_check.py --config ../configs/config.yaml \
    --checkpoint /path/to/model.ckpt --onnx models/<date>/hww_regressor_fold0.onnx
```

The second command compares ONNX Runtime with the PyTorch checkpoint. Keep each
export under `onnx/models/<date>/` with a model card; the `*.onnx` binaries are
gitignored.

## Input contract

The model takes raw float32 inputs of shape `(batch, 18)` with a dynamic batch
dimension: positive-lepton, negative-lepton, leading-jet, and subleading-jet
`(px, py, pz, energy)`, then MET `(px, py)`. Do not pre-transform them; the graph
applies `log1p` to the energies and the training standardization internally.

- Lepton energies must be finite and strictly positive.
- A missing jet is an exact-zero four-vector; a present jet needs a finite,
  strictly positive energy.

The model rejects any other input width, so checkpoints from an earlier input
contract must be retrained rather than exported.
