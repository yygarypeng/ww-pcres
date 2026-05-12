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

The exported model uses a dynamic batch dimension.
