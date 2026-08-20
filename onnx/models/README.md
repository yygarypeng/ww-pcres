# Local ONNX Models

Store ONNX model binaries in this directory. Files matching `*.onnx` are ignored by Git so trained model artifacts remain local and are not removed during source cleanup.

Current local models include the 18-column current-checkpoint export, its dated copy, a historical export, and the two reconstructed-fold exports. Check `model.graph.input[0]` before using historical models because their input schemas predate the current 18/21-column contracts.
