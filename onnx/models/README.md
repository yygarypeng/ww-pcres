# Local ONNX Models

Store ONNX model binaries in this directory. Files matching `*.onnx` are ignored by Git so trained model artifacts remain local and are not removed during source cleanup.

Current local models are the eight cross-fitting exports under `260921/`, which use the `eventNumber % 8` fold assignment and the ggF-only training input. The earlier `260919/` cross-fitting exports and the historical `260821/` export are superseded and kept for reference only. Check `model.graph.input[0]` before using historical models because their input schemas predate the current 18/21-column contracts.
