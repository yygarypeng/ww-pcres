# Local ONNX Models

Each dated directory holds one export and a README describing it; the `*.onnx`
binaries are gitignored and stay local.

- `260921/`: current. Eight cross-fitting models selected by `eventNumber % 8`,
  trained on the ggF-only input.
- `260919/`: superseded. Row-index folds on a mixed ggF/VBF input.
- `260821/`: historical single-fold model.

All three take the 18-column input described in `onnx/README.md`; check
`model.graph.input[0]` before using any other historical export.
