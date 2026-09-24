# Local ONNX Models

One directory per export, each with a model card; the `*.onnx` binaries are
gitignored and stay local.

- `260925/`: latest. Eight cross-fitting models (`eventNumber % 8`) trained with
  the sweep-tuned `configs/kfold_config.yaml` on the ggF-only input.
- `260921/`: the untuned baseline with the same architecture, folds, and input;
  `docs/checkpoint_manual.md` documents its published copy.
- `260919/`: superseded. Eight models trained with row-index folds on an input
  that mixed ggF and VBF.
- `260821/`: historical single model; only its model card remains.

Every model card documents an 18-column input and an 8-column output.
