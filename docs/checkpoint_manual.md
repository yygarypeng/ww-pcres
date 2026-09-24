# PCRES Inference Manual

The PCRES regressor reconstructs the W+ and W- four-vectors from two charged
leptons, up to two jets, and missing transverse momentum (MET).

Use the ONNX models in `/eos/home-y/ypeng/qe/models/PCRes_260925`. Inference
needs `onnxruntime` and those eight files; PyTorch, Lightning, and this
repository's source are not required.

Trained on ggF samples from the QE `v6.1` dataset, with the different-flavor
(DF) lepton selection and the default TCPT selection.

> These models replace `PCRes_260921` and fix its shifted W momentum mean. On
> the 140,565 test events, the mean W± px and py residuals drop from up to
> 4 GeV to at most 0.3 GeV, and pz stays within 0.4 GeV. The W energies still
> sit about 20 GeV low, and the px and py resolution is 3 to 6% wider.

## Models

| Property | Value |
| --- | --- |
| Location | `/eos/home-y/ypeng/qe/models/PCRes_260925` |
| Files | `hww_regressor_fold0.onnx` ... `hww_regressor_fold7.onnx` |
| Input tensor | `inputs`, `(n_events, 18)`, `float32` |
| Output tensor | `outputs`, `(n_events, 8)`, `float32` |
| Opset / IR version | 11 / 6 |

The batch dimension is dynamic. `float32` is required; ONNX Runtime rejects a
`float64` feed.

Pick the file for each event from its HWWFrames `eventNumber`:

```
fold = eventNumber % 8
```

Take the modulus on the unsigned 64-bit integer, not on a float. The eight
models are not interchangeable: each was trained without the events of its own
fold and carries its own normalization.

## Input

One row per event. All momenta and energies in GeV, four-vectors in Cartesian
order `(px, py, pz, E)`.

Pass raw values. Do not apply logarithms or standardization; the model does its
own preprocessing.

| Index | Input name | Description |
| ---: | --- | --- |
| 0 | `pos_lep_px` | Positive-lepton x momentum |
| 1 | `pos_lep_py` | Positive-lepton y momentum |
| 2 | `pos_lep_pz` | Positive-lepton z momentum |
| 3 | `pos_lep_energy` | Positive-lepton energy |
| 4 | `neg_lep_px` | Negative-lepton x momentum |
| 5 | `neg_lep_py` | Negative-lepton y momentum |
| 6 | `neg_lep_pz` | Negative-lepton z momentum |
| 7 | `neg_lep_energy` | Negative-lepton energy |
| 8 | `jet0_px` | First stored jet x momentum |
| 9 | `jet0_py` | First stored jet y momentum |
| 10 | `jet0_pz` | First stored jet z momentum |
| 11 | `jet0_energy` | First stored jet energy |
| 12 | `jet1_px` | Second stored jet x momentum |
| 13 | `jet1_py` | Second stored jet y momentum |
| 14 | `jet1_pz` | Second stored jet z momentum |
| 15 | `jet1_energy` | Second stored jet energy |
| 16 | `met_px` | Missing transverse x momentum |
| 17 | `met_py` | Missing transverse y momentum |

Jets are pT-sorted, as TCPT provides them: jet0 is the leading jet and jet1 the
subleading one. The model does not sort, so keep that order. Write an absent jet
as exactly `[0, 0, 0, 0]`, which the model masks, and fill the trailing slot
first: an event with one jet has it in the jet0 columns. Do not use NaN or
another sentinel.

Validate before inference; the model does not check its input. All values must
be finite, lepton energies positive, and each jet either exact-zero padding or
positive in energy.

## Output

| Index | Output name | Meaning |
| ---: | --- | --- |
| 0 | `w_pos_px` | Reconstructed W+ x momentum [GeV] |
| 1 | `w_pos_py` | Reconstructed W+ y momentum [GeV] |
| 2 | `w_pos_pz` | Reconstructed W+ z momentum [GeV] |
| 3 | `w_pos_energy` | Reconstructed W+ energy [GeV] |
| 4 | `w_neg_px` | Reconstructed W- x momentum [GeV] |
| 5 | `w_neg_py` | Reconstructed W- y momentum [GeV] |
| 6 | `w_neg_pz` | Reconstructed W- z momentum [GeV] |
| 7 | `w_neg_energy` | Reconstructed W- energy [GeV] |

W+ is matched to the positive input lepton, W- to the negative one. Each W is
its lepton plus a reconstructed massless neutrino, so the eight components are
not independent.

## Example

```python
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/eos/home-y/ypeng/qe/models/PCRes_260925"
FOLDS = 8
SESSIONS = {
    fold: ort.InferenceSession(
        f"{MODEL_DIR}/hww_regressor_fold{fold}.onnx", providers=["CPUExecutionProvider"]
    )
    for fold in range(FOLDS)
}


def predict(raw_inputs, event_numbers):
    """Reconstruct the W+ and W- four-vectors, one model per eventNumber."""
    inputs = np.ascontiguousarray(raw_inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != 18:
        raise ValueError(f"Expected (N, 18), got {inputs.shape}")

    folds = np.asarray(event_numbers, dtype=np.uint64) % np.uint64(FOLDS)
    outputs = np.empty((len(inputs), 8), dtype=np.float32)
    for fold, session in SESSIONS.items():
        rows = folds == fold
        if rows.any():
            outputs[rows] = session.run(["outputs"], {"inputs": inputs[rows]})[0]
    return outputs
```

Keep this manual with the models: the graphs store the architecture and the
normalization statistics, but not the column names or the fold rule.
