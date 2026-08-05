# Periodic Local-MMD Conditioning Design

## Goal

Condition the angular local-MMD loss on all four reconstructed high-level
observables while preserving the periodic geometry of azimuthal differences.
The predictor's existing 22 input features and high-level embedding remain
unchanged.

## Conditioning Features

The four physical observables are represented by six numerical features:

1. `m_ll`
2. `deta_ll`
3. `sin(dphi_ll)`
4. `cos(dphi_ll)`
5. `sin(dphi_llmet)`
6. `cos(dphi_llmet)`

Using sine and cosine prevents values close to `-pi` and `pi` from appearing
distant to the Euclidean condition kernel.

## Normalization

Compute the mean and standard deviation of the six transformed conditioning
features from `X_train` only. Clamp constant-feature scales to a positive
floating-point epsilon. Store the resulting statistics as model buffers so
they move across devices and remain in checkpoints.

Validation, test, and inference batches use the fixed training statistics.
No batch-dependent normalization is introduced.

## Data Flow

`build_datamodule` computes two independent sets of statistics:

- Existing 22-feature statistics for predictor input standardization.
- New six-feature statistics for local-MMD conditioning.

`run_training` passes both sets to `LightningWBoson`. `WBosonRegressor.forward`
constructs the six conditioning features from the raw high-level input columns,
normalizes them, and returns them as `aux["cond"]`. The existing loss path then
passes this tensor through `angular_loss_mmd` to `compute_local_mmd`.

The network's high-level token continues to use the four existing standardized
raw values. This isolates the periodic representation change to the MMD kernel.

## Input Contract

The six-feature condition requires the current 22-column input layout. Model
construction with fewer than 22 columns remains valid for prediction and other
losses, but enabling angular local MMD requires the conditioning features.
For such models, the auxiliary condition tensor is empty and the angular loss
must raise a clear error rather than silently use an invalid kernel.

## Testing

Add focused tests that verify:

- The transformed conditioning feature order and values.
- Training-only statistics standardize the six features correctly.
- Angles straddling `-pi` and `pi` are close in condition space.
- `aux["cond"]` has shape `[batch_size, 6]` for the 22-feature model.
- The exact normalized condition reaches `compute_local_mmd`.
- Existing predictor output-shape and training tests continue to pass.

## Non-Goals

- Changing the predictor's input representation or architecture.
- Selecting a subset of the four physical observables.
- Retuning MMD bandwidths or loss weights.
- Adding learned conditioning features.
