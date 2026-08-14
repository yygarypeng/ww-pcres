# Physics Loss Warm-Up Design

## Goal

Train the regressor only with the standardized W-four-vector Huber objective
during the first 20 epochs, then enable every configured physics constraint at
its full fixed weight.

## Behavior

Use `parameters.physics_start_epoch` as the single physics-loss activation epoch.
Before that epoch, training sets every effective loss weight except `huber` to
zero. At and after that epoch, all configured weights become active. There is no
ramp and no separate schedule.

Validation and test evaluation always include the configured physics terms,
using any epoch-specific angular schedule. This prevents warm-up checkpoints
from winning solely because their objective omits physics terms.

The active run uses `physics_start_epoch: 20` and raises the Higgs-shell weight from
6 to 30. Existing MMD kernels, MMD weights, batch size, learning rate, and
gradient clipping remain unchanged.

## Verification

- Training before epoch 20 computes only `huber`.
- Evaluation before epoch 20 computes every configured loss.
- Training at epoch 20 computes every configured loss.
- Effective training-weight logs report zero for every non-Huber loss during
  warm-up and configured weights after activation.
- Omitting the option retains immediate activation.
- Negative activation epochs remain invalid.
- Older checkpoints and configurations using `mmd_start_epoch` migrate to the
  new name, while new checkpoints save only `physics_start_epoch`.
