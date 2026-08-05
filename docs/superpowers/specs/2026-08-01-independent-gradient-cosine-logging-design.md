# Independent Gradient-Cosine Logging

## Problem

`parameters.log_loss_gradient_cosines: true` currently creates `grad_cos/*` CSV columns but does not populate them unless `adaptive_loss_weights` is also enabled. Both the first-batch capture and the epoch-end calculation are gated on adaptive weighting. This makes the logging option ineffective for runs that intentionally use fixed loss weights.

## Design

Separate gradient-cosine calculation from adaptive weight mutation.

- Capture the first training batch of each epoch when either gradient-cosine logging or adaptive weighting is enabled.
- Calculate loss-versus-total and loss-versus-rest cosine values in a side-effect-free method.
- Log calculated values only when `log_loss_gradient_cosines` is enabled.
- Update loss weights only when `adaptive_loss_weights` is enabled.
- Continue using the existing non-Huber active loss set and existing `grad_cos/{loss}__total` and `grad_cos/{loss}__rest` metric names.
- Keep the current once-per-epoch calculation based on the first training batch.

The adaptive update may reject an update when its normalized weighting signal is invalid or degenerate. That rejection must not discard otherwise valid cosine diagnostics.

## Data Flow

At the first `training_step` in an epoch, the model stores detached CPU copies of the batch if either feature needs epoch-end gradient analysis. At `on_train_epoch_end`, it recomputes losses with gradients enabled, calculates cosine diagnostics, optionally logs them, and optionally applies an adaptive weight update. It then clears the stored batch.

## Testing

Tests will verify that:

- logging enabled with adaptive weighting disabled still captures a batch and emits cosine metrics;
- fixed loss weights are unchanged by diagnostic logging;
- both options disabled avoid retaining a diagnostic batch;
- the existing CSV metric names remain unchanged;
- the notebook warning describes the actual conditions instead of incorrectly claiming logging was disabled.
