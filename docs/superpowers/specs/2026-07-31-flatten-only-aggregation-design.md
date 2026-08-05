# Flatten-Only Aggregation Design

## Goal

Use one event aggregation architecture: flatten all refined object tokens.
Remove the pooling mode flag, event cross-attention, and all supporting
configuration, compatibility, pooling diagnostics, and tests.

## Model

`WBosonRegressor` will embed, mask, and refine object tokens with the existing
self-attention stack. `global_feature_aggregation` will then normalize and
reshape the complete token tensor to `[batch, num_tokens * d_model]`.

The model and Lightning constructors will no longer accept `pooling_mode`.
`event_pool`, `pool_norm`, `CrossAttentionBlock`, and attention-return behavior
will be removed. The decoder trunk will always accept
`num_tokens * d_model` features.

## Interfaces And Tooling

Training configuration and command-line parsing will no longer expose a pooling
mode. The example configuration and README pooling-ablation section will be
removed. The pooling-ablation evaluator will be deleted because there are no
pooling alternatives to compare.

Checkpoint consumers will call `LightningWBoson.load_from_checkpoint`
directly. Pooling-mode inference, the guarded shared loader, and compatibility
handling for event-pooling checkpoints will be removed. Old `lepton` and
`learned_query` checkpoints are unsupported and may fail ordinary Lightning
loading because their parameters and decoder dimensions differ.

## Cross-Attention And ONNX

`CrossAttentionBlock` will be deleted. The opset-11 multi-head-attention adapter
will support only the self-attention path used by `SelfAttentionBlock`; its
cross-attention-specific code and tests will be removed. Dynamic-batch ONNX
export and PyTorch parity for self-attention remain required.

## Preserved Behavior

- Object-specific embeddings and optional high-level token creation.
- Empty-jet masking before and after every self-attention block.
- Context normalization before flattening.
- Decoder, regression heads, physics reconstruction, losses, and output schema.
- Existing non-pooling diagnostics, even if they were initially used by the
  ablation evaluator.

Historical design and plan documents remain unchanged as records.

## Testing

Tests will verify flattened aggregation shape, absence of `event_pool`, empty-jet
masking, embedding gradients, model output shapes, self-attention behavior,
training overrides unrelated to pooling, self-attention ONNX parity, and dynamic
ONNX export. Pooling-mode, cross-attention, attention-weight, pooling-inference,
and pooling-ablation tests will be removed.
