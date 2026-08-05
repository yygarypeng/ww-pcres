# Cross-Attention Event Pooling Design

## Goal

Replace token flattening in `WBosonRegressor` with a fixed-size learned event
representation. Two trainable latent queries will extract information from the
object tokens after those tokens have interacted through the existing
self-attention stack.

The change should preserve the current physics reconstruction outputs, empty-jet
masking behavior, training interface, and supported ONNX export workflow.

## Architecture

The object-token path remains unchanged through self-attention:

1. Standardize the event features.
2. Embed the two leptons, two jets, MET, and optional high-level features as
   separate tokens.
3. Mask empty jet tokens.
4. Refine the object tokens with the configured `SelfAttentionBlock` stack.

After refinement, two learned query vectors cross-attend to the object tokens.
The queries have parameter shape `[1, 2, d_model]` and are expanded to the batch
size during the forward pass. Object tokens are the keys and values, so the
existing key padding mask prevents either query from attending to empty jets.

The cross-attention output is normalized and flattened from
`[batch, 2, d_model]` to `[batch, 2 * d_model]`. This replaces the current
flattened `[batch, num_tokens * d_model]` representation and becomes the input
to the existing residual decoder trunk.

The two queries remain semantically unconstrained. They may specialize during
training, but neither is assigned to a particular W boson.

## Components

### Cross-Attention Block

Add a pre-normalized residual block with this interface:

```python
forward(queries, context, key_padding_mask=None)
```

It contains:

- Separate layer normalization for queries and context.
- Multi-head attention with normalized queries as queries and normalized
  context as keys and values.
- A residual connection and dropout around attention.
- A pre-normalized feed-forward network with a residual connection and dropout.

Its output shape matches the query shape.

### Learned Event Queries

`WBosonRegressor` owns an `nn.Parameter` initialized from a normal distribution
with standard deviation `0.02`. It contains exactly two query vectors. Query
count is intentionally fixed rather than exposed as configuration because this
design targets the two-W event topology and does not yet have evidence that it
needs tuning.

### Decoder Input

The first decoder linear layer changes from `d_model * num_tokens` inputs to
`2 * d_model`. No regression heads or physics reconstruction layers change.

## Data Flow

```text
event features
  -> standardization and object embeddings
  -> masked self-attention over object tokens
  -> two learned queries cross-attend to object tokens
  -> normalize and flatten the two query outputs
  -> residual decoder and neutrino parameter heads
  -> physics-aware W four-vector construction
```

## ONNX Export

The opset-11 replacement for `nn.MultiheadAttention` currently rejects
cross-attention. Extend it to project queries, keys, and values separately when
their tensors differ while preserving its current self-attention behavior.
Masking, attention scaling, softmax, dropout-free evaluation, output projection,
and dynamic batch support must remain equivalent to PyTorch inference.

The replacement should continue to obtain projection weights from the original
`nn.MultiheadAttention` module so checkpoint loading and conversion require no
special migration.

## Error Handling

Construction continues to rely on `nn.MultiheadAttention` validation for an
invalid `d_model`/head-count combination. The cross-attention block assumes
queries and context share `d_model`, which is guaranteed by construction.

The existing `input_dim >= 18` validation remains unchanged. Empty jets are
represented only through the existing boolean key padding mask; no new missing
object convention is introduced.

## Testing

Add focused tests that verify:

- The regressor produces the expected output shape with and without high-level
  features.
- The event aggregation output has shape `[batch, 2 * d_model]`.
- Backpropagation produces a finite, nonzero gradient for the learned queries.
- Changing the values of masked jet tokens does not change aggregation output.
- The cross-attention block has stable forward and backward behavior.
- ONNX conversion supports distinct query and context tensors.
- ONNX Runtime predictions remain within the existing numerical tolerance of
  PyTorch predictions for a dynamic batch size.

## Compatibility

This architecture changes parameter names and decoder dimensions, so old model
checkpoints are not expected to load strictly. No compatibility shim will be
added because the architecture itself is changing and there is no requirement
to continue training old checkpoints.

Configuration and command-line interfaces remain unchanged.
