# Remove Learned-Query Pooling Design

## Goal

Remove the `learned_query` pooling mode because its trainable query offsets can
distort predictions. Keep `flatten` and lepton-anchored cross-attention as the
only supported pooling modes, and prevent old learned-query checkpoints from
being silently reinterpreted.

## Model Behavior

`WBosonRegressor` will accept only `flatten` and `lepton`. Lepton pooling will
always use the two refined lepton tokens, `context[:, :2]`, directly as the
cross-attention queries. The `w_query_offsets` parameter and the conditional
offset addition will be removed.

The flattening path, cross-attention implementation, decoder, regression heads,
physics reconstruction, and output schema will not change.

## Checkpoint Handling

`infer_pooling_mode_from_checkpoint` remains the central compatibility check.
It will reject a checkpoint with a clear `ValueError` when either:

- `hyper_parameters.pooling_mode` is `learned_query`; or
- the state dictionary contains `model.w_query_offsets`.

The state-dictionary check covers older checkpoints without saved pooling-mode
metadata. Other checkpoints retain the current behavior: saved supported modes
are returned directly, a state dictionary containing `model.event_pool.*`
infers `lepton`, and older pre-pooling checkpoints infer `flatten`.

## Interfaces And Documentation

The `--pooling-mode` command-line option will accept only `flatten` and
`lepton`. The README will describe and demonstrate only these two modes. The
pooling-ablation evaluator remains generic and needs no behavioral changes.

Historical design and implementation-plan documents remain unchanged as records
of prior design work. Current executable code, tests, and user-facing README
content will contain no supported `learned_query` path.

## Error Handling

Constructing a regressor with `pooling_mode="learned_query"` will use the
existing unsupported-mode `ValueError`. Loading, evaluating, or converting an
old learned-query checkpoint will fail before model construction with an error
that identifies `learned_query` as unsupported. This avoids silently dropping
the learned offsets and changing checkpoint semantics.

## Testing

Focused tests will verify:

- The regressor rejects `learned_query` as an unsupported pooling mode.
- Lepton pooling passes the refined lepton tokens directly to cross-attention.
- Attention weights remain available for lepton pooling.
- Checkpoint inference still detects `flatten` and `lepton`.
- Checkpoints identified as learned-query through metadata or state parameters
  are rejected clearly.
- CLI configuration override coverage uses a supported mode.

Existing broader tests will verify that the removal does not change supported
model, training, evaluation, or ONNX behavior.
