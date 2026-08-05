# Flatten-Only Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse event aggregation to one fixed flattened-token architecture and remove pooling flags, event cross-attention, and their supporting code.

**Architecture:** `WBosonRegressor` always flattens all normalized self-attended object tokens into the decoder. Training and checkpoint interfaces no longer know about pooling modes; cross-attention and pooling-ablation utilities are deleted while self-attention and ONNX export remain supported.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, argparse, unittest, pytest, ONNX, ONNX Runtime

## Global Constraints

- Preserve object embeddings, self-attention refinement, empty-jet masking, context normalization, decoder heads, physics reconstruction, losses, and output schema.
- Remove `pooling_mode`, `event_pool`, `pool_norm`, `CrossAttentionBlock`, attention-return behavior, and pooling-specific tooling.
- Do not add compatibility handling for old `lepton` or `learned_query` checkpoints.
- Preserve non-pooling physics diagnostics and unrelated dirty-worktree changes.
- Do not create commits unless explicitly requested.

---

### Task 1: Make The Regressor Flatten-Only

**Files:**
- Modify: `tests/test_attention.py`
- Modify: `model/model.py`

**Interfaces:**
- Consumes: object embeddings and `SelfAttentionBlock` refinement.
- Produces: `WBosonRegressor.global_feature_aggregation(x) -> Tensor` with shape `[batch, num_tokens * d_model]` and constructors without `pooling_mode`.

- [ ] **Step 1: Rewrite model tests around the fixed flattened representation**

Remove `CrossAttentionBlockTest` and all lepton-pooling, attention-weight, and mode-validation tests. Rename the model test class to `FlattenedAggregationTest`. Define `make_model` without `pooling_mode`:

```python
@staticmethod
def make_model(input_dim=18):
    return WBosonRegressor(
        input_dim=input_dim,
        d_model=8,
        num_heads=2,
        std_mean_train=np.zeros(input_dim, dtype=np.float32),
        std_scale_train=np.ones(input_dim, dtype=np.float32),
        attention_blocks=1,
        attention_dropout=0.0,
        decoder_dropout=0.0,
    )
```

Require fixed aggregation and no event-pooling modules:

```python
def test_aggregation_flattens_all_refined_tokens(self):
    model = self.make_model()
    aggregated = model.global_feature_aggregation(torch.randn(3, 18))

    self.assertFalse(hasattr(model, "pooling_mode"))
    self.assertFalse(hasattr(model, "event_pool"))
    self.assertFalse(hasattr(model, "pool_norm"))
    self.assertEqual(aggregated.shape, (3, model.num_tokens * 8))
```

Keep output-shape and embedding-gradient tests. Replace the event-pool-hook masking test with a direct flatten-path check:

```python
def test_empty_jet_embedding_values_are_masked_before_flattening(self):
    torch.manual_seed(3)
    model = self.make_model().eval()
    missing_jet = torch.randn(2, 18)
    missing_jet[:, 8:12] = 0.0
    present_jet = missing_jet.clone()
    present_jet[:, 8:12] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with torch.no_grad():
        missing_before = model.global_feature_aggregation(missing_jet)
        present_before = model.global_feature_aggregation(present_jet)
        model.jet0_embed.bias.add_(torch.arange(8) * 1000.0)
        missing_after = model.global_feature_aggregation(missing_jet)
        present_after = model.global_feature_aggregation(present_jet)

    torch.testing.assert_close(missing_after, missing_before)
    self.assertFalse(torch.allclose(present_after, present_before))
```

- [ ] **Step 2: Run the rewritten model tests and verify RED**

Run: `pytest tests/test_attention.py -v`

Expected: the fixed aggregation test fails because the current default creates `pooling_mode`, `event_pool`, and a `[batch, 2 * d_model]` representation.

- [ ] **Step 3: Remove pooling from model construction and aggregation**

In `WBosonRegressor` and `LightningWBoson`, remove the `pooling_mode` parameters and forwarding. Remove the `CrossAttentionBlock` import, `self.pooling_mode`, validation, `event_pool`, and `pool_norm`. Replace the conditional trunk-width setup and first layer with the direct flattened width:

```python
nn.Linear(d_model * self.num_tokens, 512),
```

Replace the end of `global_feature_aggregation` with:

```python
context = self.context_norm(context)
context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
return context.reshape(batch_size, -1)
```

Remove the `return_attention` parameter.

- [ ] **Step 4: Run focused model and loss tests**

Run: `pytest tests/test_attention.py tests/test_model_loss.py -v`

Expected: all tests pass.

### Task 2: Remove Pooling Interfaces And Ablation Tooling

**Files:**
- Modify: `tests/test_train_overrides.py`
- Modify: `train/train.py`
- Modify: `configs/config.example.yaml`
- Modify: `README.md`
- Modify: `model/model.py`
- Modify: `model/__init__.py`
- Modify: `scripts/save_pcres_io.py`
- Modify: `onnx/convert_to_onnx.py`
- Modify: `onnx/onnxruntime_check.py`
- Modify: `notebooks/visualize.ipynb`
- Delete: `scripts/evaluate_pooling_ablation.py`
- Delete: `tests/test_pooling_ablation.py`

**Interfaces:**
- Consumes: flatten-only `LightningWBoson` checkpoints.
- Produces: training and checkpoint consumers with no pooling configuration or compatibility layer.

- [ ] **Step 1: Add a failing CLI-interface test**

Update override fixtures to contain no pooling keys or arguments. Add:

```python
def test_parser_exposes_no_pooling_mode(self):
    with unittest.mock.patch("sys.argv", ["train.py"]):
        args = parse_args()

    self.assertFalse(hasattr(args, "pooling_mode"))
```

Import `parse_args` and `unittest.mock`. Keep assertions for seed, epochs, event limit, and saved path.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `pytest tests/test_train_overrides.py -v`

Expected: `test_parser_exposes_no_pooling_mode` fails because argparse still creates that attribute.

- [ ] **Step 3: Remove pooling configuration and CLI plumbing**

Delete the `pooling_mode` tuple from `apply_cli_overrides`, the `pooling_mode=` argument in `run_training`, and the complete `--pooling-mode` parser argument. Remove `parameters.pooling_mode` from `configs/config.example.yaml`.

- [ ] **Step 4: Remove checkpoint compatibility plumbing**

Delete `infer_pooling_mode_from_checkpoint` and `load_model_from_checkpoint` from `model/model.py`; export only `LightningWBoson` from `model/__init__.py`. In each checkpoint consumer, import `LightningWBoson` and call its class method directly while preserving existing loading keyword arguments:

```python
model = LightningWBoson.load_from_checkpoint(
    checkpoint_path,
    map_location=device,
    weights_only=False,
    strict=False,
)
```

Apply the equivalent existing local variable names in `scripts/save_pcres_io.py`, `onnx/convert_to_onnx.py`, `onnx/onnxruntime_check.py`, and `notebooks/visualize.ipynb`.

- [ ] **Step 5: Delete pooling-ablation surfaces**

Delete `scripts/evaluate_pooling_ablation.py` and `tests/test_pooling_ablation.py`. Remove the complete `### Pooling ablation` section from `README.md`, through its final report-description sentence, while preserving the following `## Data` section.

- [ ] **Step 6: Run training-interface tests and validate the notebook**

Run: `pytest tests/test_train_overrides.py -v && python -m json.tool notebooks/visualize.ipynb >/dev/null`

Expected: all tests pass and notebook JSON validation exits 0.

### Task 3: Remove Cross-Attention And Keep ONNX Self-Attention

**Files:**
- Modify: `tests/test_onnx_attention.py`
- Modify: `model/layers.py`
- Modify: `onnx/convert_to_onnx.py`

**Interfaces:**
- Consumes: `SelfAttentionBlock` instances containing batch-first `nn.MultiheadAttention`.
- Produces: opset-11 self-attention replacement and dynamic ONNX export without cross-attention code.

- [ ] **Step 1: Rewrite ONNX tests for self-attention only**

Delete `test_cross_attention_matches_pytorch`. Import `model.layers` and add:

```python
def test_cross_attention_block_is_removed(self):
    self.assertFalse(hasattr(layers, "CrossAttentionBlock"))
```

In the dynamic export test, replace the event-pool assertion with:

```python
self.assertTrue(all(
    isinstance(block.mha, Opset11MultiheadAttention)
    for block in export_model.sa_blocks
))
self.assertFalse(hasattr(export_model, "event_pool"))
self.assertEqual(export_model.num_tokens, 5)
```

- [ ] **Step 2: Run ONNX tests and verify RED**

Run: `pytest tests/test_onnx_attention.py -v`

Expected: `test_cross_attention_block_is_removed` fails because the class still exists.

- [ ] **Step 3: Delete cross-attention and simplify the opset-11 adapter**

Delete `CrossAttentionBlock` from `model/layers.py`. Restore a self-attention-only adapter projection:

```python
def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
    if query is not key or key is not value:
        raise ValueError("Opset11MultiheadAttention only supports self-attention")

    batch_size, seq_len, _ = query.shape
    qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if key_padding_mask is not None:
        mask = key_padding_mask.reshape(batch_size, 1, 1, seq_len)
        scores = scores.masked_fill(mask, -1.0e9)
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=self.dropout, training=self.training)
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
    return self.out_proj(context), None
```

- [ ] **Step 4: Run ONNX and full tests**

Run: `pytest tests/test_onnx_attention.py -v && pytest -v`

Expected: all remaining tests pass.

- [ ] **Step 5: Verify complete removal and inspect the diff**

Run: `rg -n "pooling_mode|pooling-mode|event_pool|pool_norm|CrossAttentionBlock|return_attention|evaluate_pooling_ablation" model train onnx scripts tests configs README.md notebooks/visualize.ipynb`

Expected: matches occur only in negative tests asserting that removed names are
absent; production code, tooling, configs, README, and notebook contain no
matches.

Run: `git diff --check`

Expected: no whitespace errors.
