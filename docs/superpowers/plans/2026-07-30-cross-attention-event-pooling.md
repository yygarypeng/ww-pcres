# Cross-Attention Event Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flattened object-token aggregation with two learned queries that cross-attend to self-attended event tokens while preserving training and ONNX inference.

**Architecture:** Keep the current object embeddings and self-attention refiners. Add a pre-normalized cross-attention block whose two trainable queries produce a fixed `[batch, 2 * d_model]` decoder input, and extend the opset-11 attention adapter to support differing query and context sequence lengths.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, unittest, pytest, ONNX, ONNX Runtime

## Global Constraints

- Preserve the existing physics reconstruction outputs, empty-jet masking behavior, and public training configuration.
- Use exactly two semantically unconstrained learned queries initialized from `Normal(0, 0.02)`.
- Do not add checkpoint compatibility shims for the changed architecture.
- Preserve opset-11 export and dynamic batch support.

---

### Task 1: Pre-Normalized Cross-Attention Block

**Files:**
- Modify: `model/layers.py:14-50`
- Create: `tests/test_attention.py`

**Interfaces:**
- Consumes: `_AttnFFN(d_model, ffn_dim, dropout)` from `model.layers`.
- Produces: `CrossAttentionBlock(d_model, nhead, dropout=0.3)` with `forward(queries, context, key_padding_mask=None) -> Tensor` matching the query shape.

- [ ] **Step 1: Write failing shape, masking, and gradient tests**

```python
import unittest

import torch

from model.layers import CrossAttentionBlock


class CrossAttentionBlockTest(unittest.TestCase):
    def test_output_matches_query_shape_and_backpropagates(self):
        torch.manual_seed(1)
        block = CrossAttentionBlock(d_model=8, nhead=2, dropout=0.0)
        queries = torch.randn(3, 2, 8, requires_grad=True)
        context = torch.randn(3, 5, 8, requires_grad=True)

        output = block(queries, context)
        output.square().mean().backward()

        self.assertEqual(output.shape, queries.shape)
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_masked_context_values_do_not_affect_output(self):
        torch.manual_seed(2)
        block = CrossAttentionBlock(d_model=8, nhead=2, dropout=0.0).eval()
        queries = torch.randn(2, 2, 8)
        context = torch.randn(2, 5, 8)
        mask = torch.tensor([[False, False, True, False, False], [False, True, True, False, False]])
        changed = context.clone()
        changed[mask] = 1.0e6

        expected = block(queries, context, key_padding_mask=mask)
        actual = block(queries, changed, key_padding_mask=mask)

        torch.testing.assert_close(actual, expected)
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest tests/test_attention.py -v`

Expected: collection fails because `CrossAttentionBlock` is not defined in `model.layers`.

- [ ] **Step 3: Implement the block with a normalized FFN residual**

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = _AttnFFN(d_model, d_model * 4, dropout)

    def forward(self, queries, context, key_padding_mask=None):
        x = queries + self.dropout(self.mha(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0])
        return x + self.dropout(self.ffn(self.ffn_norm(x)))
```

Avoid computing `context_norm(context)` twice in the final implementation: assign the normalized context to a local variable before calling `mha`.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_attention.py -v`

Expected: 2 passed.

### Task 2: Learned Event-Query Pooling

**Files:**
- Modify: `model/model.py:6-112`
- Modify: `tests/test_attention.py`

**Interfaces:**
- Consumes: `CrossAttentionBlock.forward(queries, context, key_padding_mask=None)`.
- Produces: `WBosonRegressor.event_queries` with shape `[1, 2, d_model]` and `global_feature_aggregation(x) -> Tensor` with shape `[batch, 2 * d_model]`.

- [ ] **Step 1: Add failing regressor pooling tests**

```python
import numpy as np

from model.model import WBosonRegressor


class LearnedEventPoolingTest(unittest.TestCase):
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

    def test_aggregation_uses_two_learned_queries(self):
        model = self.make_model()
        aggregated = model.global_feature_aggregation(torch.randn(3, 18))

        self.assertEqual(model.event_queries.shape, (1, 2, 8))
        self.assertEqual(aggregated.shape, (3, 16))

    def test_learned_queries_receive_gradients(self):
        model = self.make_model()
        model(torch.randn(4, 18)).square().mean().backward()

        gradient = model.event_queries.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_regressor_supports_high_level_features(self):
        model = self.make_model(input_dim=22)
        self.assertEqual(model(torch.randn(2, 22)).shape, (2, 8))

    def test_empty_jet_values_are_ignored_by_pooling(self):
        model = self.make_model().eval()
        x = torch.randn(2, 18)
        x[:, 8:12] = 0.0
        changed = x.clone()
        changed[:, 8:12] = 1.0e6

        empty_mask = torch.zeros((2, model.num_tokens), dtype=torch.bool)
        empty_mask[:, 2] = True
        with torch.no_grad():
            x_std = model.norm(x)
            changed_std = model.norm(changed)
            base = [model.lep0_embed(x_std[:, 0:4]), model.lep1_embed(x_std[:, 4:8]), model.jet0_embed(x_std[:, 8:12]), model.jet1_embed(x_std[:, 12:16]), model.met_embed(x_std[:, 16:18])]
            altered = [model.lep0_embed(changed_std[:, 0:4]), model.lep1_embed(changed_std[:, 4:8]), model.jet0_embed(changed_std[:, 8:12]), model.jet1_embed(changed_std[:, 12:16]), model.met_embed(changed_std[:, 16:18])]
            base_context = torch.stack(base, dim=1).masked_fill(empty_mask.unsqueeze(-1), 0.0)
            altered_context = torch.stack(altered, dim=1).masked_fill(empty_mask.unsqueeze(-1), 0.0)

        torch.testing.assert_close(base_context, altered_context)
```

- [ ] **Step 2: Run tests and verify aggregation shape failure**

Run: `pytest tests/test_attention.py::LearnedEventPoolingTest -v`

Expected: failures because `event_queries` does not exist and aggregation still returns `num_tokens * d_model` features.

- [ ] **Step 3: Integrate two learned queries into the regressor**

Import `CrossAttentionBlock`, then initialize the pooling components:

```python
self.event_queries = nn.Parameter(torch.empty(1, 2, d_model))
nn.init.normal_(self.event_queries, std=0.02)
self.event_pool = CrossAttentionBlock(d_model, num_heads, dropout=attention_dropout)
self.context_norm = nn.LayerNorm(d_model)
self.pool_norm = nn.LayerNorm(d_model)
```

Change the trunk input to `nn.Linear(d_model * 2, 512)`. At the end of `global_feature_aggregation`, retain masked object normalization, then pool:

```python
context = self.context_norm(context)
context = context.masked_fill(key_mask.unsqueeze(-1), 0.0)
queries = self.event_queries.expand(batch_size, -1, -1)
pooled = self.event_pool(queries, context, key_padding_mask=key_mask)
return self.pool_norm(pooled).reshape(batch_size, -1)
```

- [ ] **Step 4: Run model and existing loss tests**

Run: `pytest tests/test_attention.py tests/test_model_loss.py -v`

Expected: all tests pass.

### Task 3: Opset-11 Cross-Attention Support

**Files:**
- Modify: `onnx/convert_to_onnx.py:17-49`
- Create: `tests/test_onnx_attention.py`

**Interfaces:**
- Consumes: a batch-first `nn.MultiheadAttention` with shared Q/K/V embedding dimension.
- Produces: `Opset11MultiheadAttention.forward(query, key, value, key_padding_mask=None)` supporting both self-attention and cross-attention with unequal query/key sequence lengths.

- [ ] **Step 1: Write failing PyTorch parity tests**

```python
import unittest

import torch
import torch.nn as nn

from onnx.convert_to_onnx import Opset11MultiheadAttention


class Opset11MultiheadAttentionTest(unittest.TestCase):
    def test_cross_attention_matches_pytorch(self):
        torch.manual_seed(3)
        source = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).eval()
        replacement = Opset11MultiheadAttention(source).eval()
        query = torch.randn(3, 2, 8)
        context = torch.randn(3, 5, 8)
        mask = torch.tensor([[False, False, True, False, False]] * 3)

        expected = source(query, context, context, key_padding_mask=mask, need_weights=False)[0]
        actual = replacement(query, context, context, key_padding_mask=mask)[0]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_self_attention_matches_pytorch(self):
        torch.manual_seed(4)
        source = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).eval()
        replacement = Opset11MultiheadAttention(source).eval()
        inputs = torch.randn(2, 5, 8)

        expected = source(inputs, inputs, inputs, need_weights=False)[0]
        actual = replacement(inputs, inputs, inputs)[0]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run tests and verify cross-attention rejection**

Run: `pytest tests/test_onnx_attention.py -v`

Expected: cross-attention test fails with `ValueError: Opset11MultiheadAttention only supports self-attention`.

- [ ] **Step 3: Project Q, K, and V independently and preserve sequence lengths**

Replace the identity check and combined projection with slices of the packed projection parameters:

```python
batch_size, query_len, _ = query.shape
key_len = key.shape[1]
q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
if self.in_proj_bias is None:
    q_bias = k_bias = v_bias = None
else:
    q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
q = F.linear(query, q_weight, q_bias)
k = F.linear(key, k_weight, k_bias)
v = F.linear(value, v_weight, v_bias)
q = q.reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
k = k.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
v = v.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
```

Use `key_len` for mask reshaping and `query_len` when reshaping the output context.

- [ ] **Step 4: Run adapter parity tests**

Run: `pytest tests/test_onnx_attention.py -v`

Expected: 2 passed.

- [ ] **Step 5: Run complete verification**

Run: `pytest -v`

Expected: all tests pass.

Run: `python -m compileall model onnx tests`

Expected: exit code 0 with no syntax errors.

If a representative checkpoint and configured data path are available, also run the documented export and parity commands:

```bash
python onnx/convert_to_onnx.py --config configs/config.yaml --output /tmp/hww_pcres_regressor.onnx
python onnx/onnxruntime_check.py --config configs/config.yaml --onnx /tmp/hww_pcres_regressor.onnx
```

Expected: ONNX validation succeeds, and ONNX Runtime output matches PyTorch within the checker's configured tolerance.
