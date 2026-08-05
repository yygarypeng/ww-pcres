# Remove Learned-Query Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `learned_query` as a supported pooling mode and explicitly reject checkpoints that contain its prediction-distorting offsets.

**Architecture:** Keep `infer_pooling_mode_from_checkpoint` as the single checkpoint compatibility boundary, but make it reject learned-query metadata or parameters. Restrict `WBosonRegressor` and the training CLI to `flatten` and `lepton`, with lepton pooling always using the two refined lepton tokens directly.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, argparse, unittest, pytest

## Global Constraints

- Preserve `flatten` and `lepton` model behavior.
- Do not silently reinterpret an existing learned-query checkpoint as a lepton checkpoint.
- Do not change the cross-attention block, decoder, regression heads, physics reconstruction, output schema, or generic ablation evaluator.
- Preserve unrelated uncommitted work already present in the worktree.
- Do not create git commits unless the user explicitly requests them.

---

### Task 1: Reject Learned-Query Checkpoints

**Files:**
- Modify: `tests/test_pooling_ablation.py:55-70`
- Modify: `model/model.py:14-25`

**Interfaces:**
- Consumes: `infer_pooling_mode_from_checkpoint(checkpoint_path) -> str` and PyTorch checkpoint dictionaries.
- Produces: supported mode inference for `flatten` and `lepton`, plus `ValueError` for learned-query metadata or `model.w_query_offsets` state.

- [ ] **Step 1: Replace learned-query inference coverage with rejection tests**

Keep the existing supported-mode inference test limited to `flatten` and `lepton`:

```python
def test_infers_supported_pooling_from_state_when_hyperparameter_is_missing(self):
    cases = {
        "flatten": {"model.trunk.0.weight": torch.zeros(1)},
        "lepton": {"model.event_pool.mha.in_proj_weight": torch.zeros(1)},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for expected, state_dict in cases.items():
            with self.subTest(expected=expected):
                path = Path(temp_dir) / f"{expected}.ckpt"
                torch.save({"hyper_parameters": {}, "state_dict": state_dict}, path)
                self.assertEqual(infer_pooling_mode_from_checkpoint(path), expected)
```

Add one test covering both learned-query identification mechanisms:

```python
def test_rejects_learned_query_checkpoints(self):
    checkpoints = {
        "metadata": {
            "hyper_parameters": {"pooling_mode": "learned_query"},
            "state_dict": {},
        },
        "state": {
            "hyper_parameters": {},
            "state_dict": {"model.w_query_offsets": torch.zeros(2, 8)},
        },
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for label, checkpoint in checkpoints.items():
            with self.subTest(label=label):
                path = Path(temp_dir) / f"{label}.ckpt"
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "learned_query.*not supported"):
                    infer_pooling_mode_from_checkpoint(path)
```

- [ ] **Step 2: Run the focused checkpoint tests and verify the new test fails**

Run: `pytest tests/test_pooling_ablation.py::LegacyPoolingInferenceTest -v`

Expected: the supported inference cases pass, while both learned-query subtests fail because the helper currently returns `learned_query`.

- [ ] **Step 3: Implement centralized learned-query checkpoint rejection**

Replace the start of `infer_pooling_mode_from_checkpoint` with:

```python
def infer_pooling_mode_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_mode = checkpoint.get("hyper_parameters", {}).get("pooling_mode")
    state_dict = checkpoint.get("state_dict", {})
    if saved_mode == "learned_query" or "model.w_query_offsets" in state_dict:
        raise ValueError("learned_query checkpoints are not supported")
    if saved_mode is not None:
        return saved_mode

    if any(name.startswith("model.event_pool.") for name in state_dict):
        return "lepton"
    return "flatten"
```

- [ ] **Step 4: Run checkpoint and evaluator tests**

Run: `pytest tests/test_pooling_ablation.py -v`

Expected: all tests in `tests/test_pooling_ablation.py` pass.

### Task 2: Remove The Model And CLI Mode

**Files:**
- Modify: `tests/test_attention.py:72-151`
- Modify: `tests/test_train_overrides.py:8-27`
- Modify: `model/model.py:55-75,132-149`
- Modify: `train/train.py:279-304`

**Interfaces:**
- Consumes: `WBosonRegressor(..., pooling_mode: str)` and `parse_args()`.
- Produces: only `flatten` and `lepton` model modes; `--pooling-mode` accepts those same two values.

- [ ] **Step 1: Change model tests to require direct lepton queries and reject learned-query mode**

In `test_aggregation_uses_refined_leptons_as_event_dependent_queries`, add assertions that no offset parameter exists:

```python
self.assertFalse(hasattr(model, "w_query_offsets"))
self.assertNotIn("w_query_offsets", dict(model.named_parameters()))
```

Delete `test_learned_queries_are_offsets_from_corresponding_leptons` and `test_learned_query_offsets_receive_gradients`. Change the attention test to construct the supported mode:

```python
def test_cross_attention_pooling_exposes_per_head_weights(self):
    model = self.make_model(pooling_mode="lepton")

    aggregated, attention = model.global_feature_aggregation(
        torch.randn(3, 18),
        return_attention=True,
    )

    self.assertEqual(aggregated.shape, (3, 16))
    self.assertEqual(attention.shape, (3, 2, 2, model.num_tokens))
```

Extend the unsupported-mode test to cover the removed mode explicitly:

```python
def test_rejects_unknown_and_learned_query_pooling_modes(self):
    for pooling_mode in ("unknown", "learned_query"):
        with self.subTest(pooling_mode=pooling_mode):
            with self.assertRaisesRegex(ValueError, "Unsupported pooling_mode"):
                self.make_model(pooling_mode=pooling_mode)
```

- [ ] **Step 2: Run the focused model tests and verify learned-query rejection fails**

Run: `pytest tests/test_attention.py::LeptonAnchoredPoolingTest -v`

Expected: the learned-query subtest fails because `WBosonRegressor` still accepts that mode.

- [ ] **Step 3: Remove offset creation and addition from the regressor**

Restrict validation:

```python
self.pooling_mode = pooling_mode
if pooling_mode not in {"flatten", "lepton"}:
    raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")
```

Keep cross-attention setup for every non-flatten mode, but delete the complete block that creates `self.w_query_offsets`. In `global_feature_aggregation`, keep only direct lepton queries:

```python
queries = context[:, :2]
if return_attention:
    pooled, attention = self.event_pool(
        queries,
        context,
        key_padding_mask=key_mask,
        return_attention=True,
    )
else:
    pooled = self.event_pool(queries, context, key_padding_mask=key_mask)
    attention = None
```

- [ ] **Step 4: Change override coverage and CLI choices to supported modes**

In `tests/test_train_overrides.py`, use `flatten` as the override and expected value:

```python
args = Namespace(
    pooling_mode="flatten",
    saved_path="outputs/query-115",
    seed=115,
    epochs=12,
    max_events_per_category=5000,
)

updated = apply_cli_overrides(config, args)

self.assertEqual(updated["parameters"]["pooling_mode"], "flatten")
```

Restrict the parser choice in `train/train.py`:

```python
parser.add_argument(
    "--pooling-mode",
    choices=("flatten", "lepton"),
    help="Override parameters.pooling_mode for pooling ablations",
)
```

- [ ] **Step 5: Run model and training-interface tests**

Run: `pytest tests/test_attention.py tests/test_train_overrides.py -v`

Expected: all tests in both files pass.

### Task 3: Remove User-Facing References And Verify

**Files:**
- Modify: `README.md:86-112`

**Interfaces:**
- Consumes: the supported `flatten` and `lepton` CLI modes.
- Produces: README examples that expose only supported modes.

- [ ] **Step 1: Update pooling-ablation documentation**

Change the mode count and delete the learned-query bullet and commands. The resulting section must contain:

````markdown
The event representation supports two pooling modes:

- `flatten`: preserve every refined event token, matching the pre-cross-attention representation.
- `lepton`: use the two refined lepton tokens as cross-attention queries.

Run both modes with identical settings and separate output directories:

```bash
python train/train.py --pooling-mode flatten --seed 114 --saved-path outputs/ablation-flatten-114
python train/train.py --pooling-mode lepton --seed 114 --saved-path outputs/ablation-lepton-114
```
````

The evaluation example must retain only:

```bash
python scripts/evaluate_pooling_ablation.py \
  --run flatten=outputs/ablation-flatten-114 \
  --run lepton=outputs/ablation-lepton-114 \
  --output outputs/pooling-ablation.csv
```

- [ ] **Step 2: Confirm no supported learned-query path remains**

Run: `rg -n "learned_query|learned-query" model train scripts onnx tests README.md configs`

Expected: matches occur only in the explicit checkpoint/model rejection code and tests; no model implementation, CLI choice, config, evaluator example, or README feature description supports the mode.

- [ ] **Step 3: Run the complete test suite**

Run: `pytest -v`

Expected: all tests pass with no failures.

- [ ] **Step 4: Inspect the final scoped diff**

Run: `git diff -- model/model.py train/train.py tests/test_attention.py tests/test_pooling_ablation.py tests/test_train_overrides.py README.md docs/superpowers/specs/2026-07-31-remove-learned-query-pooling-design.md docs/superpowers/plans/2026-07-31-remove-learned-query-pooling.md`

Expected: the diff contains only the approved mode removal, checkpoint guard, tests, documentation, spec, and plan; it preserves unrelated existing edits in shared files.
