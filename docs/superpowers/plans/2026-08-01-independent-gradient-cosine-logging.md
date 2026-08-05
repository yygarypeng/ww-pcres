# Independent Gradient-Cosine Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate epoch-level gradient-cosine metrics when logging is enabled without requiring or activating adaptive loss weights.

**Architecture:** Split side-effect-free cosine calculation from adaptive weight mutation. Retain one detached first batch per epoch whenever either consumer is enabled, then independently log diagnostics and update weights at epoch end.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, unittest/pytest, Jupyter notebook JSON

## Global Constraints

- Keep fixed loss weights unchanged when only `log_loss_gradient_cosines` is enabled.
- Preserve the existing non-Huber loss selection and `grad_cos/{loss}__total` / `grad_cos/{loss}__rest` metric names.
- Perform gradient analysis once per epoch using the first training batch.
- Do not discard valid cosine diagnostics when an adaptive update is degenerate.

---

### Task 1: Decouple Gradient Diagnostics From Weight Updates

**Files:**
- Modify: `model/model.py:273-373`
- Test: `tests/test_model_loss.py`

**Interfaces:**
- Produces: `_compute_loss_gradient_cosines(losses: dict[str, Tensor], total: Tensor) -> dict[str, dict[str, Tensor]]`
- Produces: `_update_adaptive_loss_weights(cosines: dict[str, dict[str, Tensor]]) -> None`
- Maintains: `grad_cos/{loss}__total` and `grad_cos/{loss}__rest`

- [ ] **Step 1: Add failing lifecycle tests**

Add tests that instantiate a small `LightningWBoson`, stub expensive loss calculation, and prove that logging-only mode captures a batch, emits cosine metrics at epoch end, and leaves `loss_weights` unchanged. Add a disabled-mode assertion proving no batch is retained.

```python
def test_gradient_cosines_are_logged_without_adaptive_weights(self):
    model = self._make_model(
        loss_weights={"huber": 1.0, "higgs_mass": 2.0},
        adaptive_loss_weights=False,
        log_loss_gradient_cosines=True,
    )
    batch = (torch.randn(2, 18), torch.randn(2, 10))
    total = torch.tensor(1.0, requires_grad=True)
    losses = {"huber": total, "higgs_mass": total}
    cosines = {
        "higgs_mass": {
            "total": torch.tensor(0.5),
            "rest": torch.tensor(-0.25),
        }
    }
    original_weights = dict(model.loss_weights)

    with patch.object(model, "_compute_batch_losses", return_value=(total, losses)):
        with patch.object(model, "_compute_loss_gradient_cosines", return_value=cosines):
            with patch.object(model, "_log_grad_cosines") as log_cosines:
                model.training_step(batch, 0)
                model.on_train_epoch_end()

    log_cosines.assert_called_once_with(cosines)
    self.assertEqual(model.loss_weights, original_weights)
    self.assertIsNone(model._gradient_analysis_batch)

def test_training_does_not_retain_batch_when_gradient_analysis_is_disabled(self):
    model = self._make_model(
        adaptive_loss_weights=False,
        log_loss_gradient_cosines=False,
    )
    batch = (torch.randn(2, 18), torch.randn(2, 10))
    total = torch.tensor(1.0, requires_grad=True)

    with patch.object(model, "_compute_batch_losses", return_value=(total, {"huber": total})):
        model.training_step(batch, 0)

    self.assertIsNone(model._gradient_analysis_batch)
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run: `pytest -q tests/test_model_loss.py -k "gradient_cosines_are_logged_without_adaptive_weights or gradient_analysis_is_disabled"`

Expected: FAIL because logging-only mode does not capture the batch and `_gradient_analysis_batch` does not yet exist.

- [ ] **Step 3: Extract cosine calculation and narrow adaptive mutation**

In `model/model.py`, move gradient-vector and cosine calculation into `_compute_loss_gradient_cosines`. Change `_update_adaptive_loss_weights` to consume the calculated cosine dictionary and only normalize/apply adaptive weights. Validate cosine finiteness during calculation, but let a zero adaptive raw-weight sum skip mutation without removing diagnostics.

```python
def _compute_loss_gradient_cosines(self, losses, total):
    names = [name for name in self.adaptive_loss_names if name in losses]
    parameters = tuple(p for p in self.parameters() if p.requires_grad)
    if not names or not parameters:
        return {}

    total_grad = self._loss_grad_vector(total, parameters)
    cosines = {}
    for name in names:
        grad = self._loss_grad_vector(losses[name], parameters)
        weight = self.loss_weights.get(name, 0.0)
        rest_grad = total_grad - weight * grad
        cos_total = torch.nn.functional.cosine_similarity(grad, total_grad, dim=0, eps=1.0e-6)
        cos_rest = torch.nn.functional.cosine_similarity(grad, rest_grad, dim=0, eps=1.0e-6)
        if not torch.isfinite(cos_total).item() or not torch.isfinite(cos_rest).item():
            return {}
        cosines[name] = {"total": cos_total.detach(), "rest": cos_rest.detach()}
    return cosines

def _update_adaptive_loss_weights(self, cosines):
    if not cosines or self.adaptive_loss_budget <= 0.0:
        return
    raw_weights = [torch.clamp(1.0 - cosines[name]["total"], min=0.0) for name in cosines]
    raw_sum = torch.stack(raw_weights).sum()
    if not torch.isfinite(raw_sum).item() or raw_sum.item() <= 0.0:
        return
    # Preserve the existing smoothing and budget-conserving update.
```

- [ ] **Step 4: Update epoch lifecycle gates**

Rename `_adaptive_batch` to `_gradient_analysis_batch`. Capture the first batch when either option is enabled, compute cosines at epoch end, log according to the existing logging gate, and mutate weights only under the adaptive gate.

```python
needs_gradient_analysis = self.adaptive_loss_weights or self.log_loss_gradient_cosines
if needs_gradient_analysis and self._gradient_analysis_batch is None:
    self._gradient_analysis_batch = (x.detach().cpu(), y.detach().cpu())

cosines = self._compute_loss_gradient_cosines(losses, total)
if self.adaptive_loss_weights:
    self._update_adaptive_loss_weights(cosines)
self._log_grad_cosines(cosines)
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_model_loss.py tests/test_train_overrides.py`

Expected: PASS.

### Task 2: Correct Empty-Metric Guidance

**Files:**
- Modify: `notebooks/visualize.ipynb` cell `grad-cos-plot`

**Interfaces:**
- Consumes: populated `grad_cos/*` columns from Task 1
- Produces: accurate guidance for legacy empty-column runs

- [ ] **Step 1: Replace the misleading diagnostic text**

Change the all-empty message to explain that older training code tied diagnostics to adaptive weighting even when logging was enabled, and direct users to rerun with independent logging.

```python
print("No gradient-cosine values were emitted. Older training code required adaptive_loss_weights: true even when log_loss_gradient_cosines was enabled.")
print("Rerun with the current independent gradient-cosine logging implementation.")
```

- [ ] **Step 2: Validate notebook structure and message**

Run: `python -m json.tool notebooks/visualize.ipynb >/dev/null`

Expected: exit code 0.

Run: `rg -n "Older training code required adaptive_loss_weights" notebooks/visualize.ipynb`

Expected: one source-cell match.

### Task 3: Regression Verification

**Files:**
- Verify: `model/model.py`
- Verify: `tests/test_model_loss.py`
- Verify: `tests/test_train_overrides.py`
- Verify: `notebooks/visualize.ipynb`

**Interfaces:**
- Consumes: completed behavior and documentation changes
- Produces: verification evidence

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect the final diff**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git diff -- model/model.py tests/test_model_loss.py notebooks/visualize.ipynb docs/superpowers/specs/2026-08-01-independent-gradient-cosine-logging-design.md docs/superpowers/plans/2026-08-01-independent-gradient-cosine-logging.md`

Expected: only the approved diagnostic decoupling, tests, notebook guidance, and design documentation.
