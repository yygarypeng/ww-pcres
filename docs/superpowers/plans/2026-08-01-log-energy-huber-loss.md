# Log-Energy Huber Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply `log1p` to both W-boson energy components in the standardized four-vector Huber loss and compute the matching energy scale in transformed space.

**Architecture:** Represent the first eight values as two `[px, py, pz, E]` vectors during loss computation so one transformation applies consistently to both energies. Fit the existing four shared component scales after applying the same transformation to training energies.

**Tech Stack:** Python, NumPy, PyTorch, unittest/pytest

## Global Constraints

- Keep momentum components in their original linear space.
- Transform energy with `log1p` in both the loss and training-scale calculation.
- Keep scales shared between the two W slots and clamped to positive float32 epsilon.
- Do not commit changes unless the user explicitly requests a commit.

---

### Task 1: Transform Energy Residuals and Scales

**Files:**
- Modify: `tests/test_model_loss.py:12-43`
- Modify: `model/losses.py:160-163`
- Modify: `train/train.py:109-112`

**Interfaces:**
- Consumes: `standardized_fourvec_huber_loss(y_true, y_pred, component_scales)` with targets ending in at least eight values, predictions ending in eight values, and four shared component scales.
- Produces: A scalar mean Huber loss over two transformed four-vectors; `compute_w_fourvec_scales(targets)` continues returning a float32 NumPy array of shape `(4,)`.

- [ ] **Step 1: Write failing tests for transformed scales and both energy slots**

Replace the scale expectation and loss test in `tests/test_model_loss.py`, and add a zero-energy test:

```python
def test_compute_scales_pools_w_slots_by_component(self):
    targets = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0, 3.0, 30.0, 300.0, 3000.0, 0.0, 0.0],
            [5.0, 50.0, 500.0, 5000.0, 7.0, 70.0, 700.0, 7000.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    scales = compute_w_fourvec_scales(targets)

    transformed = targets[:, :8].reshape(-1, 4).copy()
    transformed[:, 3] = np.log1p(transformed[:, 3])
    expected = np.std(transformed, axis=0)
    np.testing.assert_allclose(scales, expected)

def test_loss_applies_log1p_to_both_energy_slots(self):
    truth = torch.tensor([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]])
    prediction = torch.tensor([[1.0, 2.0, 3.0, 7.0, 1.0, 2.0, 3.0, 15.0]])
    scales = torch.ones(4)

    loss = standardized_fourvec_huber_loss(truth, prediction, scales)

    log_two = torch.log(torch.tensor(2.0))
    standardized_residual = torch.tensor(
        [[[1.0, 2.0, 3.0, log_two], [1.0, 2.0, 3.0, log_two]]]
    )
    expected = F.huber_loss(standardized_residual, torch.zeros_like(standardized_residual))
    torch.testing.assert_close(loss, expected)

def test_loss_is_finite_for_zero_energy(self):
    truth = torch.zeros((1, 10))
    prediction = torch.zeros((1, 8))

    loss = standardized_fourvec_huber_loss(truth, prediction, torch.ones(4))

    self.assertTrue(torch.isfinite(loss))
```

- [ ] **Step 2: Run the focused tests and verify the new expectations fail**

Run: `pytest tests/test_model_loss.py -v`

Expected: `test_compute_scales_pools_w_slots_by_component` and `test_loss_applies_log1p_to_both_energy_slots` fail because energy is still linear. The pre-existing tests and zero-energy test pass.

- [ ] **Step 3: Apply `log1p` consistently in scale fitting and loss computation**

Update `compute_w_fourvec_scales` in `train/train.py`:

```python
def compute_w_fourvec_scales(targets):
    w_fourvecs = targets[:, :8].reshape(-1, 4).copy()
    w_fourvecs[:, 3] = np.log1p(w_fourvecs[:, 3])
    scales = np.std(w_fourvecs, axis=0)
    return np.maximum(scales, np.finfo(np.float32).eps).astype(np.float32)
```

Update `standardized_fourvec_huber_loss` in `model/losses.py`:

```python
def standardized_fourvec_huber_loss(y_true, y_pred, component_scales):
    true_fourvecs = y_true[..., :8].reshape(*y_true.shape[:-1], 2, 4)
    pred_fourvecs = y_pred.reshape(*y_pred.shape[:-1], 2, 4)
    true_transformed = torch.cat(
        [true_fourvecs[..., :3], torch.log1p(true_fourvecs[..., 3:4])],
        dim=-1,
    )
    pred_transformed = torch.cat(
        [pred_fourvecs[..., :3], torch.log1p(pred_fourvecs[..., 3:4])],
        dim=-1,
    )
    residual = (pred_transformed - true_transformed) / component_scales
    return F.huber_loss(residual, torch.zeros_like(residual))
```

- [ ] **Step 4: Run focused verification**

Run: `pytest tests/test_model_loss.py -v`

Expected: all tests in `tests/test_model_loss.py` pass.

- [ ] **Step 5: Run full regression verification**

Run: `pytest -q`

Expected: the full test suite passes with no failures.

- [ ] **Step 6: Inspect the final patch**

Run: `git diff -- model/losses.py train/train.py tests/test_model_loss.py docs/superpowers/specs/2026-08-01-log-energy-huber-loss-design.md docs/superpowers/plans/2026-08-01-log-energy-huber-loss.md`

Expected: only the approved log-energy loss, matching scale calculation, tests, and documentation are present. Leave all changes uncommitted unless the user explicitly asks for a commit.
