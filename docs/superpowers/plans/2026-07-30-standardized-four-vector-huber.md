# Standardized Four-Vector Huber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize W four-vector residuals by training-set component scales before applying the main Huber loss.

**Architecture:** Compute four pooled training scales for `(px, py, pz, E)` across both W slots, store them in the Lightning checkpoint as a buffer, and broadcast them over the eight predicted W components. Keep model outputs and all other physics calculations in GeV.

**Tech Stack:** Python, NumPy, PyTorch, PyTorch Lightning, unittest/pytest

## Global Constraints

- Fit scales from the training targets only.
- Share each component scale between W+ and W−.
- Do not change optimizer settings or auxiliary losses.
- Preserve existing unrelated worktree changes.

---

### Task 1: Standardized Four-Vector Loss

**Files:**
- Create: `tests/test_model_loss.py`
- Modify: `model/model.py`
- Modify: `train/train.py`

**Interfaces:**
- Consumes: training targets with shape `(n_events, 10)`.
- Produces: `compute_w_fourvec_scales(targets) -> np.ndarray` with shape `(4,)`.
- Produces: `LightningWBoson(..., w_fourvec_scales)` storing a checkpointed tensor buffer.

- [ ] **Step 1: Write failing tests**

Test pooled W-component scales and verify that `_compute_losses` applies Huber to standardized residuals while leaving predictions in GeV.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_model_loss.py -v`

Expected: failure because `compute_w_fourvec_scales` and `w_fourvec_scales` do not exist.

- [ ] **Step 3: Implement the minimal production change**

Compute scales from `Y_train[:, :8].reshape(-1, 2, 4)`, pass them into `LightningWBoson`, register a positive clamped buffer, and calculate the main loss from `(y_pred - y[..., :8]) / repeated_scales`.

- [ ] **Step 4: Verify focused and full tests**

Run: `python -m pytest tests/test_model_loss.py -v`

Expected: all focused tests pass.

Run: `python -m pytest -v`

Expected: all repository tests pass.

- [ ] **Step 5: Verify a model checkpoint round trip**

Run the focused test that saves and reloads the model state dictionary and confirms the scale buffer is preserved.

Expected: the restored scale tensor exactly matches the original tensor.
