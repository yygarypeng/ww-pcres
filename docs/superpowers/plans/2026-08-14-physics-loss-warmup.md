# Physics Loss Warm-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train with only W-four-vector Huber through epoch 19, then activate all configured physics losses at epoch 20.

**Architecture:** Use `physics_start_epoch` to gate every non-Huber loss during training. Keep evaluation on configured weights so checkpoint monitoring has a stable objective, and migrate the retired `mmd_start_epoch` name when loading older artifacts.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, unittest/pytest, YAML.

## Global Constraints

- Use `physics_start_epoch` as the single activation control.
- Do not add a ramp, scheduler state, or additional optimizer.
- Keep batch size 512, learning rate 1e-5, angular MMD weight 240, and existing kernels.
- Set the active Higgs loss weight to 30 and activation epoch to 20.

---

### Task 1: Physics Warm-Up Semantics

**Files:**
- Modify: `tests/test_model_loss.py`
- Modify: `model/model.py`

**Interfaces:**
- Consumes: `LightningWBoson._effective_loss_weights()` and `_compute_losses(...)`.
- Produces: training-only effective weights that retain only `huber` before `physics_start_epoch`.

- [ ] **Step 1: Write failing tests**

Extend the warm-up fixture with all configured losses. Assert that a training
model before activation computes only `huber`, an evaluation model computes all
losses, and all training losses activate exactly at the configured epoch.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_model_loss.py -q -k "warmup or start_epoch"`

Expected: failures showing non-MMD physics losses remain active during training
and evaluation incorrectly receives warm-up weights.

- [ ] **Step 3: Implement minimal behavior**

In `model/model.py`, define the warm-up set as every supported loss except
`huber`. Apply the zeroed weights only while `self.training` and
`current_epoch < physics_start_epoch`. Preserve configured weights in evaluation.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_model_loss.py -q -k "warmup or start_epoch"`

Expected: all selected tests pass.

### Task 2: Configuration And Documentation

**Files:**
- Modify: `configs/config.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `parameters.physics_start_epoch` and `parameters.loss_weights.higgs_mass`.
- Produces: active activation epoch 20 and Higgs weight 30.

- [ ] **Step 1: Update active configuration**

Set `physics_start_epoch: 20` and `higgs_mass: 30.0`; leave all other requested
training settings unchanged.

- [ ] **Step 2: Update documentation**

Document that the option gates every non-Huber physics loss during training and
that validation/test always evaluate configured weights.

- [ ] **Step 3: Run focused verification**

Run: `pytest tests/test_model_loss.py tests/test_train_overrides.py -q`

Expected: all tests pass.

### Task 3: Final Verification

**Files:**
- Verify only; no production files added.

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: test and diff evidence for completion.

- [ ] **Step 1: Run full test suite**

Run: `pytest -q`

Expected: all project tests pass, aside from any explicitly identified
pre-existing failures.

- [ ] **Step 2: Inspect final diff**

Run: `git diff -- model/model.py tests/test_model_loss.py README.md`

Also inspect the ignored active `configs/config.yaml` values directly. Confirm
that every changed line belongs to the approved warm-up behavior.
