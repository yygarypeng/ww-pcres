# Periodic Local-MMD Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Condition angular local MMD on normalized `m_ll`, `deta_ll`, and periodic sine/cosine encodings of both reconstructed azimuthal differences.

**Architecture:** The data layer transforms the four physical observables into six numerical features and fits their statistics on `X_train`. The regressor independently performs the same Torch transformation on each batch, normalizes it with fixed buffers, and exposes it through `aux["cond"]`; predictor inputs remain unchanged.

**Tech Stack:** Python, NumPy, scikit-learn `StandardScaler`, PyTorch, PyTorch Lightning, unittest/pytest.

## Global Constraints

- Preserve the existing 22-column predictor input layout and high-level embedding.
- Fit all conditioning statistics on the training split only.
- Represent each `dphi` with both sine and cosine before computing Euclidean condition distances.
- Keep the conditioning path deterministic and free of batch-dependent normalization.
- Do not retune loss weights, MMD bandwidths, or model architecture.
- Do not create commits unless the user explicitly requests them.

## File Map

- `data/load_data.py`: NumPy conditioning transformation and training-statistics calculation.
- `model/model.py`: Torch conditioning transformation, normalization buffers, constructor plumbing, and input validation.
- `model/losses.py`: Reject angular local MMD when required conditioning observables are unavailable.
- `train/train.py`: Compute conditioning statistics from `X_train` and pass them into the model.
- `tests/test_load_data.py`: Transformation and statistics tests.
- `tests/test_attention.py`: Regressor auxiliary-condition shape, values, and periodic-boundary tests.
- `tests/test_model_loss.py`: End-to-end condition propagation into `compute_local_mmd` and missing-condition error test.

---

### Task 1: Training-Only Conditioning Statistics

**Files:**
- Modify: `data/load_data.py:50-62`
- Modify: `tests/test_load_data.py`

**Interfaces:**
- Produces: `mmd_condition_features(features: np.ndarray) -> np.ndarray` with final dimension 6.
- Produces: `compute_mmd_condition_stats(train_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]`.
- Feature order: `[m_ll, deta_ll, sin(dphi_ll), cos(dphi_ll), sin(dphi_llmet), cos(dphi_llmet)]`.

- [ ] **Step 1: Write failing transformation and statistics tests**

Update imports in `tests/test_load_data.py` and add:

```python
from data.load_data import (
    _valid_truth_w_rows,
    compute_mmd_condition_stats,
    mmd_condition_features,
)


class MMDConditionFeaturesTest(unittest.TestCase):
    def test_builds_periodic_condition_features_in_documented_order(self):
        features = np.zeros((2, 22), dtype=np.float32)
        features[:, 18:22] = [
            [10.0, -0.5, 0.0, np.pi / 2.0],
            [20.0, 0.5, np.pi, -np.pi / 2.0],
        ]

        condition = mmd_condition_features(features)

        expected = np.array([
            [10.0, -0.5, 0.0, 1.0, 1.0, 0.0],
            [20.0, 0.5, 0.0, -1.0, -1.0, 0.0],
        ])
        np.testing.assert_allclose(condition, expected, atol=1.0e-6)

    def test_statistics_are_fitted_to_transformed_training_features(self):
        features = np.zeros((3, 22), dtype=np.float32)
        features[:, 18:22] = [
            [10.0, -1.0, -0.5, -1.0],
            [20.0, 0.0, 0.0, 0.0],
            [30.0, 1.0, 0.5, 1.0],
        ]

        mean, scale = compute_mmd_condition_stats(features)
        transformed = mmd_condition_features(features)

        np.testing.assert_allclose(mean, transformed.mean(axis=0))
        np.testing.assert_allclose(scale, transformed.std(axis=0))

    def test_condition_features_require_all_four_observables(self):
        with self.assertRaisesRegex(ValueError, "at least 22 features"):
            mmd_condition_features(np.zeros((2, 21), dtype=np.float32))
```

- [ ] **Step 2: Run tests and confirm the missing imports fail**

Run: `pytest tests/test_load_data.py -v`

Expected: collection error because `mmd_condition_features` and `compute_mmd_condition_stats` do not exist.

- [ ] **Step 3: Implement the NumPy transformation and statistics**

Add above `compute_standardization_stats` in `data/load_data.py`:

```python
def mmd_condition_features(features):
    features = np.asarray(features)
    if features.shape[-1] < 22:
        raise ValueError(
            f"MMD conditioning requires at least 22 features, got {features.shape[-1]}"
        )

    m_ll = features[..., 18:19]
    deta_ll = features[..., 19:20]
    dphi_ll = features[..., 20:21]
    dphi_llmet = features[..., 21:22]
    return np.concatenate([
        m_ll,
        deta_ll,
        np.sin(dphi_ll),
        np.cos(dphi_ll),
        np.sin(dphi_llmet),
        np.cos(dphi_llmet),
    ], axis=-1)


def compute_mmd_condition_stats(train_obj):
    condition = mmd_condition_features(train_obj)
    scaler = StandardScaler().fit(condition)
    return scaler.mean_, scaler.scale_
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_load_data.py -v`

Expected: all tests in the file pass.

---

### Task 2: Regressor Periodic Condition Path

**Files:**
- Modify: `model/model.py:13-31,108-157`
- Modify: `model/losses.py:213-250`
- Modify: `tests/test_attention.py`
- Modify: `tests/test_model_loss.py`

**Interfaces:**
- Consumes: six-element `mmd_cond_mean_train` and `mmd_cond_scale_train` arrays produced by Task 1.
- Produces: `WBosonRegressor._mmd_condition(x: torch.Tensor) -> torch.Tensor`.
- Produces: `aux["cond"]` with shape `[batch_size, 6]` for inputs with at least 22 columns.
- For fewer than 22 input columns, produces an empty condition with shape `[batch_size, 0]`.

- [ ] **Step 1: Write failing regressor condition tests**

Extend `FlattenedAggregationTest.make_model` in `tests/test_attention.py`:

```python
    @staticmethod
    def make_model(input_dim=18, mmd_cond_mean=None, mmd_cond_scale=None):
        return WBosonRegressor(
            input_dim=input_dim,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(input_dim, dtype=np.float32),
            std_scale_train=np.ones(input_dim, dtype=np.float32),
            mmd_cond_mean_train=mmd_cond_mean,
            mmd_cond_scale_train=mmd_cond_scale,
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )
```

Add these tests:

```python
    def test_aux_condition_uses_normalized_periodic_features(self):
        mean = np.array([10.0, 1.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        scale = np.array([2.0, 4.0, 0.5, 0.5, 0.25, 0.25], dtype=np.float32)
        model = self.make_model(22, mean, scale).eval()
        x = torch.zeros(2, 22)
        x[:, 18:22] = torch.tensor([
            [12.0, 5.0, 0.0, torch.pi / 2.0],
            [8.0, -3.0, torch.pi, -torch.pi / 2.0],
        ])

        with torch.no_grad():
            _, aux = model(x, return_aux=True)

        raw_condition = torch.stack([
            x[:, 18],
            x[:, 19],
            torch.sin(x[:, 20]),
            torch.cos(x[:, 20]),
            torch.sin(x[:, 21]),
            torch.cos(x[:, 21]),
        ], dim=-1)
        expected = (raw_condition - torch.from_numpy(mean)) / torch.from_numpy(scale)
        torch.testing.assert_close(aux["cond"], expected)
        self.assertEqual(aux["cond"].shape, (2, 6))

    def test_periodic_condition_is_continuous_across_pi_boundary(self):
        model = self.make_model(
            22,
            np.zeros(6, dtype=np.float32),
            np.ones(6, dtype=np.float32),
        )
        x = torch.zeros(2, 22)
        x[:, 20] = torch.tensor([-torch.pi + 1.0e-4, torch.pi - 1.0e-4])

        condition = model._mmd_condition(x)

        self.assertLess(torch.linalg.vector_norm(condition[0] - condition[1]), 5.0e-4)

    def test_model_without_high_level_features_returns_empty_condition(self):
        model = self.make_model(18).eval()

        with torch.no_grad():
            _, aux = model(torch.randn(2, 18), return_aux=True)

        self.assertEqual(aux["cond"].shape, (2, 0))
```

- [ ] **Step 2: Run tests and verify constructor/method failures**

Run: `pytest tests/test_attention.py -v`

Expected: failures because the constructor arguments and `_mmd_condition` do not exist, and `aux["cond"]` still contains raw high-level columns.

- [ ] **Step 3: Implement condition buffers and transformation**

Add optional arguments after input-standardization arguments in both model constructors:

```python
mmd_cond_mean_train=None,
mmd_cond_scale_train=None,
```

In `WBosonRegressor.__init__`, register a dedicated normalizer:

```python
        if (mmd_cond_mean_train is None) != (mmd_cond_scale_train is None):
            raise ValueError("MMD condition mean and scale must be provided together")
        if mmd_cond_mean_train is None:
            mmd_cond_mean_train = torch.zeros(6, dtype=torch.float32)
            mmd_cond_scale_train = torch.ones(6, dtype=torch.float32)
        if len(mmd_cond_mean_train) != 6 or len(mmd_cond_scale_train) != 6:
            raise ValueError("MMD condition mean and scale must each contain 6 values")
        self.cond_norm = Standardization(mmd_cond_mean_train, mmd_cond_scale_train)
```

Add to `WBosonRegressor` before `forward`:

```python
    def _mmd_condition(self, x):
        if x.shape[-1] < 22:
            return x.new_empty((*x.shape[:-1], 0))

        condition = torch.stack([
            x[..., 18],
            x[..., 19],
            torch.sin(x[..., 20]),
            torch.cos(x[..., 20]),
            torch.sin(x[..., 21]),
            torch.cos(x[..., 21]),
        ], dim=-1)
        return self.cond_norm(condition)
```

Replace the existing auxiliary condition expression with:

```python
                "cond": self._mmd_condition(x),
```

Pass the two new arguments from `LightningWBoson` into `WBosonRegressor`.

- [ ] **Step 4: Add a clear missing-condition failure**

At the start of `angular_loss_mmd` in `model/losses.py`, add:

```python
    if cond.shape[-1] == 0:
        raise ValueError("angular local MMD requires the four high-level conditioning features")
```

Add to `tests/test_model_loss.py`:

```python
    def test_angular_mmd_requires_high_level_condition_features(self):
        model = LightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18, dtype=np.float32),
            std_scale_train=np.ones(18, dtype=np.float32),
            loss_weights={"huber": 0.0, "angular_loss_mmd": 1.0},
        )

        with self.assertRaisesRegex(ValueError, "requires the four high-level"):
            model._compute_batch_losses(torch.randn(2, 18), torch.randn(2, 10))
```

- [ ] **Step 5: Run focused model tests**

Run: `pytest tests/test_attention.py tests/test_model_loss.py -v`

Expected: all tests in both files pass.

---

### Task 3: Training Wiring and End-to-End Propagation

**Files:**
- Modify: `train/train.py:112-139,182-202,242-271`
- Modify: `tests/test_model_loss.py`

**Interfaces:**
- Consumes: `compute_mmd_condition_stats(X_train)` from Task 1.
- Consumes: `mmd_cond_mean_train` and `mmd_cond_scale_train` constructor arguments from Task 2.
- `build_datamodule` returns `(dm, input_dim, standardization, mmd_condition_standardization, w_fourvec_scales)`.

- [ ] **Step 1: Write an end-to-end propagation test**

Import `patch` and add a test using a 22-feature model with nontrivial condition statistics. Build `x`, obtain `aux = model(x, return_aux=True)`, use timelike handcrafted W vectors so all rows pass the rest-frame mask, and patch `model.losses.compute_local_mmd`:

```python
    def test_normalized_periodic_condition_reaches_local_mmd(self):
        mean = np.arange(6, dtype=np.float32)
        scale = np.arange(1, 7, dtype=np.float32)
        model = LightningWBoson(
            input_dim=22,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(22, dtype=np.float32),
            std_scale_train=np.ones(22, dtype=np.float32),
            mmd_cond_mean_train=mean,
            mmd_cond_scale_train=scale,
            loss_weights={"huber": 0.0, "angular_loss_mmd": 1.0},
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        ).eval()
        x = torch.zeros(4, 22)
        x[:, 18:22] = torch.tensor([
            [10.0, -1.0, -0.4, -0.7],
            [20.0, -0.5, -0.2, -0.3],
            [30.0, 0.5, 0.2, 0.3],
            [40.0, 1.0, 0.4, 0.7],
        ])
        _, aux = model(x, return_aux=True)
        w0 = torch.tensor([30.0, 5.0, 40.0, 100.0])
        w1 = torch.tensor([-20.0, 15.0, -30.0, 90.0])
        prediction = torch.cat([w0, w1]).repeat(4, 1)
        target_w = torch.cat([w0 + torch.tensor([1.0, 0.0, 0.0, 0.0]), w1]).repeat(4, 1)
        target = torch.cat([target_w, torch.zeros(4, 2)], dim=-1)
        captured = {}

        def capture_local_mmd(pred_angles, true_angles, cond, **kwargs):
            captured["cond"] = cond.detach().clone()
            return pred_angles.sum() * 0.0

        with patch("model.losses.compute_local_mmd", side_effect=capture_local_mmd) as local_mmd:
            model._compute_losses(x, target, prediction, aux["cond"], aux)

        self.assertEqual(local_mmd.call_count, 1)
        torch.testing.assert_close(captured["cond"], aux["cond"])
        self.assertEqual(captured["cond"].shape, (4, 6))
```

- [ ] **Step 2: Run the propagation test**

Run: `pytest tests/test_model_loss.py::StandardizedFourVectorHuberTest::test_normalized_periodic_condition_reaches_local_mmd -v`

Expected: pass after Task 2, proving the model-to-loss boundary independently of training wiring.

- [ ] **Step 3: Wire training statistics into model construction**

In `build_datamodule`, compute and return:

```python
    mmd_condition_standardization = data.compute_mmd_condition_stats(X_train)
    return (
        dm,
        X_train.shape[1],
        standardization,
        mmd_condition_standardization,
        w_fourvec_scales,
    )
```

Update `run_training` to accept `mmd_condition_standardization` after `standardization`, unpack it, and pass:

```python
        mmd_cond_mean_train=mmd_cond_mean_train,
        mmd_cond_scale_train=mmd_cond_scale_train,
```

Update `main` to unpack the fifth return value and pass the new statistics to `run_training`.

- [ ] **Step 4: Run all tests**

Run: `pytest -q`

Expected: the entire suite passes with zero failures.

- [ ] **Step 5: Inspect the final diff**

Run: `git diff --check && git diff -- data/load_data.py model/model.py model/losses.py train/train.py tests/test_load_data.py tests/test_attention.py tests/test_model_loss.py docs/superpowers/specs/2026-08-01-periodic-mmd-conditioning-design.md docs/superpowers/plans/2026-08-01-periodic-mmd-conditioning.md`

Expected: `git diff --check` emits no whitespace errors; the diff contains only the approved conditioning change, tests, spec, and plan.
