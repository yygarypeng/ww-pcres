import numpy as np
import pytest
import torch

from data import (
    BASE_INPUT_DIM,
    NEURAL_INPUT_DIM,
    RAW_INPUT_DIM,
)
from data.preprocessing import (
    compute_neural_input_stats,
    neural_input_features_numpy,
    neural_input_features_torch,
    valid_input_energy_rows,
)


def valid_raw_rows(dtype=np.float32):
    rows = np.zeros((2, RAW_INPUT_DIM), dtype=dtype)
    rows[:, 3] = [1.0, 2.0]
    rows[:, 7] = [3.0, 4.0]
    rows[0, 8:12] = [5.0, 6.0, 7.0, 8.0]
    rows[1, 12:16] = [9.0, 10.0, 11.0, 12.0]
    rows[:, 16:21] = [[13.0, 14.0, 15.0, 16.0, 0.5], [17.0, 18.0, 19.0, 20.0, -1.0]]
    return rows


def expected_neural_features(raw):
    return np.concatenate(
        [
            raw[:, :3],
            np.log1p(raw[:, 3:4]),
            raw[:, 4:7],
            np.log1p(raw[:, 7:8]),
            raw[:, 8:11],
            np.log1p(raw[:, 11:12]),
            raw[:, 12:15],
            np.log1p(raw[:, 15:16]),
            raw[:, 16:21],
        ],
        axis=1,
    )


def test_preprocessing_schema_constants_are_exported():
    assert (RAW_INPUT_DIM, NEURAL_INPUT_DIM) == (21, 21)


def test_numpy_transform_has_exact_order_and_does_not_mutate_input():
    raw = valid_raw_rows()
    original = raw.copy()

    transformed = neural_input_features_numpy(raw)

    np.testing.assert_allclose(transformed, expected_neural_features(raw))
    np.testing.assert_array_equal(raw, original)
    assert transformed.shape == (2, NEURAL_INPUT_DIM)
    assert transformed.dtype == np.float32


def test_numpy_and_torch_transforms_match_without_mutating_input():
    raw = valid_raw_rows()
    tensor = torch.from_numpy(raw.copy())
    original = tensor.clone()

    transformed = neural_input_features_torch(tensor)

    np.testing.assert_allclose(transformed.numpy(), expected_neural_features(raw), rtol=1e-6)
    torch.testing.assert_close(tensor, original)
    assert transformed.dtype == torch.float32


def test_integer_inputs_are_converted_to_floating_point():
    raw = valid_raw_rows(dtype=np.int64)

    numpy_result = neural_input_features_numpy(raw)
    torch_result = neural_input_features_torch(torch.from_numpy(raw))

    assert np.issubdtype(numpy_result.dtype, np.floating)
    assert torch_result.is_floating_point()
    np.testing.assert_allclose(torch_result.numpy(), numpy_result, rtol=1e-6)


@pytest.mark.parametrize("shape", [(21,), (2, 20), (2, 21, 1)])
def test_numpy_transform_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        neural_input_features_numpy(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("column,value", [(3, 0.0), (3, -1.0), (7, np.inf)])
def test_numpy_transform_rejects_invalid_lepton_energies(column, value):
    raw = valid_raw_rows()
    raw[0, column] = value

    with pytest.raises(ValueError, match="energy"):
        neural_input_features_numpy(raw)


@pytest.mark.parametrize(
    "jet",
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, np.inf],
    ],
)
def test_numpy_transform_rejects_invalid_jets(jet):
    raw = valid_raw_rows()
    raw[0, 8:12] = jet

    with pytest.raises(ValueError, match="energy"):
        neural_input_features_numpy(raw)


def test_numpy_transform_rejects_negative_energy_jet():
    raw = valid_raw_rows()
    raw[0, 8:12] = [1.0, 2.0, 3.0, -1.0]

    with pytest.raises(ValueError, match="energy"):
        neural_input_features_numpy(raw)


def test_transform_retains_exact_zero_jet_padding():
    raw = valid_raw_rows()
    raw[:, 8:16] = 0.0

    transformed = neural_input_features_numpy(raw)

    np.testing.assert_array_equal(transformed[:, 8:16], 0.0)


def test_valid_input_energy_rows_enforces_leptons_and_exact_jet_padding():
    rows = np.repeat(valid_raw_rows()[:1], 6, axis=0)
    rows[:, 8:16] = 0.0
    rows[1, 3] = 0.0
    rows[2, 7] = -1.0
    rows[3, 11] = -1.0
    rows[4, 8:12] = [1.0, 0.0, 0.0, 0.0]
    rows[5, 12:16] = [0.0, 0.0, 0.0, np.nan]

    np.testing.assert_array_equal(
        valid_input_energy_rows(rows),
        [True, False, False, False, False, False],
    )


def test_neural_stats_mask_each_jet_slot_and_use_all_rows_for_other_scalars():
    raw = np.repeat(valid_raw_rows()[:1], 3, axis=0)
    raw[:, 8:16] = 0.0
    raw[0, 8:12] = [1.0, 2.0, 3.0, 4.0]
    raw[2, 8:12] = [5.0, 6.0, 7.0, 8.0]
    raw[1, 12:16] = [10.0, 20.0, 30.0, 40.0]
    raw[:, 16] = [2.0, 4.0, 9.0]
    raw[:, 17] = 3.0
    raw[:, 20] = [0.1, 0.3, 0.5]
    transformed = neural_input_features_numpy(raw)

    mean, scale = compute_neural_input_stats(raw)

    np.testing.assert_allclose(mean[8:12], transformed[[0, 2], 8:12].mean(axis=0))
    np.testing.assert_allclose(scale[8:12], transformed[[0, 2], 8:12].std(axis=0))
    np.testing.assert_allclose(mean[12:16], transformed[[1], 12:16].mean(axis=0))
    np.testing.assert_allclose(scale[12:16], np.ones(4))
    np.testing.assert_allclose(mean[16], transformed[:, 16].mean())
    np.testing.assert_allclose(scale[16], transformed[:, 16].std())
    np.testing.assert_allclose(mean[17], transformed[:, 17].mean())
    assert scale[17] > 0.0
    np.testing.assert_array_equal(mean[20:21], np.zeros(1))
    np.testing.assert_array_equal(scale[20:21], np.ones(1))


def test_neural_stats_use_neutral_fallback_for_completely_absent_jet_slot():
    raw = valid_raw_rows()
    raw[:, 12:16] = 0.0

    mean, scale = compute_neural_input_stats(raw)

    np.testing.assert_array_equal(mean[12:16], np.zeros(4))
    np.testing.assert_array_equal(scale[12:16], np.ones(4))


def test_preprocessing_supports_base_width_without_high_level_features():
    rows = np.zeros((2, BASE_INPUT_DIM), dtype=np.float32)
    rows[:, 3] = [1.0, 2.0]
    rows[:, 7] = [3.0, 4.0]
    rows[0, 8:12] = [5.0, 6.0, 7.0, 8.0]
    rows[1, 12:16] = [9.0, 10.0, 11.0, 12.0]
    rows[:, 16:18] = [[13.0, 14.0], [17.0, 18.0]]

    transformed = neural_input_features_numpy(rows)

    expected = np.concatenate(
        [
            rows[:, :3],
            np.log1p(rows[:, 3:4]),
            rows[:, 4:7],
            np.log1p(rows[:, 7:8]),
            rows[:, 8:11],
            np.log1p(rows[:, 11:12]),
            rows[:, 12:15],
            np.log1p(rows[:, 15:16]),
            rows[:, 16:],
        ],
        axis=1,
    )
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_array_equal(
        valid_input_energy_rows(rows),
        [True, True],
    )


def test_neural_stats_without_high_level_features_skip_periodic_column_rule():
    rows = np.zeros((2, BASE_INPUT_DIM), dtype=np.float32)
    rows[:, 3] = [1.0, 2.0]
    rows[:, 7] = [3.0, 4.0]
    rows[:, 16:18] = [[2.0, 3.0], [4.0, 5.0]]

    mean, scale = compute_neural_input_stats(rows)

    assert mean.shape == (BASE_INPUT_DIM,)
    assert scale.shape == (BASE_INPUT_DIM,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(scale))
