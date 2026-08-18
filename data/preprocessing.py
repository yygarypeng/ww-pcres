import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

BASE_INPUT_DIM = 18  # w/o high-level features
RAW_INPUT_DIM = 21  # with high-level features
NEURAL_INPUT_DIM = 21
INPUT_PREPROCESSING_VERSION = 3


def _require_raw_input_shape(features):
    if features.ndim != 2 or features.shape[1] not in (BASE_INPUT_DIM, RAW_INPUT_DIM):
        raise ValueError(
            f"Raw input features must have shape (N, {BASE_INPUT_DIM}) or (N, {RAW_INPUT_DIM}), "
            f"got {features.shape}"
        )


def valid_input_energy_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    _require_raw_input_shape(features)

    valid = (
        np.isfinite(features[:, 3])
        & (features[:, 3] > 0.0)
        & np.isfinite(features[:, 7])
        & (features[:, 7] > 0.0)
    )
    for start in (8, 12):
        jet = features[:, start : start + 4]
        is_padding = (jet == 0.0).all(axis=1)
        has_positive_energy = np.isfinite(jet[:, 3]) & (jet[:, 3] > 0.0)
        valid &= is_padding | has_positive_energy

    return valid


def neural_input_features_numpy(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    _require_raw_input_shape(features)
    if not np.issubdtype(features.dtype, np.floating):
        features = features.astype(np.float64)
    if not valid_input_energy_rows(features).all():
        raise ValueError("Raw input contains invalid lepton or jet energy values")

    return np.concatenate(
        [
            features[:, :3],
            np.log1p(features[:, 3:4]),
            features[:, 4:7],
            np.log1p(features[:, 7:8]),
            features[:, 8:11],
            np.log1p(features[:, 11:12]),
            features[:, 12:15],
            np.log1p(features[:, 15:16]),
            features[:, 16:],
        ],
        axis=1,
    )


def neural_input_features_torch(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] not in (BASE_INPUT_DIM, RAW_INPUT_DIM):
        raise ValueError(
            f"Raw input features must have shape (N, {BASE_INPUT_DIM}) or (N, {RAW_INPUT_DIM}), "
            f"got {tuple(features.shape)}"
        )
    if not features.is_floating_point():
        features = features.to(torch.get_default_dtype())

    return torch.cat(
        [
            features[:, :3],
            torch.log1p(features[:, 3:4]),
            features[:, 4:7],
            torch.log1p(features[:, 7:8]),
            features[:, 8:11],
            torch.log1p(features[:, 11:12]),
            features[:, 12:15],
            torch.log1p(features[:, 15:16]),
            features[:, 16:],
        ],
        dim=1,
    )


def compute_neural_input_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = neural_input_features_numpy(features)
    scaler = StandardScaler().fit(transformed)
    mean = scaler.mean_.copy()
    scale = scaler.scale_.copy()

    for raw_energy_column, neural_start in ((11, 8), (15, 12)):
        present = features[:, raw_energy_column] > 0.0
        if present.any():
            jet_scaler = StandardScaler().fit(transformed[present, neural_start : neural_start + 4])
            mean[neural_start : neural_start + 4] = jet_scaler.mean_
            scale[neural_start : neural_start + 4] = jet_scaler.scale_
        else:
            mean[neural_start : neural_start + 4] = 0.0
            scale[neural_start : neural_start + 4] = 1.0

    if features.shape[1] == RAW_INPUT_DIM:
        mean[20] = 0.0
        scale[20] = 1.0
    return mean, scale
