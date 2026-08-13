import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

RAW_INPUT_DIM = 21
NEURAL_INPUT_DIM = 22
INPUT_PREPROCESSING_VERSION = 2


def _require_raw_input_shape(features):
    if features.ndim != 2 or features.shape[1] != RAW_INPUT_DIM:
        raise ValueError(
            f"Raw input features must have shape (N, {RAW_INPUT_DIM}), got {features.shape}"
        )


def normalize_negative_energy_jets_numpy(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    _require_raw_input_shape(features)
    if not np.issubdtype(features.dtype, np.floating):
        features = features.astype(np.float64)

    parts = [features[:, :8]]
    for start in (8, 12):
        jet = features[:, start : start + 4]
        absent = np.isfinite(jet[:, 3:4]) & (jet[:, 3:4] < 0.0)
        parts.append(np.where(absent, np.zeros_like(jet), jet))
    parts.append(features[:, 16:])
    return np.concatenate(parts, axis=1)


def normalize_negative_energy_jets_torch(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != RAW_INPUT_DIM:
        raise ValueError(
            f"Raw input features must have shape (N, {RAW_INPUT_DIM}), got {tuple(features.shape)}"
        )
    if not features.is_floating_point():
        features = features.to(torch.get_default_dtype())

    parts = [features[:, :8]]
    for start in (8, 12):
        jet = features[:, start : start + 4]
        absent = torch.isfinite(jet[:, 3:4]) & (jet[:, 3:4] < 0.0)
        parts.append(torch.where(absent, torch.zeros_like(jet), jet))
    parts.append(features[:, 16:])
    return torch.cat(parts, dim=1)


def valid_input_energy_rows(features: np.ndarray) -> np.ndarray:
    features = normalize_negative_energy_jets_numpy(features)

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
    features = normalize_negative_energy_jets_numpy(features)
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
            features[:, 16:20],
            np.sin(features[:, 20:21]),
            np.cos(features[:, 20:21]),
        ],
        axis=1,
    )


def neural_input_features_torch(features: torch.Tensor) -> torch.Tensor:
    features = normalize_negative_energy_jets_torch(features)

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
            features[:, 16:20],
            torch.sin(features[:, 20:21]),
            torch.cos(features[:, 20:21]),
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

    mean[20:22] = 0.0
    scale[20:22] = 1.0
    return mean, scale
