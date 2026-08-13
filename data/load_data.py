import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from data.preprocessing import normalize_negative_energy_jets_numpy, valid_input_energy_rows
from physics import deta, dphi, eta


def select_categories(available_categories, categories=None):
    """Select HDF5 categories. None means load all categories."""
    available = list(available_categories)
    if categories:
        missing = sorted(set(categories) - set(available))
        if missing:
            raise ValueError(
                f"Requested HDF5 categories not found: {missing}. Available: {available}"
            )
        selected = list(categories)
    else:
        selected = available

    if not selected:
        raise ValueError(f"No HDF5 categories selected. Available: {available}")

    return selected


def split_categories(data_cfg, split):
    explicit_categories = data_cfg.get(f"{split}_categories")
    if explicit_categories is not None:
        return explicit_categories

    categories = data_cfg.get("categories")
    if categories:
        split_suffix = f"_{split}"
        selected_categories = []
        for category in categories:
            category = str(category)
            if category.endswith(("_train", "_val", "_test")):
                if category.endswith(split_suffix):
                    selected_categories.append(category)
            else:
                selected_categories.append(f"{category}{split_suffix}")

        if not selected_categories:
            raise ValueError(f"No categories selected for {split} split from data.categories")
        return selected_categories

    return [f"ggF_{split}"]


def mmd_condition_features(features):
    features = np.asarray(features)
    if features.shape[-1] != 21:
        raise ValueError(f"MMD conditioning requires exactly 21 features, got {features.shape[-1]}")

    m_ll = features[..., 18:19]
    deta_ll = features[..., 19:20]
    dphi_ll = features[..., 20:21]
    return np.concatenate(
        [
            m_ll,
            deta_ll,
            np.sin(dphi_ll),
            np.cos(dphi_ll),
        ],
        axis=-1,
    )


def compute_mmd_condition_stats(train_obj):
    condition = mmd_condition_features(train_obj)
    scaler = StandardScaler().fit(condition[:, :2])
    mean = np.zeros(condition.shape[1], dtype=scaler.mean_.dtype)
    scale = np.ones(condition.shape[1], dtype=scaler.scale_.dtype)
    mean[:2] = scaler.mean_
    scale[:2] = scaler.scale_
    return mean, scale


def compute_standardization_stats(train_obj, target_obj=None, train_indices=None):
    """Fit standardization statistics, optionally on a training subset only."""
    feature_source = train_obj if train_indices is None else train_obj[train_indices]
    feature_scaler = StandardScaler().fit(feature_source)
    feature_stats = (feature_scaler.mean_, feature_scaler.scale_)

    if target_obj is None:
        return feature_stats, None

    target_source = target_obj if train_indices is None else target_obj[train_indices]
    target_scaler = StandardScaler().fit(target_source)
    target_stats = (target_scaler.mean_, target_scaler.scale_)
    return feature_stats, target_stats


def _read_dataset(dataset, max_events=None):
    if max_events is not None and dataset.shape and dataset.shape[0] > max_events:
        return dataset[:max_events]
    return dataset[:]


def _valid_truth_w_rows(target_obj):
    truth = np.asarray(target_obj, dtype=np.float64)
    pos_px, pos_py, pos_pz, pos_energy = truth[:, :4].T
    neg_px, neg_py, neg_pz, neg_energy = truth[:, 4:8].T
    pos_mass, neg_mass = truth[:, 8:10].T

    valid = np.ones(len(truth), dtype=bool)
    for px, py, pz, energy, mass in (
        (pos_px, pos_py, pos_pz, pos_energy, pos_mass),
        (neg_px, neg_py, neg_pz, neg_energy, neg_mass),
    ):
        raw_m2 = energy**2 - px**2 - py**2 - pz**2
        delta = raw_m2 - mass**2
        scale = energy**2 + px**2 + py**2 + pz**2 + mass**2
        valid &= (
            np.isfinite(np.column_stack((px, py, pz, energy, mass))).all(axis=1)
            & (energy > 0.0)
            & (mass >= 0.0)
            & (raw_m2 > 0.0)
            & (np.abs(delta) <= 1.0e-6 + 1.0e-6 * scale)
        )

    pair_m2 = (
        (pos_energy + neg_energy) ** 2
        - (pos_px + neg_px) ** 2
        - (pos_py + neg_py) ** 2
        - (pos_pz + neg_pz) ** 2
    )
    return valid & np.isfinite(pair_m2) & (pair_m2 > 0.0)


def load_particles_from_h5(filename, categories=None, max_events=None):
    result = {}

    with h5py.File(filename, "r") as f:
        selected_categories = select_categories(f.keys(), categories=categories)
        # For each category (ggF_train, ggF_test, VBF_train, etc.)
        for category_name in selected_categories:
            category_data = {}

            # For each particle/object group within the category
            for group_name in f[category_name].keys():
                group_data = {}

                # Load datasets (numpy arrays)
                if isinstance(f[category_name][group_name], h5py.Group):
                    for dataset_name in f[category_name][group_name].keys():
                        group_data[dataset_name] = _read_dataset(
                            f[category_name][group_name][dataset_name],
                            max_events=max_events,
                        )

                    # Load attributes (scalars)
                    for attr_name, attr_value in f[category_name][group_name].attrs.items():
                        group_data[attr_name] = attr_value
                else:
                    # Handle case where it's a dataset directly
                    group_data = _read_dataset(f[category_name][group_name], max_events=max_events)

                category_data[group_name] = group_data

            result[category_name] = category_data

    return result


def load_data(
    data_path,
    categories=None,
    max_events_per_category=None,
):

    data = load_particles_from_h5(
        data_path,
        categories=categories,
        max_events=max_events_per_category,
    )

    def col(a):
        return a.reshape(a.shape[0], -1)

    # Collect all training and target objects from all categories
    all_train_objs = []
    all_target_objs = []

    selected_categories = list(data.keys())
    print("Using HDF5 categories:", ", ".join(selected_categories))

    # Iterate through selected categories (ggF_train, VBF_train, etc.)
    for category in selected_categories:
        category_data = data[category]

        # training features
        lep_pos_px = category_data["pos_lep"]["px"]
        lep_pos_py = category_data["pos_lep"]["py"]
        lep_pos_pz = category_data["pos_lep"]["pz"]
        lep_pos_energy = category_data["pos_lep"]["energy"]
        lep_neg_px = category_data["neg_lep"]["px"]
        lep_neg_py = category_data["neg_lep"]["py"]
        lep_neg_pz = category_data["neg_lep"]["pz"]
        lep_neg_energy = category_data["neg_lep"]["energy"]

        lep_pos_pt = category_data["pos_lep"]["pt"]  # noqa: F841
        lep_neg_pt = category_data["neg_lep"]["pt"]  # noqa: F841
        lep_pos_eta = category_data["pos_lep"]["eta"]
        lep_neg_eta = category_data["neg_lep"]["eta"]
        lep_pos_phi = category_data["pos_lep"]["phi"]
        lep_neg_phi = category_data["neg_lep"]["phi"]

        dilep_px = lep_pos_px + lep_neg_px
        dilep_py = lep_pos_py + lep_neg_py
        dilep_pz = lep_pos_pz + lep_neg_pz
        dilep_energy = lep_pos_energy + lep_neg_energy
        dilep_eta = eta(dilep_px, dilep_py, dilep_pz)  # noqa: F841
        m_ll2 = dilep_energy**2 - dilep_px**2 - dilep_py**2 - dilep_pz**2
        m_ll = np.where(m_ll2 >= -1.0e-6, np.sqrt(np.clip(m_ll2, 0.0, None)), np.nan)

        met_px = category_data["met"]["px"]
        met_py = category_data["met"]["py"]

        dphi_ll = dphi(lep_pos_phi, lep_neg_phi)
        deta_ll = deta(lep_pos_eta, lep_neg_eta)

        jet_px = category_data["jets"]["px"][:, 0:2]
        jet_py = category_data["jets"]["py"][:, 0:2]
        jet_pz = category_data["jets"]["pz"][:, 0:2]
        jet_energy = category_data["jets"]["energy"][:, 0:2]

        # pack them
        # all training mass-like objects are in GeV unit

        train_obj = np.concatenate(
            [
                col(lep_pos_px),  # 0
                col(lep_pos_py),  # 1
                col(lep_pos_pz),  # 2
                col(lep_pos_energy),  # 3
                col(lep_neg_px),  # 4
                col(lep_neg_py),  # 5
                col(lep_neg_pz),  # 6
                col(lep_neg_energy),  # 7
                col(jet_px[:, 0]),  # 8
                col(jet_py[:, 0]),  # 9
                col(jet_pz[:, 0]),  # 10
                col(jet_energy[:, 0]),  # 11
                col(jet_px[:, 1]),  # 12
                col(jet_py[:, 1]),  # 13
                col(jet_pz[:, 1]),  # 14
                col(jet_energy[:, 1]),  # 15
                col(met_px),  # 16
                col(met_py),  # 17
                # high level features
                col(m_ll),  # 18
                col(deta_ll),  # 19
                col(dphi_ll),  # 20
            ],
            axis=-1,
        )

        # target objects
        target_obj = np.concatenate(
            [
                col(category_data["truth_pos_w"]["px"]),
                col(category_data["truth_pos_w"]["py"]),
                col(category_data["truth_pos_w"]["pz"]),
                col(category_data["truth_pos_w"]["energy"]),
                col(category_data["truth_neg_w"]["px"]),
                col(category_data["truth_neg_w"]["py"]),
                col(category_data["truth_neg_w"]["pz"]),
                col(category_data["truth_neg_w"]["energy"]),
                col(category_data["truth_pos_w"]["m"]),
                col(category_data["truth_neg_w"]["m"]),
            ],
            axis=-1,
        )

        all_train_objs.append(train_obj)
        all_target_objs.append(target_obj)

    # Concatenate all categories
    train_obj = np.concatenate(all_train_objs, axis=0)
    target_obj = np.concatenate(all_target_objs, axis=0)

    print("Training objects shape:", train_obj.shape)
    print("Target objects shape:", target_obj.shape)

    train_obj = normalize_negative_energy_jets_numpy(train_obj)

    # Remove rows with non-finite values or invalid truth W kinematics
    valid_train = np.isfinite(train_obj).all(axis=1)
    valid_target = np.isfinite(target_obj).all(axis=1)
    valid_physics = _valid_truth_w_rows(target_obj)
    valid_input_energy = valid_input_energy_rows(train_obj)
    valid_idx = valid_train & valid_target & valid_physics & valid_input_energy

    train_obj = train_obj[valid_idx]
    target_obj = target_obj[valid_idx]

    print(
        "Removed",
        (~valid_idx).sum(),
        "rows with non-finite values, invalid input energies, or invalid truth W kinematics",
    )

    (std_mean_train, std_scale_train), (std_mean_target, std_scale_target) = (
        compute_standardization_stats(
            train_obj,
            target_obj,
        )
    )

    return (
        train_obj,
        target_obj,
        (std_mean_train, std_scale_train),
        (std_mean_target, std_scale_target),
    )


def load_presplit_data(data_path, data_cfg=None):
    """Load train/val/test arrays from HDF5 groups that are already split."""
    if data_cfg is None:
        data_cfg = {}

    train_categories = split_categories(data_cfg, "train")
    val_categories = split_categories(data_cfg, "val")
    test_categories = split_categories(data_cfg, "test")

    print("Using original pre-split HDF5 data")
    print("Train categories:", ", ".join(train_categories))
    print("Validation categories:", ", ".join(val_categories))
    print("Test categories:", ", ".join(test_categories))

    max_events = data_cfg.get("max_events_per_category")
    X_train, Y_train, _, _ = load_data(
        data_path, categories=train_categories, max_events_per_category=max_events
    )
    X_val, Y_val, _, _ = load_data(
        data_path, categories=val_categories, max_events_per_category=max_events
    )
    X_test, Y_test, _, _ = load_data(
        data_path, categories=test_categories, max_events_per_category=max_events
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
