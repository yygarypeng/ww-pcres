import h5py
import numpy as np

from data.preprocessing import valid_input_energy_rows
from physics.physics import HIGGS_MASS

SPLITS = ("train", "val", "test")

# Danning's splitting policy: each split is one fixed top-level HDF5 group.
PRESPLIT_CATEGORIES = {split: f"ggF_{split}" for split in SPLITS}

# Keys accepted in the config ``data:`` section. Anything else is a typo or a
# leftover from a removed split policy, so it is rejected instead of ignored.
DATA_CONFIG_KEYS = frozenset({"max_events_per_category"})

FOUR_VECTOR = ("px", "py", "pz", "energy")
JET_SLOTS = 2
LEPTONS = ("pos_lep", "neg_lep")

# HDF5 fields read from every selected category; anything else in the file is ignored.
CATEGORY_FIELDS = {
    "pos_lep": FOUR_VECTOR,
    "neg_lep": FOUR_VECTOR,
    "jets": FOUR_VECTOR,
    "met": ("px", "py"),
    "truth_pos_w": FOUR_VECTOR + ("m",),
    "truth_neg_w": FOUR_VECTOR + ("m",),
}


def select_categories(available_categories, categories):
    """Validate that every requested HDF5 category exists in the file."""
    available = list(available_categories)
    selected = list(categories)

    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Requested HDF5 categories not found: {missing}. Available: {available}")
    if not selected:
        raise ValueError(f"No HDF5 categories selected. Available: {available}")

    return selected


def load_particles_from_h5(filename, categories, max_events=None):
    """Read the fields of CATEGORY_FIELDS from each selected category of an HDF5 file."""
    with h5py.File(filename, "r") as f:
        return {
            category: {
                group: {field: f[category][group][field][:max_events] for field in fields}
                for group, fields in CATEGORY_FIELDS.items()
            }
            for category in select_categories(f.keys(), categories=categories)
        }


def _pack_inputs(category):
    """Pack 18 raw input columns: both leptons, two jet slots, then MET, all in GeV."""
    columns = [category[lepton][field] for lepton in LEPTONS for field in FOUR_VECTOR]
    columns += [
        category["jets"][field][:, slot] for slot in range(JET_SLOTS) for field in FOUR_VECTOR
    ]
    columns += [category["met"][field] for field in ("px", "py")]
    return np.column_stack(columns)


def _pack_targets(category):
    """Pack 10 target columns: both truth W four-vectors, then the two truth W masses."""
    bosons = ("truth_pos_w", "truth_neg_w")
    columns = [category[boson][field] for boson in bosons for field in FOUR_VECTOR]
    columns += [category[boson]["m"] for boson in bosons]
    return np.column_stack(columns)


def _valid_truth_w_rows(target_obj):
    """Rows whose truth W bosons are finite, timelike, and consistent with their stored mass."""
    truth = np.asarray(target_obj, dtype=np.float64)
    momenta = truth[:, :8].reshape(-1, 2, 4)  # (event, W boson, (px, py, pz, energy))
    masses = truth[:, 8:10]

    px, py, pz, energy = momenta.transpose(2, 0, 1)
    momentum2 = px**2 + py**2 + pz**2
    mass2 = energy**2 - momentum2
    tolerance = 1.0e-6 + 1.0e-6 * (energy**2 + momentum2 + masses**2)
    valid = (
        np.isfinite(momenta).all(axis=2)
        & np.isfinite(masses)
        & (energy > 0.0)
        & (masses >= 0.0)
        & (mass2 > 0.0)
        & (np.abs(mass2 - masses**2) <= tolerance)
    ).all(axis=1)

    pair = momenta.sum(axis=1)
    pair_mass2 = pair[:, 3] ** 2 - (pair[:, :3] ** 2).sum(axis=1)
    return valid & np.isfinite(pair_mass2) & (pair_mass2 > 0.0)


def _valid_dilepton_mass_rows(train_obj):
    """Rows whose measured dilepton mass stays below the Higgs mass, in GeV.

    Adding massless neutrinos can only raise an invariant mass, so an event
    whose two leptons already reach the Higgs mass can never be put on the Higgs
    mass shell, and WConstraintsLayer has no physical solution for it. The bound
    is therefore an event selection on measured leptons, decidable before the
    model runs, and it applies to every split.
    """
    leptons = np.asarray(train_obj[:, :8], dtype=np.float64)
    px, py, pz, energy = (leptons[:, i] + leptons[:, i + 4] for i in range(4))
    mass2 = energy**2 - px**2 - py**2 - pz**2
    return np.isfinite(mass2) & (mass2 < HIGGS_MASS**2)


def _row_masks(train_obj, target_obj):
    """The two row masks load_data applies, as (valid, kept).

    `valid` drops unusable kinematics and `kept` additionally applies the dilepton mass
    bound to the survivors, so `kept` selects exactly the rows load_data returns.
    """
    valid = (
        np.isfinite(train_obj).all(axis=1)
        & valid_input_energy_rows(train_obj)
        & _valid_truth_w_rows(target_obj)
    )
    kept = valid.copy()
    kept[valid] = _valid_dilepton_mass_rows(train_obj[valid])
    return valid, kept


def load_data(data_path, categories, max_events_per_category=None):
    """Load the raw input and target arrays of the selected categories, dropping unusable rows."""
    data = load_particles_from_h5(data_path, categories, max_events_per_category)
    print("Using HDF5 categories:", ", ".join(data))

    train_obj = np.concatenate([_pack_inputs(category) for category in data.values()])
    target_obj = np.concatenate([_pack_targets(category) for category in data.values()])
    print("Training objects shape:", train_obj.shape)
    print("Target objects shape:", target_obj.shape)

    valid, kept = _row_masks(train_obj, target_obj)
    print(
        "Removed",
        (~valid).sum(),
        "rows with non-finite values, invalid input energies, or invalid truth W kinematics",
    )
    print(
        "Removed",
        (valid & ~kept).sum(),
        f"rows with a dilepton mass of {HIGGS_MASS} GeV or more",
    )

    return train_obj[kept], target_obj[kept]


def load_presplit_data(data_path, data_cfg=None):
    """Load train/val/test arrays from the fixed pre-split HDF5 groups."""
    data_cfg = data_cfg or {}
    unknown_keys = set(data_cfg) - DATA_CONFIG_KEYS
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unsupported data config key(s): {names}")

    print("Using original pre-split HDF5 data")
    for split, category in PRESPLIT_CATEGORIES.items():
        print(f"{split.capitalize()} category:", category)

    max_events = data_cfg.get("max_events_per_category")
    return tuple(
        array
        for category in PRESPLIT_CATEGORIES.values()
        for array in load_data(data_path, [category], max_events)
    )
