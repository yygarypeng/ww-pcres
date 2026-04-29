import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler

from physics import pt, eta, phi, deta, dphi


def select_categories(available_categories, categories=None):
    """Select HDF5 categories. None means load all categories."""
    available = list(available_categories)
    if categories:
        missing = sorted(set(categories) - set(available))
        if missing:
            raise ValueError(f"Requested HDF5 categories not found: {missing}. Available: {available}")
        selected = list(categories)
    else:
        selected = available

    if not selected:
        raise ValueError(f"No HDF5 categories selected. Available: {available}")

    return selected


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


def _safe_log_positive(values):
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = values > 0.0
    result[valid] = np.log(values[valid])
    return result

def load_particles_from_h5(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        # For each category (ggF_train, ggF_test, VBF_train, etc.)
        for category_name in f.keys():
            category_data = {}
            
            # For each particle/object group within the category
            for group_name in f[category_name].keys():
                group_data = {}

                # Load datasets (numpy arrays)
                if isinstance(f[category_name][group_name], h5py.Group):
                    for dataset_name in f[category_name][group_name].keys():
                        group_data[dataset_name] = f[category_name][group_name][dataset_name][:]

                    # Load attributes (scalars)
                    for attr_name, attr_value in f[category_name][group_name].attrs.items():
                        group_data[attr_name] = attr_value
                else:
                    # Handle case where it's a dataset directly
                    group_data = f[category_name][group_name][:]

                category_data[group_name] = group_data

            result[category_name] = category_data

    return result

def load_data(
    data_path,
    categories=None,
):
    
    data = load_particles_from_h5(data_path)

    def col(a):
        return a.reshape(a.shape[0], -1)
    
    # Collect all training and target objects from all categories
    all_train_objs = []
    all_target_objs = []
    
    selected_categories = select_categories(
        data.keys(),
        categories=categories,
    )
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
        
        lep_pos_pt = category_data["pos_lep"]["pt"]
        lep_neg_pt = category_data["neg_lep"]["pt"]
        lep_pos_eta = category_data["pos_lep"]["eta"]
        lep_neg_eta = category_data["neg_lep"]["eta"]
        lep_pos_phi = category_data["pos_lep"]["phi"]
        lep_neg_phi = category_data["neg_lep"]["phi"]
        
        dilep_px = lep_pos_px + lep_neg_px
        dilep_py = lep_pos_py + lep_neg_py
        dilep_pz = lep_pos_pz + lep_neg_pz
        dilep_energy = lep_pos_energy + lep_neg_energy
        dilep_pt = pt(dilep_px, dilep_py)
        dilep_eta = eta(dilep_px, dilep_py, dilep_pz)
        dilep_phi = phi(dilep_px, dilep_py)
        m_ll2 = dilep_energy**2 - dilep_px**2 - dilep_py**2 - dilep_pz**2
        m_ll = np.where(m_ll2 >= -1.0e-6, np.sqrt(np.clip(m_ll2, 0.0, None)), np.nan)

        met_px = category_data["met"]["px"]
        met_py = category_data["met"]["py"]
        met_pt = category_data["met"]["pt"]
        met_phi = category_data["met"]["phi"]
        
        dphi_ll = dphi(lep_pos_phi, lep_neg_phi)
        dphi_llmet = dphi(dilep_phi, met_phi)
        dphi_l1met = dphi(lep_pos_phi, met_phi)
        dphi_l2met = dphi(lep_neg_phi, met_phi)
        
        deta_ll = deta(lep_pos_eta, lep_neg_eta)
        jet_px = category_data["jets"]["px"][:, 0:2]
        jet_py = category_data["jets"]["py"][:, 0:2]
        jet_pz = category_data["jets"]["pz"][:, 0:2]
        jet_energy = category_data["jets"]["energy"][:, 0:2]

        # pack them
        # all training mass-like objects are in GeV unit
        
        train_obj = np.concatenate([
            col(lep_pos_px), #0
            col(lep_pos_py), #1
            col(lep_pos_pz), #2
            col(lep_pos_energy), #3
            col(lep_neg_px), #4
            col(lep_neg_py), #5
            col(lep_neg_pz), #6
            col(lep_neg_energy), #7
            col(jet_px[:, 0]), #8
            col(jet_py[:, 0]), #9
            col(jet_pz[:, 0]), #10
            col(jet_energy[:, 0]), #11
            col(jet_px[:, 1]), #12
            col(jet_py[:, 1]), #13
            col(jet_pz[:, 1]), #14
            col(jet_energy[:, 1]), #15
            col(met_px), #16
            col(met_py), #17
            # high level features
            col(m_ll), #18
            col(dilep_pt), #19
            col(met_pt), #20
            col(deta_ll), #21
            col(dphi_llmet), #22
            col(dphi_l1met), #23 (l1 -> pos_lep; l2 -> neg_lep)
            col(dphi_l2met), #24
            col(dphi_ll), #25
        ], axis=-1)
        
        # target objects
        target_obj = np.concatenate([
			col(category_data["truth_pos_w"]["px"]),
			col(category_data["truth_pos_w"]["py"]),
			col(category_data["truth_pos_w"]["pz"]),
			col(_safe_log_positive(category_data["truth_pos_w"]["energy"])),
			col(category_data["truth_neg_w"]["px"]),
			col(category_data["truth_neg_w"]["py"]),
			col(category_data["truth_neg_w"]["pz"]),
			col(_safe_log_positive(category_data["truth_neg_w"]["energy"])),
			col(category_data["truth_pos_w"]["m"]),
			col(category_data["truth_neg_w"]["m"]),
        ], axis=-1)
        
        all_train_objs.append(train_obj)
        all_target_objs.append(target_obj)
    
    # Concatenate all categories
    train_obj = np.concatenate(all_train_objs, axis=0)
    target_obj = np.concatenate(all_target_objs, axis=0)
    
    print("Training objects shape:", train_obj.shape)
    print("Target objects shape:", target_obj.shape)

    # Remove rows with NaN or infinite values
    valid_train = np.isfinite(train_obj).all(axis=1)
    valid_target = np.isfinite(target_obj).all(axis=1)
    valid_idx = valid_train & valid_target

    train_obj = train_obj[valid_idx]
    target_obj = target_obj[valid_idx]

    print("Removed", (~valid_idx).sum(), "rows with NaN or infinite values")
    
    (std_mean_train, std_scale_train), (std_mean_target, std_scale_target) = compute_standardization_stats(
        train_obj,
        target_obj,
    )


    return train_obj, target_obj, (std_mean_train, std_scale_train), (std_mean_target, std_scale_target)
