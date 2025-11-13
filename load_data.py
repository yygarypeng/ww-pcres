import numpy as np
import h5py
import ohbboosting as ohb
from sklearn.preprocessing import StandardScaler

def load_particles_from_h5(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        # For each group in the file
        for group_name in f.keys():
            group_data = {}

            # Load datasets (numpy arrays)
            for dataset_name in f[group_name].keys():
                group_data[dataset_name] = f[group_name][dataset_name][:]

            # Load attributes (scalars)
            for attr_name, attr_value in f[group_name].attrs.items():
                group_data[attr_name] = attr_value

            result[group_name] = group_data

    return result

def load_data(data_path):
    
	data = load_particles_from_h5(data_path)
	# print("Data keys:", data.keys())

	truth_pos_lep = data["truth_pos_lep"]
	truth_neg_lep = data["truth_neg_lep"]
	truth_dilep = data["truth_dilep"]
	truth_pos_nu = data["truth_pos_nu"]
	truth_neg_nu = data["truth_neg_nu"]
	# truth_pos_nu_p4 = data["truth_pos_nu"]["p4"]
	# truth_neg_nu_p4 = data["truth_neg_nu"]["p4"]
	# truth_met_pt = np.sqrt(np.square((truth_pos_nu_p4 + truth_neg_nu_p4)[...,0:2]).sum(axis=-1))

	# training objects
	pos_lep_px = truth_pos_lep["px"]
	pos_lep_py = truth_pos_lep["py"]
	pos_lep_pz = truth_pos_lep["pz"]
	pos_lep_energy = truth_pos_lep["energy"]
	neg_lep_px = truth_neg_lep["px"]
	neg_lep_py = truth_neg_lep["py"]
	neg_lep_pz = truth_neg_lep["pz"]
	neg_lep_energy = truth_neg_lep["energy"]
	pos_nu_px = truth_pos_nu["px"]
	pos_nu_py = truth_pos_nu["py"]
	neg_nu_px = truth_neg_nu["px"]
	neg_nu_py = truth_neg_nu["py"]
	met_px = pos_nu_px + neg_nu_px
	met_py = pos_nu_py + neg_nu_py
	dilep_phi = truth_dilep["dphi"]
	dilep_eta = truth_dilep["deta"]
	# pack them
	train_obj = np.column_stack(
		(
			pos_lep_px,
			pos_lep_py,
			pos_lep_pz,
			pos_lep_energy,
			neg_lep_px,
			neg_lep_py,
			neg_lep_pz,
			neg_lep_energy,
			met_px,
			met_py,
			dilep_phi,
			dilep_eta,
		)
	)
	scaler = StandardScaler()
	train_obj = scaler.fit_transform(train_obj)
	print("Training objects shape:", train_obj.shape)

	# target objects (need to use truth-level samples)
	w_pos = data["pos_w"]
	w_neg = data["neg_w"]
	particles = np.concatenate(
		[
			w_pos["p4"],
			truth_pos_lep["p4"],
			w_neg["p4"],
			truth_neg_lep["p4"],
		],
		axis=-1,
	)
	print(particles.shape)
	booster = ohb.Booster(particles)
	booster.setup()
	lep_pos_in_w, lep_neg_in_w = booster.lep_4_in_w_rest()
	print("Lepton in W rest frame shapes:", lep_pos_in_w.shape, lep_neg_in_w.shape)
	# unpack
	lp_in_w_px = lep_pos_in_w[:, 0]
	lp_in_w_py = lep_pos_in_w[:, 1]
	lp_in_w_pz = lep_pos_in_w[:, 2]
	lp_in_w_energy = lep_pos_in_w[:, 3]
	ln_in_w_px = lep_neg_in_w[:, 0]
	ln_in_w_py = lep_neg_in_w[:, 1]
	ln_in_w_pz = lep_neg_in_w[:, 2]
	ln_in_w_energy = lep_neg_in_w[:, 3]
	# nu and lep are back-to-back in W rest frame; neutrinos are massless
	np_in_w_px = -lp_in_w_px
	np_in_w_py = -lp_in_w_py
	np_in_w_pz = -lp_in_w_pz
	np_in_w_energy = np.sqrt(np.square(np_in_w_px) + np.square(np_in_w_py) + np.square(np_in_w_pz))
	nn_in_w_px = -ln_in_w_px
	nn_in_w_py = -ln_in_w_py
	nn_in_w_pz = -ln_in_w_pz
	nn_in_w_energy = np.sqrt(np.square(nn_in_w_px) + np.square(nn_in_w_py) + np.square(nn_in_w_pz))
	wp_px = np_in_w_px + lp_in_w_px
	wp_py = np_in_w_py + lp_in_w_py
	wp_pz = np_in_w_pz + lp_in_w_pz
	wp_energy = np_in_w_energy + lp_in_w_energy
	wn_px = nn_in_w_px + ln_in_w_px
	wn_py = nn_in_w_py + ln_in_w_py
	wn_pz = nn_in_w_pz + ln_in_w_pz
	wn_energy = nn_in_w_energy + ln_in_w_energy
	wp_mass = np.sqrt(np.square(wp_energy) - np.square(wp_px) - np.square(wp_py) - np.square(wp_pz))
	wn_mass = np.sqrt(np.square(wn_energy) - np.square(wn_px) - np.square(wn_py) - np.square(wn_pz))
	#  pack them
	target_obj = np.column_stack(
		(
			lp_in_w_px,
			lp_in_w_py,
			lp_in_w_pz,
			lp_in_w_energy,
			ln_in_w_px,
			ln_in_w_py,
			ln_in_w_pz,
			ln_in_w_energy,
			wp_mass,
			wn_mass,
		)
	)
	print("Target objects shape:", target_obj.shape)

	return train_obj, target_obj

if __name__ == "__main__":
	data_path = "/root/data/mc20_truth_v4_SM.h5"
	train_obj, target_obj = load_data(data_path)