import numpy as np
import h5py

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

	# preselection
	truth_pos_lep = data["truth_pos_lep"]
	truth_neg_lep = data["truth_neg_lep"]
	truth_pos_nu = data["truth_pos_nu"]
	truth_neg_nu = data["truth_neg_nu"]

	# training objects
	lep_pos_px = truth_pos_lep["px"]
	lep_pos_py = truth_pos_lep["py"]
	lep_pos_pz = truth_pos_lep["pz"]
	lep_pos_energy = truth_pos_lep["energy"]
	lep_neg_px = truth_neg_lep["px"]
	lep_neg_py = truth_neg_lep["py"]
	lep_neg_pz = truth_neg_lep["pz"]
	lep_neg_energy = truth_neg_lep["energy"]
	lep_pos_nu_px = truth_pos_nu["px"]
	lep_pos_nu_py = truth_pos_nu["py"]
	lep_neg_nu_px = truth_neg_nu["px"]
	lep_neg_nu_py = truth_neg_nu["py"]
	met_px = lep_pos_nu_px + lep_neg_nu_px
	met_py = lep_pos_nu_py + lep_neg_nu_py
	# pack them
	train_obj = np.column_stack(
		(
			lep_pos_px,
			lep_pos_py,
			lep_pos_pz,
			lep_pos_energy,
			lep_neg_px,
			lep_neg_py,
			lep_neg_pz,
			lep_neg_energy,
			met_px,
			met_py,
		)
	)
	print("Training objects shape:", train_obj.shape)

	# target objects
	w_pos = data["pos_w"]
	w_neg = data["neg_w"]
	#  pack them
	target_obj = np.column_stack(
		(
			w_pos["px"],
			w_pos["py"],
			w_pos["pz"],
			w_pos["energy"],
			w_neg["px"],
			w_neg["py"],
			w_neg["pz"],
			w_neg["energy"],
			w_pos["m"],
			w_neg["m"],
		)
	)
	print("Target objects shape:", target_obj.shape)

	return train_obj, target_obj

if __name__ == "__main__":
	from matplotlib import pyplot as plt
	data_path = "/root/data/mc20_truth_v4_SM.h5"
	train_obj, target_obj = load_data(data_path)
	w_pos_mass = target_obj[:, 8]
	w_neg_mass = target_obj[:, 9]
	plt.hist(w_pos_mass, bins=50, range=(0, 120), histtype='step', label='W+ mass')
	plt.hist(w_neg_mass, bins=50, range=(0, 120), histtype='bar', label='W- mass')
	plt.xlabel("W mass [GeV]")
	plt.ylabel("Entries")
	plt.legend()
	plt.savefig("w_mass.png")
	print("Train objects:", train_obj)
	print("Target objects:", target_obj)