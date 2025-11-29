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

	def col(a):
		return a.reshape(a.shape[0], -1)

	# training features
	lep_pos_px = data["pos_lep"]["px"]
	lep_pos_py = data["pos_lep"]["py"]
	lep_pos_pz = data["pos_lep"]["pz"]
	lep_pos_energy = data["pos_lep"]["energy"]
	lep_neg_px = data["neg_lep"]["px"]
	lep_neg_py = data["neg_lep"]["py"]
	lep_neg_pz = data["neg_lep"]["pz"]
	lep_neg_energy = data["neg_lep"]["energy"]

	met_px = data["met"]["px"]
	met_py = data["met"]["py"]

	jet_px = data["jets"]["px"]
	jet_py = data["jets"]["py"]
	jet_pz = data["jets"]["pz"]
	jet_energy = data["jets"]["energy"]
	jet_pt = data["jets"]["pt"]
	jet_btag = data["jets"]["btag"]
	n_jets = data["jets"]["n_jets"]
	n_bjets = data["jets"]["n_bjets"]

	# pack them
	train_obj = np.concat([
		col(lep_pos_px),
		col(lep_pos_py),
		col(lep_pos_pz),
		col(lep_pos_energy),
		col(lep_neg_px),
		col(lep_neg_py),
		col(lep_neg_pz),
		col(lep_neg_energy),
		col(met_px),
		col(met_py),
		# col(jet_px),
		# col(jet_py),
		# col(jet_pz),
		# col(jet_energy),
		# col(jet_pt),
		# col(jet_btag),
		# col(n_jets),
		# col(n_bjets),
	], axis=-1)
	print("Training objects shape:", train_obj.shape)

	# target objects
	# pack them
	target_obj = np.concatenate([
		col(data["truth_pos_w"]["px"]),
		col(data["truth_pos_w"]["py"]),
		col(data["truth_pos_w"]["pz"]),
		col(data["truth_pos_w"]["energy"]),
		col(data["truth_neg_w"]["px"]),
		col(data["truth_neg_w"]["py"]),
		col(data["truth_neg_w"]["pz"]),
		col(data["truth_neg_w"]["energy"]),
		col(data["truth_pos_w"]["m"]),
		col(data["truth_neg_w"]["m"]),
	], axis=-1)
	print("Target objects shape:", target_obj.shape)

	return train_obj, target_obj

if __name__ == "__main__":
	from matplotlib import pyplot as plt
	data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_ggF_train.h5"
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