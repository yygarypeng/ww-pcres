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
	truth_lead_lep = data["truth_lead_lep"]
	truth_sublead_lep = data["truth_sublead_lep"]
	truth_lead_nu = data["truth_lead_nu"]
	truth_sublead_nu = data["truth_sublead_nu"]
	truth_lead_nu_p4 = data["truth_lead_nu"]["p4"]
	truth_sublead_nu_p4 = data["truth_sublead_nu"]["p4"]
	truth_met_pt = np.sqrt(np.square((truth_lead_nu_p4 + truth_sublead_nu_p4)[...,0:2]).sum(axis=-1))
	# cut!!
	cut_pre_pt_lead = truth_lead_lep["pt"] > 22
	cut_pre_pt_sub = truth_sublead_lep["pt"] > 15
	cut_pre_dilep_m = (
		np.square(truth_lead_lep["energy"] + truth_sublead_lep["energy"])
		- np.square(truth_lead_lep["px"] + truth_sublead_lep["px"])
		- np.square(truth_lead_lep["py"] + truth_sublead_lep["py"])
		- np.square(truth_lead_lep["pz"] + truth_sublead_lep["pz"])
		> 10**2
	)
	cut_pre_pt_miss = truth_met_pt > 20
	pre_cut = cut_pre_pt_lead & cut_pre_pt_sub & cut_pre_dilep_m & cut_pre_pt_miss
	del (cut_pre_pt_lead, cut_pre_pt_sub, cut_pre_dilep_m, cut_pre_pt_miss)

	# training objects
	lead_lep_px = truth_lead_lep["px"][pre_cut]
	lead_lep_py = truth_lead_lep["py"][pre_cut]
	lead_lep_pz = truth_lead_lep["pz"][pre_cut]
	lead_lep_energy = truth_lead_lep["energy"][pre_cut]
	sublead_lep_px = truth_sublead_lep["px"][pre_cut]
	sublead_lep_py = truth_sublead_lep["py"][pre_cut]
	sublead_lep_pz = truth_sublead_lep["pz"][pre_cut]
	sublead_lep_energy = truth_sublead_lep["energy"][pre_cut]
	lead_nu_px = truth_lead_nu["px"][pre_cut]
	lead_nu_py = truth_lead_nu["py"][pre_cut]
	sublead_nu_px = truth_sublead_nu["px"][pre_cut]
	sublead_nu_py = truth_sublead_nu["py"][pre_cut]
	met_px = lead_nu_px + sublead_nu_px
	met_py = lead_nu_py + sublead_nu_py
	# pack them
	train_obj = np.column_stack(
		(
			lead_lep_px,
			lead_lep_py,
			lead_lep_pz,
			lead_lep_energy,
			sublead_lep_px,
			sublead_lep_py,
			sublead_lep_pz,
			sublead_lep_energy,
			met_px,
			met_py,
		)
	)
	print("Training objects shape:", train_obj.shape)

	# target objects
	w_lead = data["lead_w"]
	w_sublead = data["sublead_w"]
	#  pack them
	target_obj = np.column_stack(
		(
			w_lead["px"][pre_cut],
			w_lead["py"][pre_cut],
			w_lead["pz"][pre_cut],
			w_lead["energy"][pre_cut],
			w_sublead["px"][pre_cut],
			w_sublead["py"][pre_cut],
			w_sublead["pz"][pre_cut],
			w_sublead["energy"][pre_cut],
			w_lead["m"][pre_cut],
			w_sublead["m"][pre_cut],
		)
	)
	print("Target objects shape:", target_obj.shape)

	return train_obj, target_obj