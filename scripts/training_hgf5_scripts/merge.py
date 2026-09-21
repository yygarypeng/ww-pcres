import h5py

channel = "ggF"  # ggF, VBF

# The training loader reads one top-level group per split, named <channel>_<split>.
# "validate" is the name of the source file; "val" is the name of the group.
splits = {"train": "train", "val": "validate", "test": "test"}
out_path = f"./mc20_qe_v61_recotruth_{channel}_merged.h5"

with h5py.File(out_path, "w") as fout:
    for split, file_split in splits.items():
        src_path = f"./mc20_qe_v61_recotruth_{channel}_{file_split}.h5"
        with h5py.File(src_path, "r") as fin:
            # copy the entire source file hierarchy under a group named <channel>_<split>
            fin.copy(fin, fout, name=f"{channel}_{split}")
        print(f"Copied {src_path} into {channel}_{split}")

print("Merged into", out_path)
