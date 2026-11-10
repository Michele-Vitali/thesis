import h5py, collections
from pathlib import Path
import tensorflow_datasets as tfds

ROOT = Path.home() / "Scrivania/thesis"
RLDS_DIR = ROOT / "rlds_data/rlds_base/prob3_pick_place"
HDF5_SPLITS = ROOT / "hdf5_data/v3.0_no_randomization_20261109_142456_559122/splits"

versions = sorted(p for p in RLDS_DIR.iterdir() if p.is_dir())
for v in versions:
    b = tfds.builder_from_directory(str(v))
    print(v.name, "| versione:", b.info.version,
          "| episodi:", {k: s.num_examples for k, s in b.info.splits.items()})

latest = versions[-1]            # quella che OpenVLA caricherebbe
b = tfds.builder_from_directory(str(latest))
print("\nVersione usata dal training:", latest.name)

for split, fname in [("train", "train.hdf5"), ("validation", "validation.hdf5")]:
    paths, n_ep, instr = collections.Counter(), 0, None
    for ep in b.as_dataset(split=split):          # legge solo i metadati, non decodifica le immagini
        n_ep += 1
        paths[ep["episode_metadata"]["file_path"].numpy().decode()] += 1
        if instr is None:
            instr = next(iter(ep["steps"]))["language_instruction"].numpy().decode()

    with h5py.File(HDF5_SPLITS / fname, "r") as f:
        n_demos = len(f["data"])
        n_steps = sum(f["data"][d]["actions_openvla"].shape[0] for d in f["data"])

    print(f"\n[{split}] episodi RLDS: {n_ep} | demo nell'HDF5: {n_demos} | frame nell'HDF5: {n_steps}")
    print(f"[{split}] file di origine: {dict(paths)}")
    print(f"[{split}] istruzione: {instr!r}")
    