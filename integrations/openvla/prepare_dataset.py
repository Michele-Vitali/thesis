from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import h5py

NAMES = ("dataset.hdf5", "demo.hdf5")


def source_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"Not an HDF5 file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(p.resolve() for p in source.rglob("*.hdf5") if p.name in NAMES)
    if not files:
        raise FileNotFoundError(f"No dataset.hdf5/demo.hdf5 below {source}")
    # Avoid accidentally using two aliases in the same run.
    by_folder = {}
    for file in files:
        if file.parent in by_folder:
            raise ValueError(f"Both dataset.hdf5 and demo.hdf5 in {file.parent}; select one")
        by_folder[file.parent] = file
    return files


def inventory(files: list[Path]) -> list[dict]:

    # Empty list ot store the episodes in it.
    episodes = []

    # Loop through the passed files.
    for file in files:
        with h5py.File(file, "r") as h5:
            # Check if the hdf5 file contains some data...
            if "data" not in h5:
                raise ValueError(f"Missing /data in {file}")

            # Loop throught the data keys...
            for name in sorted(h5["data"].keys()):

                # Grab the group's name.
                group = h5["data"][name]
                if not isinstance(group, h5py.Group):
                    continue

                # Loop through some keys...
                for field in ("actions_openvla", "rewards", "dones"):

                    # Check if field is present.
                    if field not in group:
                        raise ValueError(f"Missing {field} in {file}::{name}")

                # Check for data format.
                count = len(group["actions_openvla"])

                if count < 2 or group["actions_openvla"].shape[1:] != (7,):
                    raise ValueError(f"Invalid actions in {file}::{name}")
                
                if len(group["rewards"]) != count or len(group["dones"]) != count:
                    raise ValueError(f"Mismatched lengths in {file}::{name}")
                
                if not bool(group["dones"][-1]) or float(group["rewards"][-1]) < 0.99:
                    raise ValueError(f"Episode not successful in {file}::{name}")

                # If everything was okay => Save the episode!
                episodes.append({"id": f"episode_{len(episodes):06d}", "hdf5": str(file), "demo": name, "frames": count})

    # Fallback if there are no correct episodes...
    if not episodes:
        raise ValueError("No successful episodes")
    
    return episodes


def manifest_for(episodes: list[dict], seed: int, test_size: int, val_size: int, folds: int) -> dict:

    # Grab the episodes count.
    count = len(episodes)

    # Check for dataset's splits parameters.
    if min(test_size, val_size) < 0 or test_size + val_size >= count:
        raise ValueError("Invalid test/validation sizes")

    # Grab the episodes' ids.
    ids = [ep["id"] for ep in episodes]

    # Randomize the episode choice.
    random.Random(seed).shuffle(ids)

    # Initialize the lists for saving the episodes in their splits accordingly.
    test = ids[:test_size]
    development = ids[test_size:]
    val = development[:val_size]
    train = development[val_size:]

    if folds and not (2 <= folds <= len(development)):
        raise ValueError("Folds must be zero or between 2 and development episode count")
    
    fold_entries = []

    if folds:
        # Compute the folds' sizes.
        sizes = [len(development) // folds + (i < len(development) % folds) for i in range(folds)]
        start = 0

        for idx, size in enumerate(sizes):
            # Populate the fold's list.
            fold_val = development[start:start + size]
            fold_train = development[:start] + development[start + size:]
            fold_entries.append({"fold": idx, "train": fold_train, "val": fold_val})
            start += size

    return {
        "format": "prob3-episode-splits-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "counts": {"total": count, "train": len(train), "val": len(val), "test": len(test), "folds": folds},
        "episodes": episodes,
        "split": {"train": train, "val": val, "test": test},
        "folds": fold_entries,
    }


def create_view(destination: Path, selected: list[str], manifest: dict, source_attrs: dict) -> None:

    # Grab episodes' id.
    by_id = {ep["id"]: ep for ep in manifest["episodes"]}

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate episode IDs in view")

    # Creates a file and fails to do it if it already exists (caused by the "x" mode)
    with h5py.File(destination, "x") as h5:
        group = h5.create_group("data")

        for key, value in source_attrs.items():
            group.attrs[key] = value
        group.attrs["num_demos"] = len(selected)
        group.attrs["total"] = sum(by_id[item]["frames"] for item in selected)

        for idx, episode_id in enumerate(selected):
            ep = by_id[episode_id]
            relative = os.path.relpath(ep["hdf5"], destination.parent)
            group[f"demo_{idx}"] = h5py.ExternalLink(relative, f"/data/{ep['demo']}")
            
    with h5py.File(destination, "r") as h5:
        for idx, episode_id in enumerate(selected):
            ep = by_id[episode_id]
            if len(h5["data"][f"demo_{idx}"]["actions_openvla"]) != ep["frames"]:
                raise AssertionError(f"Invalid external link for {episode_id}")


def main() -> None:

    # Various cmd parameters...
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--expected", type=int, default=None, help="Assert exact number of episodes")
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    # Compute the dataset's directory.
    thesis = Path(__file__).resolve().parents[2]
    source = args.source or (thesis / "hdf5_data")

    # Check and eventually format the directory.
    if not source.is_absolute():
        source = (thesis / source).resolve()
    if args.source is None:
        files = source_files(source)
        files = [max(files, key=lambda p: p.stat().st_mtime_ns)]
    else:
        files = source_files(source)

    if len(files) != 1:
        raise ValueError("Select just one hdf5 file by using the --source arg...")

    # Analyze the saved episodes...
    episodes = inventory(files)

    # Check if the passed number of expected episodes corresponds to the actual computed one...
    if args.expected is not None and len(episodes) != args.expected:
        raise ValueError(f"Found {len(episodes)} episodes, expected {args.expected}")

    # Create the splits/folds.
    manifest = manifest_for(episodes, args.seed, args.test_size, args.val_size, args.folds)

    # Define the output path...
    output_root = args.output_root or files[0].parent / "splits"

    if not output_root.is_absolute():
        output_root = (thesis / output_root).resolve()

    destination = output_root
    destination.mkdir(parents=True, exist_ok=False)

    # Additional conversion from hdf5 to rlds.
    with h5py.File(files[0], "r") as h5:
        attrs = dict(h5["data"].attrs.items())

    # Note: The splits are just a re-buildable view computed from the full_dataset, so we do not occupy a lot of space!
    for split in ("train", "val", "test"):
        view = destination / f"{split}.hdf5"
        create_view(view, manifest["split"][split], manifest, attrs)
        print(f"{split:5s}: {len(manifest['split'][split]):4d} episodes -> {view}")
    
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest: {destination / 'manifest.json'}")
    print("External links: keep original HDF5 files in their current relative locations.")
    print("RLDS conversion is a separate step: no TFDS builder was run.")


if __name__ == "__main__":
    main()
