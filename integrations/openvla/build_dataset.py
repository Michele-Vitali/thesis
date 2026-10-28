"""Build and validate P-Rob3 HDF5 -> TFDS/RLDS with one command.

Run from anywhere using the project's openvla_env Python.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """Expected position: <thesis>/integrations/openvla/build_dataset.py."""
    return Path(__file__).resolve().parents[2]

SPLIT_FILES = {"train": "train.hdf5", "val": "validation.hdf5", "test": "test.hdf5"}

def find_source(root: Path, source: str | None) -> Path:
    """Require a prepared directory, never infer or create split assignments."""

    if source is None:

        hdf5_root = root / "hdf5_data"
        candidates = sorted(
            path for path in hdf5_root.glob("*/splits")
            if path.is_dir() and all((path / filename).is_file() for filename in SPLIT_FILES.values())
        )
        if not candidates:
            raise FileNotFoundError("No prepared HDF5 splits; run prepare_prob3_dataset.py")

        def modification_time(path):
            manifest = path / "manifest.json"
            target = manifest if manifest.is_file() else path

            return target.stat().st_mtime_ns

        return max(candidates, key=modification_time).resolve()
        
    path = Path(source).expanduser()

    if not path.is_absolute():

        path = (Path.cwd() / path).resolve()

    if not path.is_dir():
        raise ValueError(f"Invalid HDF5 splits directory: {path}")

    missing = [filename for filename in SPLIT_FILES.values()
               if not (path / filename).is_file()]

    if missing:
        raise ValueError(f"Missing HDF5 split files in {path}: {missing}")

    return path



def hdf5_inventory(source: Path) -> dict[str, dict[tuple[str, str], int]]:
    import h5py
    import numpy as np
    inventories = {}
    for split, filename in SPLIT_FILES.items():
        path = (source / filename).resolve()
        inventory = {}
        with h5py.File(path, "r") as h5:
            if "data" not in h5:
                raise ValueError(f"Missing /data in {path}")
            for name, ep in h5["data"].items():
                for field in ("actions_openvla", "rewards", "dones"):
                    if field not in ep:
                        raise ValueError(f"Missing {field}: {path}::{name}")
                if "obs" not in ep:
                    raise ValueError(f"Missing obs: {path}::{name}")
                obs = ep["obs"]
                if any(f not in obs for f in ("agentview_rgb", "joint_states", "gripper_states")):
                    raise ValueError(f"Missing observation field: {path}::{name}")
                actions = ep["actions_openvla"][:]
                count = len(actions)
                if count < 2 or actions.shape[1:] != (7,):
                    raise ValueError(f"Invalid actions: {path}::{name}")
                if obs["agentview_rgb"].shape != (count, 256, 256, 3):
                    raise ValueError(f"Invalid images: {path}::{name}")
                if obs["joint_states"].shape != (count, 6):
                    raise ValueError(f"Invalid joints: {path}::{name}")
                if obs["gripper_states"].shape[0] != count:
                    raise ValueError(f"Invalid gripper: {path}::{name}")
                if len(ep["rewards"]) != count or len(ep["dones"]) != count:
                    raise ValueError(f"Invalid terminal arrays: {path}::{name}")
                if not np.isfinite(actions).all() or not np.isin(actions[:, 6], (0.0, 1.0)).all():
                    raise ValueError(f"Invalid actions content: {path}::{name}")
                if not bool(ep["dones"][-1]) or float(ep["rewards"][-1]) < 0.99:
                    raise ValueError(f"Invalid episode ending: {path}::{name}")
                inventory[(str(path), str(name))] = count
        
        # Allow empty validation or test splits when creating overfit datasets.
        if not inventory and split == "train":
            raise ValueError(f"Split {split} is empty")
        inventories[split] = inventory
    # Ensure the same original episode does not appear twice across splits.
    # For external-link views, retrieve the target filename and group path.
    original_targets = set()
    for split, filename in SPLIT_FILES.items():
        path = source / filename
        with h5py.File(path, "r") as h5:
            for name in h5["data"]:
                link = h5["data"].get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    original = (path.parent / link.filename).resolve()
                    identity = (str(original), link.path)
                else:
                    identity = (str(path.resolve()), f"/data/{name}")
                if identity in original_targets:
                    raise ValueError(f"Episode shared across HDF5 splits: {identity}")
                original_targets.add(identity)
    return inventories


def decode_text(item: object) -> str:
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if hasattr(item, "item"):
        item = item.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
    return str(item)


def verify_rlds(dataset_dir: Path, inventories: dict[str, dict[tuple[str, str], int]]) -> None:
    """Check all three RLDS splits, identity, episode lengths and exact actions."""
    import h5py
    import numpy as np
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    for split, expected in inventories.items():
        # Skip verification for empty splits.
        if not expected:
            print(f"SKIPPED {split}: split is empty", flush=True)
            continue
            
        info = builder.info.splits.get(split)
        if info is None or int(info.num_examples) != len(expected):
            raise AssertionError(f"Wrong RLDS episode count for {split}")
        observed = set()
        for episode in tfds.as_numpy(builder.as_dataset(split=split)):
            metadata = episode["episode_metadata"]
            key = (str(Path(decode_text(metadata["file_path"])).resolve()),
                   decode_text(metadata["demo_name"]))
            if key not in expected or key in observed:
                raise AssertionError(f"Unexpected/duplicate RLDS episode in {split}: {key}")
            with h5py.File(key[0], "r") as h5:
                original = h5["data"][key[1]]["actions_openvla"]
                count = 0
                for step in episode["steps"]:
                    if step["observation"]["image"].shape != (256, 256, 3):
                        raise AssertionError(f"Unexpected image shape: {key}")
                    if step["observation"]["state"].shape != (7,):
                        raise AssertionError(f"Unexpected state shape: {key}")
                    if step["action"].shape != (7,) or not np.allclose(
                        step["action"], original[count], atol=1e-6, rtol=0.0
                    ):
                        raise AssertionError(f"Action mismatch: {key} step {count}")
                    if not np.isfinite(step["observation"]["state"]).all():
                        raise AssertionError(f"Nonfinite state: {key}")
                    last = count == expected[key] - 1
                    if bool(step["is_first"]) != (count == 0):
                        raise AssertionError(f"Wrong is_first: {key}")
                    if bool(step["is_last"]) != last or bool(step["is_terminal"]) != last:
                        raise AssertionError(f"Wrong terminal flags: {key}")
                    if float(step["discount"]) != (0.0 if last else 1.0):
                        raise AssertionError(f"Wrong discount: {key}")
                    if not decode_text(step["language_instruction"]).strip():
                        raise AssertionError(f"Missing instruction: {key}")
                    count += 1
                if count != expected[key]:
                    raise AssertionError(f"Wrong number of steps: {key}")
            observed.add(key)
        if observed != set(expected):
            raise AssertionError(f"Missing RLDS episodes in {split}: {set(expected) - observed}")
        print(f"VERIFIED {split}: {len(observed)} episodes, {sum(expected.values())} steps", flush=True)



def publish_dataset(root: Path, staged_dataset: Path) -> tuple[Path, Path | None]:
    """Promote only a validated dataset; move any existing version into an archive."""
    active = root / "rlds_data" / "rlds" / "prob3_pick_place"
    archive_root = root / "rlds_data" / "rlds_archive"
    active.parent.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    old_backup = None
    if active.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        old_backup = archive_root / f"prob3_pick_place_{stamp}"
        print(f"Archivio RLDS precedente: {old_backup}", flush=True)
        shutil.move(str(active), str(old_backup))

    try:
        shutil.move(str(staged_dataset), str(active))
    except Exception:
        # Best-effort restore: do not silently leave OpenVLA without the old dataset.
        if old_backup is not None and old_backup.exists() and not active.exists():
            shutil.move(str(old_backup), str(active))
        raise
    return active, old_backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, verify and optionally publish P-Rob3 RLDS from HDF5."
    )
    parser.add_argument("--source", help="Directory containing prepared train.hdf5, val.hdf5 and test.hdf5; default: latest prepared split.")
    parser.add_argument("--publish", action="store_true", help="Attiva il dataset per OpenVLA; fa backup del precedente.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra le operazioni senza convertire o spostare dati.")
    args = parser.parse_args()

    root = project_root()
    builder_dir = root / "tools" / "rlds_dataset_builder" / "prob3_pick_place"
    if not (builder_dir / "prob3_pick_place_dataset_builder.py").is_file():
        raise FileNotFoundError(f"Builder mancante: {builder_dir}")

    source = find_source(root, args.source)
    staged_root = root / "rlds_data" / "rlds_builds" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
    )
    staged_dataset = staged_root / "prob3_pick_place" / "1.0.0"
    print(f"Python:      {sys.executable}")
    print(f"HDF5 input:  {source}")
    print(f"Builder:     {builder_dir}")
    print(f"Output:      {staged_dataset}")
    print(f"Pubblica:    {'sì' if args.publish else 'no (solo prova)'}", flush=True)
    if args.dry_run:
        print("DRY RUN: nessun file modificato.")
        return 0

    inventories = hdf5_inventory(source)
    print("HDF5 inventories: ", {k: len(v) for k,v in inventories.items()}, flush=True)

    tfds_executable = Path(sys.executable).with_name("tfds")
    if not tfds_executable.is_file():
        cli = shutil.which("tfds")
        if cli is None:
            raise RuntimeError("Comando tfds non trovato: attiva thesis/openvla_env.")
        tfds_executable = Path(cli)

    staged_root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PROB3_HDF5_ROOT"] = str(source)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # TFDS conversion/validation uses the CPU. This does NOT change training configuration.
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    cmd = [str(tfds_executable), "build", str(builder_dir), "--data_dir", str(staged_root)]
    print("\nConversione TFDS/RLDS...", flush=True)
    subprocess.run(cmd, env=env, cwd=root, check=True)

    print("\nValidazione RLDS vs HDF5...", flush=True)
    # Keep TensorFlow off the GPU when running validation too.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    verify_rlds(staged_dataset, inventories)

    if args.publish:
        active, backup = publish_dataset(root, staged_dataset.parent)
        print(f"\nDATASET ATTIVO: {active / '1.0.0'}")
        if backup is not None:
            print(f"Dataset precedente salvato in: {backup}")
        print("IMPORTANTE: riesegui il preflight prima del training per ricalcolare le statistiche.")
    else:
        print(f"\nDataset verificato e NON pubblicato: {staged_dataset}")
        print("Per farlo usare a OpenVLA, ripeti il comando con --publish.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\nERRORE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc