#!/usr/bin/env python3
"""Build and validate P-Rob3 HDF5 -> TFDS/RLDS with one command.

Install at thesis/integrations/openvla/build_dataset.py.
Run from anywhere using the project's openvla_env Python.

The existing TFDS builder is reused without editing its source.
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


def find_source(root: Path, source: str | None, all_runs: bool) -> Path:
    if source is not None and all_runs:
        raise ValueError("Usa --source oppure --all, non entrambi.")

    raw_dir = root / "scripts" / "data"
    if source is not None:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file() and path.suffix.lower() in (".hdf5", ".h5"):
            return path
        if path.is_dir():
            if not list(path.rglob("demo.hdf5")):
                raise ValueError(f"Nessun demo.hdf5 nella cartella: {path}")
            return path
        raise ValueError(f"Sorgente HDF5 non valida: {path}")

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Cartella dei dati mancante: {raw_dir}")

    paths = sorted(raw_dir.glob("*/demo.hdf5"), key=lambda p: (p.parent.name, p.name))
    if not paths:
        raise FileNotFoundError(f"Nessun HDF5 trovato in {raw_dir}/*/demo.hdf5")
    if all_runs:
        return raw_dir
    return paths[-1].resolve()


def hdf5_inventory(source: Path) -> dict[tuple[str, str], int]:
    """Ground truth used to check TFDS counts and the exact actions after conversion."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("L'ambiente corrente non ha h5py/numpy.") from exc

    if source.is_file():
        files = [source]
    else:
        files = sorted(source.rglob("demo.hdf5"))

    inventory: dict[tuple[str, str], int] = {}
    for file in files:
        with h5py.File(file, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"Gruppo data mancante: {file}")
            for demo_name, group in handle["data"].items():
                required = ("actions_openvla", "rewards", "dones")
                for field in required:
                    if field not in group:
                        raise ValueError(f"{file}/{demo_name}: campo {field} mancante")
                if "obs" not in group or "agentview_rgb" not in group["obs"]:
                    raise ValueError(f"{file}/{demo_name}: obs/agentview_rgb mancante")
                if "joint_states" not in group["obs"] or "gripper_states" not in group["obs"]:
                    raise ValueError(f"{file}/{demo_name}: stati robot/gripper mancanti")

                act = group["actions_openvla"]
                count = len(act)
                if count < 2 or act.shape[1:] != (7,):
                    raise ValueError(f"{file}/{demo_name}: azioni inattese {act.shape}")
                if group["obs"]["agentview_rgb"].shape != (count, 256, 256, 3):
                    raise ValueError(f"{file}/{demo_name}: immagini non allineate")
                if len(group["rewards"]) != count or len(group["dones"]) != count:
                    raise ValueError(f"{file}/{demo_name}: reward/done non allineati")
                if not np.all(np.isfinite(act[:])):
                    raise ValueError(f"{file}/{demo_name}: azioni con NaN/Inf")
                if not np.all(np.isin(act[:, 6], (0.0, 1.0))):
                    raise ValueError(f"{file}/{demo_name}: gripper non binario")
                if not bool(group["dones"][-1]):
                    raise ValueError(f"{file}/{demo_name}: demo senza done terminale")
                if float(group["rewards"][-1]) < 0.99:
                    raise ValueError(f"{file}/{demo_name}: reward terminale non riuscita")

                key = (str(file.resolve()), str(demo_name))
                if key in inventory:
                    raise ValueError(f"ID episodio duplicato: {key}")
                inventory[key] = count

    if not inventory:
        raise ValueError("Nessun episodio HDF5 valido trovato")
    return inventory


def decode_text(item: object) -> str:
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if hasattr(item, "item"):
        item = item.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
    return str(item)


def verify_rlds(dataset_dir: Path, source_inventory: dict[tuple[str, str], int]) -> None:
    """Read every TFDS episode and cross-check identity, length, flags and actions."""
    import h5py
    import numpy as np
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    split = builder.info.splits.get("train")
    if split is None or int(split.num_examples) != len(source_inventory):
        raise AssertionError("Split train assente o numero episodi errato")

    observed: dict[tuple[str, str], int] = {}
    for episode in tfds.as_numpy(builder.as_dataset(split="train")):
        metadata = episode["episode_metadata"]
        key = (
            str(Path(decode_text(metadata["file_path"])).resolve()),
            decode_text(metadata["demo_name"]),
        )
        if key in observed:
            raise AssertionError(f"Episodio duplicato in RLDS: {key}")
        if key not in source_inventory:
            raise AssertionError(f"Episodio RLDS non previsto: {key}")

        length = 0
        hdf5_file, demo_name = key
        with h5py.File(hdf5_file, "r") as h5:
            reference = h5["data"][demo_name]["actions_openvla"]
            for step in episode["steps"]:
                image = step["observation"]["image"]
                state = step["observation"]["state"]
                action = step["action"]

                if image.shape != (256, 256, 3) or state.shape != (7,) or action.shape != (7,):
                    raise AssertionError(f"Shape errata nell'episodio {demo_name} al frame {length}")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise AssertionError(f"NaN/Inf in {demo_name} al frame {length}")
                if not np.allclose(action, reference[length], atol=1e-6, rtol=0.0):
                    raise AssertionError(f"Azione non corrispondente all'HDF5: {demo_name} frame {length}")
                if bool(step["is_first"]) != (length == 0):
                    raise AssertionError(f"is_first errato: {demo_name} frame {length}")
                last = length == source_inventory[key] - 1
                if bool(step["is_last"]) != last or bool(step["is_terminal"]) != last:
                    raise AssertionError(f"Flag terminali errati: {demo_name} frame {length}")
                if float(step["discount"]) != (0.0 if last else 1.0):
                    raise AssertionError(f"Discount errato: {demo_name} frame {length}")
                if not decode_text(step["language_instruction"]).strip():
                    raise AssertionError(f"Istruzione vuota: {demo_name} frame {length}")
                length += 1

        if length != source_inventory[key]:
            raise AssertionError(f"Lunghezza RLDS errata: {demo_name} {length}/{source_inventory[key]}")
        observed[key] = length
        print(f"  OK {demo_name}: {length} frame — {Path(hdf5_file).parent.name}", flush=True)

    if observed.keys() != source_inventory.keys():
        raise AssertionError(f"Episodi mancanti: {source_inventory.keys() - observed.keys()}")
    print(f"\nVALIDAZIONE SUPERATA: {len(observed)} episodi, {sum(observed.values())} frame.", flush=True)


def publish_dataset(root: Path, staged_dataset: Path) -> tuple[Path, Path | None]:
    """Promote only a validated dataset; move any existing version into an archive."""
    active = root / "data" / "rlds" / "prob3_pick_place"
    archive_root = root / "data" / "rlds_archive"
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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source", help="HDF5 specifico o directory con demo.hdf5; default: run più recente.")
    sources.add_argument("--all", action="store_true", help="Combina tutti i demo.hdf5 di scripts/data/.")
    parser.add_argument("--publish", action="store_true", help="Attiva il dataset per OpenVLA; fa backup del precedente.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra le operazioni senza convertire o spostare dati.")
    args = parser.parse_args()

    root = project_root()
    builder_dir = root / "tools" / "rlds_dataset_builder" / "prob3_pick_place"
    if not (builder_dir / "prob3_pick_place_dataset_builder.py").is_file():
        raise FileNotFoundError(f"Builder mancante: {builder_dir}")

    source = find_source(root, args.source, args.all)
    staged_root = root / "data" / "rlds_builds" / (
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

    inventory = hdf5_inventory(source)
    print(f"\nHDF5 validati: {len(inventory)} episodi, {sum(inventory.values())} frame.", flush=True)

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
    verify_rlds(staged_dataset, inventory)

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
