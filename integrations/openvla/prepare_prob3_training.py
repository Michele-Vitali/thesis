#!/usr/bin/env python3
"""Prepare episode-level splits and optionally publish a train-only RLDS for P-Rob3.

Must live beside `prepare_prob3_splits.py` and `build_dataset.py` inside:
  thesis/integrations/openvla/

Usage from thesis root (openvla_env active):
  python integrations/openvla/prepare_prob3_training.py --publish
  python integrations/openvla/prepare_prob3_training.py --source hdf5_data/RUN/demo.hdf5 --publish

The source HDF5 files are never changed or duplicated. The split HDF5 files
contain HDF5 EXTERNAL LINKS to original episode groups, so never move/delete
source demo.hdf5 files while using these split views.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py

from prepare_prob3_splits import inventory_episodes, locate_hdf5_files, make_manifest


def find_latest_run(project_root: Path) -> Path:
    candidates = sorted((project_root / "hdf5_data").glob("prob3_pick_place_*/demo.hdf5"))
    if not candidates:
        raise FileNotFoundError("Nessun HDF5 in hdf5_data/prob3_pick_place_*/demo.hdf5")
    return candidates[-1]



def build_view(view_path: Path, selected_ids: list[str], inventory: list[dict], task: str) -> None:
    """
    Create an HDF5 split using relative External Links.

    The split and its source datasets remain connected
    when the whole project directory is moved intact.
    """
    import os

    by_id = {episode["id"]: episode for episode in inventory}

    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"Duplicate IDs in split {view_path.stem}")

    view_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(view_path, "w") as handle:
        group = handle.create_group("data")

        group.attrs["task"] = task
        group.attrs["num_demos"] = len(selected_ids)

        for index, episode_id in enumerate(selected_ids):
            episode = by_id[episode_id]

            source = Path(episode["hdf5"]).expanduser().resolve(strict=True)

            relative_source = os.path.relpath(source, start=view_path.parent.resolve())

            group[f"demo_{index}"] = h5py.ExternalLink(relative_source, f"/data/{episode['demo']}")

    # Validate every external link immediately.
    with h5py.File(view_path, "r") as handle:
        for index, episode_id in enumerate(selected_ids):
            expected_length = by_id[episode_id]["frames"]

            actual_length = len(handle["data"][f"demo_{index}"]["actions_openvla"])

            if actual_length != expected_length:
                raise AssertionError(f"Invalid HDF5 link: {episode_id}: {actual_length} != {expected_length}")



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split 300 successi P-Rob3 senza copiare gli HDF5; pubblica solo il training"
    )
    parser.add_argument("--source", type=Path, default=None,
                        help="HDF5 o cartella contenente demo.hdf5; default: run più recente")
    parser.add_argument("--expected", type=int, default=300,
                        help="Numero esatto di episodi richiesti (default: 300)")
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5,
                        help="Prepara anche i manifest dei fold (non esegue 5 training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Cartella base esperimenti, default: thesis/data/splits")
    parser.add_argument("--publish", action="store_true",
                        help="Converte e pubblica RLDS composto SOLO dallo split train")
    args = parser.parse_args()

    thesis = Path(__file__).resolve().parents[2]
    source = args.source or find_latest_run(thesis)
    source = (source if source.is_absolute() else thesis / source).resolve()
    originals = locate_hdf5_files(source)
    inventory = inventory_episodes(originals)
    if len(inventory) != args.expected:
        raise RuntimeError(
            f"Trovati {len(inventory)} episodi, ma --expected={args.expected}. "
            "Seleziona il run giusto con --source e non includere esperimenti vecchi."
        )

    manifest = make_manifest(
        inventory, seed=args.seed, test_size=args.test_size,
        val_size=args.val_size, folds=args.folds,
    )
    output_root = args.output_root or (thesis / "rlds_data" / "splits")
    output_root = (output_root if output_root.is_absolute() else thesis / output_root).resolve()
    destination = output_root / f"prob3_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=False)

    manifest_path = destination / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)

    # The original HDF5 builder looks for task under /data attrs as fallback.
    # If instruction per demo exists, it is preserved through the external link.
    with h5py.File(originals[0], "r") as handle:
        raw_task = handle["data"].attrs.get("task", "Pick up the cube and place it straight at the center of the table")
    task = raw_task.decode("utf-8") if isinstance(raw_task, bytes) else str(raw_task)

    views = {}
    for split in ("train", "val", "test"):
        view = destination / f"{split}.hdf5"
        build_view(view, manifest["split"][split], inventory, task)
        views[split] = view

    total_frames = sum(episode["frames"] for episode in inventory)
    print(f"\nORIGINALI: {len(originals)} file HDF5, {len(inventory)} successi, {total_frames} frame")
    print(f"SPLIT:     train={len(manifest['split']['train'])}, "
          f"val={len(manifest['split']['val'])}, test={len(manifest['split']['test'])}")
    print(f"MANIFEST:  {manifest_path}")
    for key, view in views.items():
        print(f"{key.upper():8s}: {view}")
    print("NOTA: le viste HDF5 sono link esterni; NON spostare gli HDF5 originali.")
    print("NOTA: gli split val/test NON entreranno nella normalizzazione o nell'RLDS di training.")
    print("NOTA: il loader OpenVLA attuale prende un suo 5% di validation interna dal train.")

    if args.publish:
        builder = thesis / "integrations" / "openvla" / "build_dataset.py"
        if not builder.is_file():
            raise FileNotFoundError(f"Builder automatico non trovato: {builder}")
        command = [sys.executable, str(builder), "--source", str(views["train"]), "--publish"]
        print("\nBUILD/PUBLISH train-only:", " ".join(command), flush=True)
        subprocess.run(command, cwd=thesis, check=True)
        print("\nPRONTO. Prossimo step: preflight_prob3.py prima del training.")
    else:
        print("\nSplit generati, RLDS attivo NON modificato. Usa --publish dopo controllo.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERRORE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
