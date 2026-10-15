#!/usr/bin/env python3
"""Prepare deterministic episode-level train/val/test and 5-fold manifests.

This does NOT modify source HDF5 files, build RLDS, or launch OpenVLA.
Store the resulting manifest with your experiment for reproducibility.

Usage (run from thesis root):
    python integrations/openvla/prepare_prob3_splits.py \
        --source hdf5_data/ --output data/splits/prob3_splits_seed42.json \
        --seed 42 --test-size 30 --val-size 30 --folds 5
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import h5py


def locate_hdf5_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError(f"Not an HDF5 file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(source.rglob("demo.hdf5"))
    if not files:
        raise RuntimeError(f"No demo.hdf5 under {source}")
    return files


def inventory_episodes(files: list[Path]) -> list[dict]:
    episodes = []
    used_ids = set()
    for path in files:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"Missing 'data' in {path}")
            for name in sorted(handle["data"].keys()):
                demo = handle["data"][name]
                if "actions_openvla" not in demo or "dones" not in demo or "rewards" not in demo:
                    raise ValueError(f"Incomplete episode {path}::{name}")
                length = len(demo["actions_openvla"])
                if length <= 1 or demo["actions_openvla"].shape[1:] != (7,):
                    raise ValueError(f"Invalid action shape in {path}::{name}")
                if len(demo["dones"]) != length or len(demo["rewards"]) != length:
                    raise ValueError(f"Mismatched lengths in {path}::{name}")
                if not bool(demo["dones"][-1]) or float(demo["rewards"][-1]) < 0.99:
                    raise ValueError(f"Non-successful episode in HDF5: {path}::{name}")
                episode_id = f"{path.parent.name}__{name}"
                if episode_id in used_ids:
                    raise ValueError(f"Duplicate episode ID: {episode_id}; rename source folders")
                used_ids.add(episode_id)
                episodes.append({
                    "id": episode_id,
                    "hdf5": str(path),
                    "demo": name,
                    "frames": length,
                })
    if not episodes:
        raise RuntimeError("No valid episodes found")
    return episodes


def make_manifest(episodes: list[dict], seed: int, test_size: int, val_size: int, folds: int) -> dict:
    total = len(episodes)
    if not (0 <= test_size < total):
        raise ValueError("test-size must be >=0 and smaller than total episode count")
    if not (0 <= val_size < total - test_size):
        raise ValueError("val-size must be >=0 and leave at least one train episode")
    development_count = total - test_size
    if folds and not (2 <= folds <= development_count):
        raise ValueError("folds must be 0 (disable) or between 2 and development episode count")

    # Shuffle episode IDs (never shuffle individual frames).
    ids = [ep["id"] for ep in episodes]
    random.Random(seed).shuffle(ids)
    test_ids = ids[:test_size]
    development_ids = ids[test_size:]
    val_ids = development_ids[:val_size]
    train_ids = development_ids[val_size:]

    fold_specs = []
    if folds:
        # Each development episode is in validation exactly once.
        # Distribute any remainder across the first few folds.
        sizes = [development_count // folds + (i < development_count % folds) for i in range(folds)]
        left = 0
        for i, size in enumerate(sizes):
            fold_val = development_ids[left:left + size]
            fold_train = development_ids[:left] + development_ids[left + size:]
            fold_specs.append({"fold": i, "train": fold_train, "val": fold_val})
            left += size

    assert set(train_ids).isdisjoint(val_ids)
    assert set(test_ids).isdisjoint(development_ids)
    assert len(set(ids)) == total
    return {
        "format": "prob3-episode-splits-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "counts": {
            "total": total,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "folds": folds,
        },
        "episodes": episodes,
        "split": {"train": train_ids, "val": val_ids, "test": test_ids},
        "folds": fold_specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic episode-level P-Rob3 split manifest")
    parser.add_argument("--source", required=True, type=Path, help="An HDF5 file or folder containing demo.hdf5 files")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5, help="Use 0 to disable cross-validation folds")
    args = parser.parse_args()

    files = locate_hdf5_files(args.source)
    episodes = inventory_episodes(files)
    manifest = make_manifest(episodes, args.seed, args.test_size, args.val_size, args.folds)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Manifest exists, not overwriting: {output}")
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Manifest: {output}")
    print(f"Raw HDF5 files: {len(files)}")
    print(f"Episodes: {len(episodes)}")
    print("Fixed split:", manifest["counts"])
    for fold in manifest["folds"]:
        print(f"  Fold {fold['fold'] + 1}: {len(fold['train'])} train, {len(fold['val'])} val")
    print("No HDF5 file was modified; no RLDS was generated.")


if __name__ == "__main__":
    main()
