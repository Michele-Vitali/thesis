#!/usr/bin/env python3
"""Safe, reversible organization for the P-Rob3 thesis repository.

Commands (from thesis root):
  python maintenance/reorganize_after_training.py audit
  python maintenance/reorganize_after_training.py apply
  python maintenance/reorganize_after_training.py apply --archive-legacy
  python maintenance/reorganize_after_training.py restore

This script intentionally DOES NOT move active TFDS RLDS, models, virtual
 environments, vendor repos, or change existing Python source imports.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_RAW = ROOT / "scripts" / "data"
NEW_RAW = ROOT / "data" / "raw" / "episodes"
LEGACY = ROOT / "rlds_dataset_builder"
LEGACY_ARCHIVE = ROOT / "archive" / "legacy" / "rlds_dataset_builder_old"
MANIFEST = ROOT / "maintenance" / "reorg_after_training.json"


def describe() -> None:
    print(f"Repository: {ROOT}")
    print(f"Raw HDF5 current: {OLD_RAW}")
    print(f"Raw HDF5 new:     {NEW_RAW}")
    print(f"Legacy root TFDS builder: {LEGACY}")
    if OLD_RAW.is_symlink():
        print(f"[OK] Compat symlink exists: {OLD_RAW} -> {os.readlink(OLD_RAW)}")
    elif OLD_RAW.is_dir():
        print("[PLAN] Move scripts/data -> data/raw/episodes")
        print("[PLAN] Install symlink scripts/data -> ../data/raw/episodes")
    else:
        print("[WARNING] scripts/data is absent")
    if LEGACY.is_dir():
        print("[OPTIONAL] Archive duplicate older .npy RLDS builder with --archive-legacy")
    else:
        print("[OK] No top-level legacy RLDS builder to archive")
    print("[PROTECTED] data/rlds/, data/splits/, runs/, adapters/, cache/,")
    print("            tools/, integrations/, assets/, openvla_env/, venv/, wandb/")
    print("[PROTECTED] No active OpenVLA files or path literals are rewritten")


def inspect_split_links(max_links_per_file: int = 10) -> tuple[int, list[str]]:
    """Try a representative sample of the external links in each split HDF5."""
    try:
        import h5py
    except ImportError:
        return 0, ["h5py not installed in this Python environment; skipped link check"]
    checked = 0
    errors: list[str] = []
    for path in sorted((ROOT / "data" / "splits").glob("**/*.hdf5")):
        try:
            with h5py.File(path, "r") as h5:
                if "data" not in h5:
                    errors.append(f"{path}: missing /data")
                    continue
                keys = sorted(h5["data"].keys())
                sample = keys[:max_links_per_file - 1]
                if keys and keys[-1] not in sample:
                    sample.append(keys[-1])
                for key in sample:
                    # h5py resolves ExternalLink targets when accessing group.
                    assert h5["data"][key]["actions_openvla"].shape[0] > 0
                    checked += 1
        except (OSError, KeyError, AssertionError, RuntimeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return checked, errors


def confirm_links(stage: str) -> None:
    checked, errors = inspect_split_links()
    print(f"{stage}: verified {checked} representative HDF5 split links")
    if errors:
        raise RuntimeError("Some existing HDF5 external links failed:\n" + "\n".join(errors))


def move_raw() -> bool:
    if OLD_RAW.is_symlink():
        if not NEW_RAW.is_dir() or OLD_RAW.resolve() != NEW_RAW.resolve():
            raise RuntimeError("Existing scripts/data symlink points to an unexpected location")
        print("[skip] raw move already done")
        return False
    if not OLD_RAW.is_dir():
        raise RuntimeError(f"Expected original directory is missing: {OLD_RAW}")
    if NEW_RAW.exists() or NEW_RAW.is_symlink():
        raise RuntimeError(f"Refusing to replace existing destination: {NEW_RAW}")
    NEW_RAW.parent.mkdir(parents=True, exist_ok=True)
    # os.rename is atomic on the same filesystem, and deliberately does not
    # silently start copying large HDF5 files across different filesystems.
    os.rename(OLD_RAW, NEW_RAW)
    try:
        OLD_RAW.symlink_to(Path("..") / "data" / "raw" / "episodes", target_is_directory=True)
        if OLD_RAW.resolve() != NEW_RAW.resolve():
            raise AssertionError("Compatibility symlink does not resolve")
    except Exception:
        if OLD_RAW.is_symlink():
            OLD_RAW.unlink()
        os.rename(NEW_RAW, OLD_RAW)
        raise
    print(f"[moved] {OLD_RAW.relative_to(ROOT)} -> {NEW_RAW.relative_to(ROOT)}")
    print(f"[linked] {OLD_RAW.relative_to(ROOT)} -> {os.readlink(OLD_RAW)}")
    return True


def archive_legacy() -> bool:
    if not LEGACY.exists():
        print("[skip] top-level legacy builder not found")
        return False
    if LEGACY.is_symlink():
        raise RuntimeError("Unexpected symlink at legacy builder location")
    if LEGACY_ARCHIVE.exists():
        raise RuntimeError(f"Archive destination already exists: {LEGACY_ARCHIVE}")
    LEGACY_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    os.rename(LEGACY, LEGACY_ARCHIVE)
    print(f"[moved] {LEGACY.relative_to(ROOT)} -> {LEGACY_ARCHIVE.relative_to(ROOT)}")
    return True


def apply(archive_old_builder: bool) -> None:
    if MANIFEST.exists():
        raise RuntimeError(f"Previous migration manifest exists: {MANIFEST}; inspect/restore first")
    print("== Before changes ==")
    confirm_links("Preflight")
    changed_raw = move_raw()
    changed_legacy = False
    try:
        confirm_links("Post-migration")
        if archive_old_builder:
            changed_legacy = archive_legacy()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "moved_raw": changed_raw,
            "archived_legacy": changed_legacy,
            "old_raw": str(OLD_RAW),
            "new_raw": str(NEW_RAW),
            "legacy": str(LEGACY),
            "legacy_archive": str(LEGACY_ARCHIVE),
        }, indent=2) + "\n", encoding="utf-8")
    except Exception:
        if changed_legacy and LEGACY_ARCHIVE.is_dir():
            os.rename(LEGACY_ARCHIVE, LEGACY)
        if changed_raw and OLD_RAW.is_symlink() and NEW_RAW.is_dir():
            OLD_RAW.unlink()
            os.rename(NEW_RAW, OLD_RAW)
        raise
    print(f"[OK] Saved restore manifest: {MANIFEST.relative_to(ROOT)}")
    print("[OK] Model checkpoints, RLDS, and existing Python paths unchanged")


def restore() -> None:
    if not MANIFEST.is_file():
        print("Nothing to restore: no migration manifest")
        return
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["archived_legacy"]:
        if LEGACY.exists() or not LEGACY_ARCHIVE.is_dir():
            raise RuntimeError("Legacy builder paths not in the expected state")
    if data["moved_raw"]:
        if not OLD_RAW.is_symlink() or not NEW_RAW.is_dir() or OLD_RAW.resolve() != NEW_RAW.resolve():
            raise RuntimeError("Raw paths not in the expected state; refusing unsafe restore")
    if data["archived_legacy"]:
        os.rename(LEGACY_ARCHIVE, LEGACY)
        print("[restored] legacy builder")
    if data["moved_raw"]:
        OLD_RAW.unlink()
        os.rename(NEW_RAW, OLD_RAW)
        print("[restored] raw HDF5 path")
        confirm_links("Restored")
    MANIFEST.unlink()
    print("[OK] restore complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit", "apply", "restore"))
    parser.add_argument("--archive-legacy", action="store_true", help="Also archive the unused old top-level .npy TFDS builder")
    args = parser.parse_args()
    if args.operation == "audit":
        describe()
        checked, errors = inspect_split_links()
        print(f"Existing split links checked: {checked}")
        for error in errors:
            print("[WARNING]", error)
    elif args.operation == "apply":
        apply(args.archive_legacy)
    else:
        restore()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
