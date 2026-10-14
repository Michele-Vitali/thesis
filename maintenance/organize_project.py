#!/usr/bin/env python3
"""Conservative, reversible organizational cleanup for the P-Rob3 thesis.

Run from any directory:
  python maintenance/organize_project.py audit
  python maintenance/organize_project.py safe --apply
  python maintenance/organize_project.py restore

During live training, only a small legacy exporter and structure.txt may move.
Never edits active environments, datasets, model repositories or run outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "archive" / "cleanup_moves.json"

PROTECTED = [
    "assets", "scripts/data", "data/raw", "data/rlds", "data/rlds_candidate",
    "cache", "runs", "adapters", "integrations/openvla", "tools",
    "openvla_env", "venv", "wandb", "preflight", "scripts/main.py",
    "scripts/custom_env.py", "scripts/custom_robot.py", "scripts/custom_osc.py",
    "scripts/movements.py", "scripts/settings.py", "scripts/hdf5_recorder.py",
]

CREATE_DIRS = ["docs", "docs/guides", "docs/architecture", "archive/legacy", "data/splits"]
MOVE_CANDIDATES = [
    ("scripts/export_lerobot_to_rlds_raw.py", "archive/legacy/export_lerobot_to_rlds_raw.py"),
    ("structure.txt", "docs/structure_original.txt"),
]

def print_overview() -> None:
    print(f"Project: {ROOT}")
    print("\nIMPORTANT: active training paths WILL NOT be moved or modified:")
    for name in PROTECTED:
        if (ROOT / name).exists():
            print(f"  [PROTECTED] {name}")
    print("\nPossible legacy duplicates (inspection only, NOT moved):")
    candidates = [
        "rlds_dataset_builder/prob3_pick_and_place/prob3_pick_place_dataset_builder.py",
        "tools/rlds_dataset_builder/prob3_pick_place/prob3_pick_place_dataset_builder.py",
        "scripts/inspect_hdf5.py", "scripts/inspect_rlds.py",
        "tools/rlds_dataset_builder/rlds_env",
    ]
    for name in candidates:
        if (ROOT / name).exists():
            print(f"  [REVIEW] {name}")
    print("\nSafe cleanup candidates:")
    for src, dst in MOVE_CANDIDATES:
        if (ROOT / src).exists():
            print(f"  {src}  ->  {dst}")
    print("\nNo files were changed. Run `safe --apply` to perform only these reversible moves.")


def is_used_in_project(name: str) -> list[str]:
    # Conservative reference search in authored code (not venv/vendor trees).
    hits: list[str] = []
    for subtree in (ROOT / "scripts", ROOT / "integrations"):
        if not subtree.is_dir():
            continue
        for p in subtree.rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".sh"):
                continue
            if p.name == name:
                continue
            try:
                if name in p.read_text(encoding="utf8"):
                    hits.append(str(p.relative_to(ROOT)))
            except (OSError, UnicodeError):
                hits.append(str(p.relative_to(ROOT)) + " (unreadable)")
    return hits


def safe_apply() -> None:
    if MANIFEST.exists():
        raise RuntimeError(f"Existing cleanup manifest at {MANIFEST}. Restore or review it first.")
    print("Creating organizational directories (no protected paths touched):")
    for name in CREATE_DIRS:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
        print("  +", name)
    moves: list[dict[str, str]] = []
    for source, target in MOVE_CANDIDATES:
        src = ROOT / source
        dst = ROOT / target
        if not src.exists():
            print(f"  [skip missing] {source}")
            continue
        if dst.exists():
            print(f"  [skip destination exists] {target}")
            continue
        hits = is_used_in_project(src.name)
        if hits:
            print(f"  [skip referenced] {source} appears in: {', '.join(hits)}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moves.append({"from": source, "to": target})
        print(f"  [moved] {source} -> {target}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "moves": moves}, indent=2) + "\n", encoding="utf8")
    print(f"\nSaved undo manifest: {MANIFEST.relative_to(ROOT)}")
    print("All active training and dataset paths untouched.")


def restore() -> None:
    if not MANIFEST.exists():
        print("No cleanup manifest found. Nothing to restore.")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    for move in reversed(manifest["moves"]):
        src = ROOT / move["to"]
        dst = ROOT / move["from"]
        if not src.exists():
            print(f"  [skip missing] {move['to']}")
            continue
        if dst.exists():
            raise RuntimeError(f"Cannot restore because destination exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  [restored] {move['to']} -> {move['from']}")
    MANIFEST.unlink()
    print("Restore completed. Created empty directories are left in place.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit", "safe", "restore"))
    parser.add_argument("--apply", action="store_true", help="Required to perform safe moves")
    args = parser.parse_args()
    if args.operation == "audit":
        print_overview()
    elif args.operation == "safe":
        if not args.apply:
            print_overview()
            print("\nTo apply: python maintenance/organize_project.py safe --apply")
        else:
            safe_apply()
    else:
        restore()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
