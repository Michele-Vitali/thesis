#!/usr/bin/env python3
"""Read-only listing of project-owned files containing path assumptions.

Usage: python maintenance/audit_path_references.py
Does not edit files and avoids scanning vendor repos or virtualenvs.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
search = ("scripts", "integrations", "maintenance", "rlds_dataset_builder")
tokens = (
    "scripts/data", "\"data\"", "'data'", "data/rlds", "data/splits",
    "data/raw", "rlds_archive", "rlds_builds", "tools/openvla",
    "tools/rlds_dataset_builder", "runs/openvla", "adapters/openvla",
    "cache/huggingface", "PGripper.xml", "PRob3.xml", "../assets",
    "__file__", "dataset_root", "PROB3_HDF5_ROOT", "RLDS_DATA_ROOT",
)

for section in search:
    folder = root / section
    if not folder.exists():
        continue
    for file in sorted(folder.rglob("*")):
        if not file.is_file() or file.suffix not in (".py", ".sh"):
            continue
        if "__pycache__" in file.parts or "rlds_env" in file.parts:
            continue
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        hits = [(i, line.strip()) for i, line in enumerate(lines, 1)
                if any(token in line for token in tokens)]
        if hits:
            print(f"\n{file.relative_to(root)}")
            for number, line in hits[:40]:
                print(f"  {number:4d}: {line[:160]}")
            if len(hits) > 40:
                print(f"  ... {len(hits)-40} more matches")
