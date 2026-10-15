#!/usr/bin/env python3
"""One-way, backup-first migration of the P-Rob3 thesis data layout.

Run from anywhere: python maintenance/migrate_data_layout.py [--apply]
Default: print plan, do nothing. With --apply: move directory trees, rewrite
split HDF5 ExternalLink targets and manifest source paths, update known scripts.
No raw HDF5 or TFRecord contents are copied or rewritten.

This is an irreversible-by-command migration: backups and a journal are provided
for manual rollback, not a magic --restore that risks damaging HDF5 files.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_RLDS = ROOT / "data"
NEW_RLDS = ROOT / "rlds_data"
NEW_RAW = ROOT / "hdf5_data"
OLD_RAW = ROOT / "scripts" / "data"
PREVIOUS_RAW = OLD_RLDS / "raw" / "episodes"

# Replace ONLY known, project-local source-code path patterns.
# Preserve unrelated expressions, especially HDF5 group keys named "data".
RULES = (
    ('root / "scripts" / "data"', 'root / "hdf5_data"'),
    ('project_root / "scripts" / "data"', 'project_root / "hdf5_data"'),
    ('thesis / "scripts" / "data"', 'thesis / "hdf5_data"'),
    ('root / "data" / "raw" / "episodes"', 'root / "hdf5_data"'),
    ('project_root / "data" / "raw" / "episodes"', 'project_root / "hdf5_data"'),
    ('thesis / "data" / "raw" / "episodes"', 'thesis / "hdf5_data"'),
    ('root / "data" / "rlds"', 'root / "rlds_data" / "rlds"'),
    (' / "data" / "rlds"', ' / "rlds_data" / "rlds"'),
    (' / "data" / "splits"', ' / "rlds_data" / "splits"'),
    ('root / "data" / "rlds_builds"', 'root / "rlds_data" / "rlds_builds"'),
    ('root / "data" / "rlds_archive"', 'root / "rlds_data" / "rlds_archive"'),
    ('root / "data" / "rlds_candidate"', 'root / "rlds_data" / "rlds_candidate"'),
    ('root / "data" / "splits"', 'root / "rlds_data" / "splits"'),
    ('thesis / "data" / "splits"', 'thesis / "rlds_data" / "splits"'),
    ('THESIS_ROOT / "data" / "rlds"', 'THESIS_ROOT / "rlds_data" / "rlds"'),
    ('PROJECT_ROOT / "data" / "rlds"', 'PROJECT_ROOT / "rlds_data" / "rlds"'),
    ('THESIS_ROOT / "data" / "splits"', 'THESIS_ROOT / "rlds_data" / "splits"'),
    ('"$ROOT/data/rlds"', '"$ROOT/rlds_data/rlds"'),
    ('"$ROOT/data/splits"', '"$ROOT/rlds_data/splits"'),
    ('"$ROOT/scripts/data"', '"$ROOT/hdf5_data"'),
    ('${ROOT}/data/rlds', '${ROOT}/rlds_data/rlds'),
    ('${ROOT}/scripts/data', '${ROOT}/hdf5_data'),
    ('/data/rlds/', '/rlds_data/rlds/'),
    ('"data/rlds"', '"rlds_data/rlds"'),
    ("'data/rlds'", "'rlds_data/rlds'"),
    ('/data/rlds_builds/', '/rlds_data/rlds_builds/'),
    ('/data/rlds_archive/', '/rlds_data/rlds_archive/'),
    ('/data/splits/', '/rlds_data/splits/'),
    ('scripts/data/', 'hdf5_data/'),
    ('data/raw/episodes/', 'hdf5_data/'),
)


def choose_old_raw() -> Path | None:
    sources = [p for p in (PREVIOUS_RAW, OLD_RAW) if p.is_dir() and not p.is_symlink()]
    populated = [p for p in sources if any(p.iterdir())]
    if len(populated) > 1:
        raise RuntimeError("Due directory HDF5 popolate: " + ", ".join(map(str, populated)) +
                           ". Uniscile manualmente dopo un backup; non elimino nulla.")
    return populated[0] if populated else (sources[0] if sources else None)


def project_source_path(path_text: str) -> Path | None:
    """Locate a raw HDF5 using the preserved suffix, regardless of old OS/root."""
    raw = str(path_text).replace("\\", "/")
    for marker in ("/data/raw/episodes/", "/scripts/data/", "/hdf5_data/"):
        if marker in raw:
            suffix = raw.split(marker, 1)[1]
            if suffix and ".." not in Path(suffix).parts:
                return NEW_RAW / Path(suffix)
    return None


def get_links(view: Path):
    import h5py
    with h5py.File(view, "r") as handle:
        if "data" not in handle:
            return []
        links = []
        for name in handle["data"]:
            link = handle["data"].get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                links.append((name, link.filename, link.path))
        return links


def find_views() -> list[Path]:
    base = NEW_RLDS if NEW_RLDS.exists() else OLD_RLDS
    return sorted(base.glob("splits/**/train.hdf5")) + sorted(base.glob("splits/**/val.hdf5")) + sorted(base.glob("splits/**/test.hdf5"))


def validate_plan() -> tuple[Path | None, list[str]]:
    errors = []
    if NEW_RLDS.exists() and OLD_RLDS.exists():
        errors.append(f"Esistono entrambe {OLD_RLDS} e {NEW_RLDS}: risolvi prima il conflitto")
    source = choose_old_raw()
    if NEW_RAW.exists() and source is not None and source != NEW_RAW:
        errors.append(f"Esiste già {NEW_RAW}: non eseguo un merge automatico")
    if OLD_RAW.is_symlink():
        link_target = OLD_RAW.resolve(strict=False)
        if link_target != PREVIOUS_RAW.resolve(strict=False) and link_target != NEW_RAW.resolve(strict=False):
            errors.append(f"Symlink inatteso {OLD_RAW} -> {link_target}")
    for view in find_views():
        for name, filename, target in get_links(view):
            candidate = project_source_path(filename)
            if candidate is None:
                errors.append(f"Link sconosciuto {view}: {name} => {filename}::{target}")
            elif not candidate.exists() and not (source and
                  (source / candidate.relative_to(NEW_RAW)).exists()):
                errors.append(f"File HDF5 originale mancante per {view.name}/{name}: {candidate}")
    if not (ROOT / "scripts" / "settings.py").is_file():
        errors.append("scripts/settings.py mancante")
    return source, errors


def rewrite_manifest(obj, changed):
    if isinstance(obj, dict):
        return {k: rewrite_manifest(v, changed) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_manifest(v, changed) for v in obj]
    if isinstance(obj, str):
        candidate = project_source_path(obj)
        if candidate is not None:
            changed.append((obj, str(candidate)))
            return str(candidate)
    return obj


def relink_view(view: Path):
    import h5py
    links = get_links(view)
    if not links:
        return 0
    with h5py.File(view, "r+") as handle:
        group = handle["data"]
        for name, original, object_path in links:
            candidate = project_source_path(original)
            if candidate is None or not candidate.is_file():
                raise RuntimeError(f"Impossibile ricreare {view}/{name}: {original}")
            target = os.path.relpath(candidate, start=view.parent)
            del group[name]
            group[name] = h5py.ExternalLink(target, object_path)
    # Verify every resolved link, not merely its recorded filename.
    with h5py.File(view, "r") as handle:
        for name, _, _ in links:
            if "actions_openvla" not in handle["data"][name]:
                raise RuntimeError(f"Link non valido dopo migrazione: {view}/{name}")
    return len(links)


def patch_known_source(text: str, path: Path) -> str:
    changed = text
    for before, after in RULES:
        changed = changed.replace(before, after)
    if path == ROOT / "scripts" / "settings.py":
        # Preserve every setting except the output root. Insert ROOT in the
        # settings module itself, independent of the calling working directory.
        pattern = r'(?m)^dataset_root\s*=\s*[^\r\n#]+(?:\s*#.*)?$'
        new_line = 'dataset_root = str(Path(__file__).resolve().parents[1] / "hdf5_data")'
        changed, n = re.subn(pattern, new_line, changed, count=1)
        if n != 1:
            raise RuntimeError("Impossibile trovare una sola assegnazione dataset_root in settings.py")
        if 'from pathlib import Path' not in changed:
            changed = 'from pathlib import Path\n' + changed
    return changed


def source_candidates() -> list[Path]:
    candidates = []
    for directory in (ROOT / "scripts", ROOT / "integrations" / "openvla"):
        if directory.is_dir():
            candidates.extend(directory.glob("*.py"))
            candidates.extend(directory.glob("*.sh"))
    return sorted(set(candidates))


def write_backup(file: Path, backup_root: Path):
    backup = backup_root / file.relative_to(ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, backup)


def apply(source: Path | None) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "maintenance" / ("data_layout_backup_" + stamp)
    if backup.exists():
        raise FileExistsError(backup)
    backup.mkdir(parents=True)
    moved = []
    old_link = os.readlink(OLD_RAW) if OLD_RAW.is_symlink() else None
    journal = {"backup_root": str(backup), "moves": moved, "old_link": old_link,
               "warning": "Keep the backup and do not delete source HDF5 files."}
    try:
        # Capture tiny view/manifest files BEFORE moving any directories.
        views_before = find_views()
        for file in views_before:
            write_backup(file, backup)
        manifest_before = list((NEW_RLDS if NEW_RLDS.exists() else OLD_RLDS).glob("splits/**/manifest.json"))
        for file in manifest_before:
            write_backup(file, backup)
        for file in source_candidates():
            write_backup(file, backup)
        if (ROOT / ".gitignore").is_file():
            write_backup(ROOT / ".gitignore", backup)

        if source is not None:
            source.rename(NEW_RAW)
            moved.append([str(source), str(NEW_RAW)])
            print(f"MOVE  {source} -> {NEW_RAW}")
        else:
            NEW_RAW.mkdir(exist_ok=True)
        if OLD_RAW.is_symlink():
            OLD_RAW.unlink()
            print(f"REMOVE symlink {OLD_RAW} ({old_link})")
        if OLD_RLDS.exists():
            OLD_RLDS.rename(NEW_RLDS)
            moved.append([str(OLD_RLDS), str(NEW_RLDS)])
            print(f"MOVE  {OLD_RLDS} -> {NEW_RLDS}")
        else:
            NEW_RLDS.mkdir(exist_ok=True)

        # Preserve split IDs, episodes, folds and experiment metadata.
        for manifest_file in sorted(NEW_RLDS.glob("splits/**/manifest.json")):
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            modified = []
            converted = rewrite_manifest(manifest, modified)
            if modified:
                manifest_file.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"MANIFEST {manifest_file}: {len(modified)} source path aggiornati")

        total = 0
        for view in find_views():
            count = relink_view(view)
            total += count
            print(f"RELINK {view}: {count} ExternalLink verificati")
        print(f"Totale ExternalLink verificati: {total}")

        for file in source_candidates():
            old = file.read_text(encoding="utf-8")
            new = patch_known_source(old, file)
            if new != old:
                file.write_text(new, encoding="utf-8")
                print(f"PATCH  {file.relative_to(ROOT)}")

        # Prevent future HDF5/TFRecord files from being added to Git.
        gitignore = ROOT / ".gitignore"
        ignored = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
        for pattern in ("/hdf5_data/", "/rlds_data/"):
            if pattern not in ignored.splitlines():
                ignored += ("\n" if ignored and not ignored.endswith("\n") else "") + pattern + "\n"
        gitignore.write_text(ignored, encoding="utf-8")

        # Old data/raw/ is now empty if it hosted the old raw HDF5s.
        unused_raw = NEW_RLDS / "raw"
        if unused_raw.is_dir() and not any(unused_raw.iterdir()):
            unused_raw.rmdir()

        (backup / "migration_journal.json").write_text(
            json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nMIGRAZIONE COMPLETATA. Backup:", backup)
        print("Non eliminare i backup finché non avrai testato gli split e il preflight.")
    except Exception:
        (backup / "migration_journal.json").write_text(
            json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nATTENZIONE: migrazione interrotta. NON rilanciare automaticamente.")
        print("Controlla il journal e i backup in:", backup)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Applica i cambiamenti; default: solo audit")
    args = parser.parse_args()
    source, errors = validate_plan()
    print("ROOT:        ", ROOT)
    print("OLD HDF5:    ", source or "nessuna directory da spostare")
    print("NEW HDF5:    ", NEW_RAW)
    print("OLD RLDS:    ", OLD_RLDS)
    print("NEW RLDS:    ", NEW_RLDS)
    print("Symlink:     ", OLD_RAW.is_symlink())
    print("Split views: ", len(find_views()))
    print("Backup codice:\n", "\n ".join(str(p.relative_to(ROOT)) for p in source_candidates()))
    if errors:
        for error in errors:
            print("BLOCCO:", error)
        raise SystemExit("Risolvi i problemi elencati: nessun file è stato modificato.")
    if args.apply:
        apply(source)
    else:
        print("\nAUDIT OK: nessun file modificato. Riesegui con --apply per migrare.")


if __name__ == "__main__":
    main()
