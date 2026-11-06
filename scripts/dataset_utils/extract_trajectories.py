import argparse
import random
import os
import h5py


def list_trajectories(h5):
    """Ritorna (root, names) dove root è 'data' o '' e names è la lista ordinata."""
    if "data" in h5 and isinstance(h5["data"], h5py.Group):
        root = "data"
        names = [k for k in h5["data"].keys() if isinstance(h5["data"][k], h5py.Group)]
    else:
        root = ""
        names = [k for k in h5.keys() if isinstance(h5[k], h5py.Group)]

    def sort_key(name):
        # ordina demo_0, demo_1, ..., demo_10 correttamente
        suffix = name.split("_")[-1]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, name)

    names.sort(key=sort_key)
    return root, names


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_dataset(src, dst_parent, name, compression, compression_opts):
    data = src[()]
    kwargs = {}
    if compression:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = compression_opts
    dst_parent.create_dataset(name, data=data, **kwargs)


def copy_group(src, dst, compression, compression_opts):
    """Copia ricorsivamente un gruppo (traiettoria)."""
    copy_attrs(src, dst)
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            sub = dst.create_group(name)
            copy_group(item, sub, compression, compression_opts)
        elif isinstance(item, h5py.Dataset):
            copy_dataset(item, dst, name, compression, compression_opts)
        else:
            print(f"[WARN] Skipping unknown object: {src.name}/{name}")


def main():
    p = argparse.ArgumentParser(description="Estrae N traiettorie da HDF5.")
    p.add_argument("--src", required=True, help="HDF5 sorgente")
    p.add_argument("--dst", required=True, help="HDF5 destinazione")
    p.add_argument("--n", type=int, default=None, help="Numero di traiettorie da estrarre")
    p.add_argument("--mode", choices=["first", "last", "random"], default="first",
                   help="Come scegliere le N traiettorie")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="Indici espliciti (ha priorità su --n/--mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compression", default="gzip",
                   choices=["gzip", "lzf", "none"], help="Compressione HDF5")
    p.add_argument("--compression_opts", type=int, default=4,
                   help="Livello compressione (per gzip, 0-9)")
    p.add_argument("--overwrite", action="store_true", help="Sovrascrive --dst se esiste")
    p.add_argument("--dry-run", action="store_true", help="Mostra la selezione senza scrivere")
    args = p.parse_args()

    if os.path.exists(args.dst) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"[ERROR] {args.dst} esiste già. Usa --overwrite o scegli un altro nome.")

    compression = None if args.compression == "none" else args.compression

    with h5py.File(args.src, "r") as fsrc:
        root, names = list_trajectories(fsrc)
        print(f"[INFO] Trovate {len(names)} traiettorie sotto '{root or '<root>'}'")

        if args.indices is not None:
            bad = [i for i in args.indices if i < 0 or i >= len(names)]
            if bad:
                raise SystemExit(f"[ERROR] Indici fuori range: {bad} (max {len(names)-1})")
            selected = [names[i] for i in args.indices]
        else:
            if args.n is None:
                raise SystemExit("[ERROR] Specifica --n oppure --indices")
            if args.n <= 0 or args.n > len(names):
                raise SystemExit(f"[ERROR] --n={args.n} non valido (disponibili: {len(names)})")
            if args.mode == "first":
                selected = names[: args.n]
            elif args.mode == "last":
                selected = names[-args.n :]
            else:
                selected = random.Random(args.seed).sample(names, args.n)

        print(f"[INFO] Selezionate {len(selected)} traiettorie")

        if args.dry_run:
            print("[INFO] Dry-run: nessun file scritto.")
            return

        with h5py.File(args.dst, "w") as fdst:
            # preserva attributi globali (es. env_args)
            copy_attrs(fsrc, fdst)

            dst_root = fdst.create_group(root) if root else fdst

            for name in selected:
                src_path = f"{root}/{name}" if root else name
                src_traj = fsrc[src_path]
                if not isinstance(src_traj, h5py.Group):
                    print(f"[WARN] {src_path} non è un gruppo, salto")
                    continue
                dst_traj = dst_root.create_group(name)
                copy_group(src_traj, dst_traj, compression, args.compression_opts)

            # aggiungi metadati di provenienza
            fdst.attrs["extracted_from"] = os.path.abspath(args.src)
            fdst.attrs["extracted_count"] = len(selected)
            fdst.attrs["extracted_selection"] = ",".join(selected)

    print(f"[INFO] Salvato in {args.dst}")


if __name__ == "__main__":
    main()