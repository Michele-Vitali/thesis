"""P-Rob3-specific OpenVLA data policy.

- Statistics are calculated from train only.
- Normalization cache is automatically versioned.
- Original OpenVLA files remain untouched.
"""

import hashlib
from pathlib import Path
from typing import Optional


def dataset_fingerprint(dataset_dir):
    """Identify the published TFDS dataset automatically."""
    root = Path(dataset_dir).resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)

    digest = hashlib.sha256()

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("dataset_statistics")
    )

    if not files:
        raise RuntimeError(f"No dataset files found in {root}")

    for path in files:
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()


def install_prob3_train_only_statistics(num_episodes: Optional[int] = None):
    import dlimp as dl
    from prismatic.vla.datasets.rlds import dataset as rlds_module

    if getattr(dl.DLataset, "_prob3_policy_installed", False):
        return

    original_from_rlds = dl.DLataset.from_rlds
    original_get_stats = rlds_module.get_dataset_statistics

    # Hook robusto: intercetta esplicitamente `split="all"` usato da OpenVLA
    def patched_from_rlds(*args, **kwargs):
        if "split" in kwargs:
            if kwargs["split"] == "all" or kwargs["split"] is None:
                kwargs["split"] = "train"
        elif len(args) >= 2:
            if args[1] == "all" or args[1] is None:
                args = list(args)
                args[1] = "train"
                args = tuple(args)
        else:
            kwargs["split"] = "train"

        return original_from_rlds(*args, **kwargs)

    dl.DLataset.from_rlds = patched_from_rlds

    def patched_get_stats(dataset, hash_dependencies, save_dir=None):
        if save_dir is None:
            raise RuntimeError(
                "P-Rob3 requires a dataset directory for normalization cache validation."
            )

        signature = dataset_fingerprint(save_dir)
        
        # AGGIUNTO "_v2" AL NOME DELLA DIPENDENZA: 
        # Questo forza OpenVLA a cambiare l'hash finale e ricalcolare 
        # le statistiche da zero ignorando il vecchio .json buggato!
        dependencies = tuple(hash_dependencies) + (
            "prob3_train_only_statistics_v2",
            signature,
        )

        return original_get_stats(
            dataset, hash_dependencies=dependencies, save_dir=save_dir
        )

    rlds_module.get_dataset_statistics = patched_get_stats

    dl.DLataset._prob3_policy_installed = True

    print("[P-Rob3] Train-only normalization enabled.")
    print("[P-Rob3] Automatic dataset cache enabled.")