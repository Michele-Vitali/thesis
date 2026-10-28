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

    def patched_from_rlds(builder, *args, **kwargs):
        if builder.info.name == "prob3_pick_place":
            # If num episodes is specificied we load only *those* number of episodes, otherwise all the training split.
            if num_episodes is not None and num_episodes > 0:
                kwargs["split"] = f"train[:{num_episodes}]"
                print(
                    f"[P-Rob3 Policy] Caricamento limitato a: train[:{num_episodes}] episodi."
                )
            else:
                kwargs["split"] = "train"
                print("[P-Rob3 Policy] Caricamento completo: split 'train'.")

        return original_from_rlds(builder, *args, **kwargs)

    def patched_get_stats(dataset, hash_dependencies, save_dir=None):
        if save_dir is None:
            raise RuntimeError(
                "P-Rob3 requires a dataset directory for normalization cache validation."
            )

        signature = dataset_fingerprint(save_dir)
        dependencies = tuple(hash_dependencies) + (
            "prob3_train_only_statistics",
            signature,
        )

        return original_get_stats(
            dataset, hash_dependencies=dependencies, save_dir=save_dir
        )

    dl.DLataset.from_rlds = staticmethod(patched_from_rlds)
    rlds_module.get_dataset_statistics = patched_get_stats

    dl.DLataset._prob3_policy_installed = True

    print("[P-Rob3] Train-only normalization enabled.")
    print("[P-Rob3] Automatic dataset cache enabled.")