"""P-Rob3-specific OpenVLA data policy.

- Statistics are calculated from train only.
- Normalization cache is automatically versioned.
- Unified mixture statistics ensure consistent action tokenization across datasets.
"""

import hashlib
from pathlib import Path
from typing import List, Optional
import numpy as np


def dataset_fingerprint(dataset_dir) -> str:
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


def _merge_statistics(stats_list: List[dict], weights: List[float]) -> dict:
    """Merge per-dataset statistics into a single unified statistics dictionary."""
    if len(stats_list) == 1:
        return stats_list[0]

    norm_w = [float(w) / sum(weights) for w in weights]
    merged = {
        "num_transitions": sum(int(s["num_transitions"]) for s in stats_list),
        "num_trajectories": sum(int(s["num_trajectories"]) for s in stats_list),
    }

    for key in ("action", "proprio"):
        if key not in stats_list[0]:
            continue
        means = np.stack([np.asarray(s[key]["mean"], dtype=np.float64) for s in stats_list])
        stds = np.stack([np.asarray(s[key]["std"], dtype=np.float64) for s in stats_list])
        mins = np.stack([np.asarray(s[key]["min"], dtype=np.float64) for s in stats_list])
        maxs = np.stack([np.asarray(s[key]["max"], dtype=np.float64) for s in stats_list])
        q01s = np.stack([np.asarray(s[key]["q01"], dtype=np.float64) for s in stats_list])
        q99s = np.stack([np.asarray(s[key]["q99"], dtype=np.float64) for s in stats_list])

        w_arr = np.asarray(norm_w, dtype=np.float64)[:, None]
        w_mean = np.sum(w_arr * means, axis=0)
        # Combined variance across mixture components
        w_var = np.sum(w_arr * (stds ** 2 + (means - w_mean) ** 2), axis=0)

        merged[key] = {
            "mean": w_mean.tolist(),
            "std": np.sqrt(w_var).tolist(),
            "min": np.min(mins, axis=0).tolist(),
            "max": np.max(maxs, axis=0).tolist(),
            "q01": np.sum(w_arr * q01s, axis=0).tolist(),
            "q99": np.sum(w_arr * q99s, axis=0).tolist(),
        }
        if "mask" in stats_list[0][key]:
            merged[key]["mask"] = stats_list[0][key]["mask"]

    return merged


def install_prob3_train_only_statistics(
    num_episodes: Optional[int] = None,
    data_root_dirs: Optional[List[str]] = None,
    dataset_name: str = "prob3_pick_place",
    dataset_weights: Optional[List[float]] = None,
):
    import dlimp as dl
    from prismatic.vla.datasets.rlds import dataset as rlds_module

    if getattr(dl.DLataset, "_prob3_policy_installed", False):
        return

    original_from_rlds = dl.DLataset.from_rlds
    original_get_stats = rlds_module.get_dataset_statistics

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

    unified_stats_holder = {"stats": None}

    def patched_get_stats(dataset, hash_dependencies, save_dir=None):
        if save_dir is None:
            raise RuntimeError(
                "P-Rob3 requires a dataset directory for normalization cache validation."
            )
        if unified_stats_holder["stats"] is not None:
            return unified_stats_holder["stats"]

        signature = dataset_fingerprint(save_dir)
        dependencies = tuple(hash_dependencies) + (
            "prob3_train_only_statistics_v2",
            signature,
        )
        return original_get_stats(
            dataset, hash_dependencies=dependencies, save_dir=save_dir
        )

    rlds_module.get_dataset_statistics = patched_get_stats
    dl.DLataset._prob3_policy_installed = True

    # Pre-compute unified statistics if mixing multiple datasets
    if data_root_dirs and len(data_root_dirs) > 1:
        from integrations.openvla.register import register_prob3
        from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights

        register_prob3()
        weights = dataset_weights or [1.0] * len(data_root_dirs)
        per_source_stats = []
        for root_dir in data_root_dirs:
            ds_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
                Path(root_dir), [dataset_name], load_camera_views=("primary",), load_proprio=False
            )
            # Build raw dlimp train dataset to compute/load its cached train-only stats
            raw_ds = dl.DLataset.from_rlds(
                ds_kwargs[0]["data_dir"], split="train", shuffle=False
            )
            traj_ds = raw_ds.traj_map(ds_kwargs[0]["standardize_fn"])
            stats_dir = Path(ds_kwargs[0]["data_dir"])
            s = patched_get_stats(
                traj_ds,
                hash_dependencies=(dataset_name,),
                save_dir=stats_dir,
            )
            per_source_stats.append(s)

        unified_stats_holder["stats"] = _merge_statistics(per_source_stats, weights)
        print(f"[P-Rob3] Unified mixture statistics active across {len(data_root_dirs)} sources.")

    print("[P-Rob3] Train-only normalization enabled.")