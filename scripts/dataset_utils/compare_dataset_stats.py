"""
Compare the action-normalization statistics (q01 / q99) of two RLDS datasets.

Why: finetune.py normalizes every dataset of the mixture with ITS OWN statistics, but the checkpoint saves (and the
evaluation server uses) only the statistics of the FIRST dataset. If the two datasets have different q01/q99, the same
physical action is turned into different tokens depending on the source, and decoding is biased for the second one.

Usage:
  python compare_stats.py --a <thesis>/rlds_data/rlds_base --b <thesis>/rlds_data/rlds_lights

The statistics files (dataset_statistics_<hash>.json) are created by OpenVLA the first time a dataset is used for
training, inside the dataset's TFDS directory. If a dataset has none yet, start (and stop) a smoke run on it first.
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


def newest_stats(root: Path) -> Path:
    files = sorted(root.rglob("dataset_statistics_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No dataset_statistics_*.json below {root} "
                                "(run a short training smoke test on this dataset to create it)")
    return files[-1]


def load_action_stats(path: Path) -> dict:
    st = json.load(open(path))
    if "action" not in st:                      # keyed by dataset name
        st = next(iter(st.values()))
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="RLDS root of the first dataset (the one whose stats get saved)")
    ap.add_argument("--b", required=True, help="RLDS root of the second dataset")
    ap.add_argument("--warn_pct", type=float, default=2.0, help="warn if a quantile moves by more than this %% of the range")
    args = ap.parse_args()

    pa, pb = newest_stats(Path(args.a)), newest_stats(Path(args.b))
    sa, sb = load_action_stats(pa), load_action_stats(pb)
    print(f"A: {pa}\n   transitions: {sa.get('num_transitions')}, trajectories: {sa.get('num_trajectories')}")
    print(f"B: {pb}\n   transitions: {sb.get('num_transitions')}, trajectories: {sb.get('num_trajectories')}\n")

    qa01, qa99 = np.asarray(sa["action"]["q01"], float), np.asarray(sa["action"]["q99"], float)
    qb01, qb99 = np.asarray(sb["action"]["q01"], float), np.asarray(sb["action"]["q99"], float)
    rng = np.maximum(qa99 - qa01, 1e-9)

    print(f"{'dim':5} {'q01 A':>9} {'q01 B':>9} {'q99 A':>9} {'q99 B':>9} {'max shift (% of range A)':>26}")
    worst = 0.0
    for i, n in enumerate(NAMES):
        shift = max(abs(qb01[i] - qa01[i]), abs(qb99[i] - qa99[i])) / rng[i] * 100.0
        worst = max(worst, shift)
        flag = "  <-- differs" if shift > args.warn_pct else ""
        print(f"{n:5} {qa01[i]:9.4f} {qb01[i]:9.4f} {qa99[i]:9.4f} {qb99[i]:9.4f} {shift:26.2f}{flag}")

    print()
    if worst <= args.warn_pct:
        print(f"Largest shift {worst:.2f}% of the range: the two datasets are normalized almost identically,"
              " mixing them with saved statistics of dataset A is a negligible approximation.")
    else:
        print(f"Largest shift {worst:.2f}% of the range: the datasets are NOT normalized the same way."
              " Use shared (union) statistics before mixing, or merge them into one RLDS dataset.")