#!/usr/bin/env python3
"""Validate the pilot HDF5 dataset before OpenVLA RLDS export.

Usage from thesis root (venv):
 python scripts/audit_dataset_v2.py --source data/raw/episodes/RUN_NAME/demo.hdf5
"""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

PHASES = ("approach", "descend", "grasp", "lift", "transport", "align", "lower", "release")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--image-dir", default="runs/v2_pilot_rgb", type=Path)
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"HDF5 not found: {args.source}")
    args.image_dir.mkdir(parents=True, exist_ok=True)

    all_problems = []
    all_frames = 0
    count_by_phase = Counter()
    with h5py.File(args.source, "r") as f:
        if "data" not in f:
            raise SystemExit("ERROR: missing data/ group")
        names = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[-1]))
        print(f"Source: {args.source.resolve()}")
        print(f"Episodes: {len(names)}")
        for i, name in enumerate(names):
            ep = f["data"][name]
            problems = []
            required = (
                "actions", "actions_openvla", "phase_id", "timestamp",
                "obs/agentview_rgb", "obs/previous_action", "gripper_command"
            )
            missing = [key for key in required if key not in ep]
            if missing:
                all_problems.append(f"{name}: missing fields {missing}")
                continue
            actions = np.asarray(ep["actions"][:], dtype=np.float32)
            labels = np.asarray(ep["actions_openvla"][:], dtype=np.float32)
            phases = np.asarray(ep["phase_id"][:], dtype=np.uint8)
            ts = np.asarray(ep["timestamp"][:], dtype=np.float64)
            previous = np.asarray(ep["obs/previous_action"][:], dtype=np.float32)
            grip = np.asarray(ep["gripper_command"][:], dtype=np.float32).reshape(-1)
            images = ep["obs/agentview_rgb"]
            n = actions.shape[0]
            all_frames += n
            if actions.shape != (n, 7) or labels.shape != (n, 7):
                problems.append("actions/labels are not [T,7]")
            if previous.shape != (n, 7):
                problems.append("previous_action shape mismatch")
            if images.shape != (n, 256, 256, 3) or images.dtype != np.uint8:
                problems.append(f"RGB shape/dtype: {images.shape}, {images.dtype}")
            if len(phases) != n or len(ts) != n or len(grip) != n:
                problems.append("metadata lengths don't match action count")
            if problems:
                all_problems.append(f"{name}: {problems}")
                continue
            if not np.isfinite(actions).all() or not np.isfinite(labels).all():
                problems.append("nonfinite actions")
            if not np.isin(np.round(actions[:, 6], 4), [0., 1.]).all():
                problems.append("raw actions contain gripper values other than 0/1")
            if not np.allclose(labels, actions, rtol=0, atol=1e-5):
                problems.append("actions_openvla differs from actions (new absolute convention!)")
            if not np.allclose(grip, actions[:, 6], rtol=0, atol=1e-5):
                problems.append("gripper_command differs from action[6]")
            if not np.allclose(previous[0], np.zeros(7), atol=1e-5):
                problems.append("first previous_action should be zeros")
            if n > 1 and not np.allclose(previous[1:], actions[:-1], rtol=0, atol=1e-5):
                problems.append("previous_action not aligned to previous recorded action")
            if n > 1 and not np.all(np.diff(ts) > 0):
                problems.append("timestamps not strictly increasing")
            if np.any(phases >= len(PHASES)):
                problems.append("unknown phase_id")
            # For full successful expert demonstrations, both open and close should occur.
            grip_values = set(np.round(actions[:, 6], 4).tolist())
            if grip_values != {0.0, 1.0}:
                problems.append(f"gripper didn't show both states: {grip_values}")
            transitions = int(np.count_nonzero(np.diff(np.rint(actions[:, 6]))))
            if transitions != 2:
                problems.append(f"expected open->closed->open (2 transitions), got {transitions}")
            phases_present = sorted(set(phases.tolist()))
            for pid in phases.tolist():
                if 0 <= pid < len(PHASES):
                    count_by_phase[PHASES[pid]] += 1
            if problems:
                all_problems.append(f"{name}: {'; '.join(problems)}")
            print(f"{name}: {n} frames, phases={[PHASES[j] for j in phases_present if j < len(PHASES)]}, "
                  f"grip transitions={transitions}; {'WARN ' + '; '.join(problems) if problems else 'OK'}")

            # Save exact recorded RGB samples, one from each available phase.
            if i < 3:
                for pid in phases_present:
                    if pid >= len(PHASES):
                        continue
                    j = int(np.flatnonzero(phases == pid)[0])
                    arr = np.asarray(images[j])
                    Image.fromarray(arr).save(args.image_dir / f"{name}_{PHASES[pid]}_{j:04d}.png")

    print(f"\nTotal frames: {all_frames}")
    print("Frames per phase:", dict(count_by_phase))
    print("RGB samples:", args.image_dir)
    if all_problems:
        print("\n!!! AUDIT FAILED !!!")
        for problem in all_problems:
            print(" -", problem)
        raise SystemExit(2)
    print("\nAUDIT PASSED: labels/metadata/structure consistent with 0 CLOSED, 1 OPEN.")
    print("IMPORTANT: inspect RGB samples and physically verify cube release in simulator.")


if __name__ == "__main__":
    main()
