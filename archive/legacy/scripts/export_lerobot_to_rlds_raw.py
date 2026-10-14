import argparse
from pathlib import Path

import numpy as np
import settings
from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


# Robosuite OSC output ranges currently used by the project.
OSC_TRANSLATION_SCALE = np.array(settings.controller_max_abs_pos_output, dtype=np.float32)

OSC_ROTATION_SCALE = np.array(settings.controller_max_abs_rot_output, dtype=np.float32)


def _to_numpy(value):
    """
    Convert torch tensors / numpy arrays / scalars to numpy.
    """

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def _scalar_int(value) -> int:
    """
    Convert a scalar tensor / numpy value to int.
    """

    array = _to_numpy(value).reshape(-1)

    if array.size == 0:
        raise ValueError("Cannot convert an empty value to int.")

    return int(array[0])


def _prepare_image(image) -> np.ndarray:
    """
    Convert a LeRobot image to HWC uint8.

    LeRobot generally exposes images as CHW torch tensors.
    This function is deliberately tolerant to both CHW and HWC.
    """

    image = _to_numpy(image)

    # CHW -> HWC (Channel-Height-Width -> Height-Width-Channel)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Drop alpha if ever present.
    if image.shape[-1] == 4:
        # This means take all dimensions from the preceding array values, for the last only the first 3
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        # LeRobot frequently returns images in [0, 1].
        if np.max(image) <= 1.0 + 1e-6:
            image *= 255.0

        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(image)


def convert_action(raw_action, current_gripper_target: float):
    """
    Convert the project's 7D robosuite command to the action
    representation used for OpenVLA training.

    Input:
        raw_action =
            [
                normalized_dx,
                normalized_dy,
                normalized_dz,
                normalized_drx,
                normalized_dry,
                normalized_drz,
                gripper_command,
            ]

        gripper command semantics:
            -1 = close
             0 = hold
            +1 = open

    Output:
        [
            delta_x_m,
            delta_y_m,
            delta_z_m,
            delta_rot_x_rad,
            delta_rot_y_rad,
            delta_rot_z_rad,
            absolute_gripper,
        ]

    where:
        absolute_gripper:
            0 = closed
            1 = open
    """

    # Flatten the action's array
    raw_action = _to_numpy(raw_action).astype(np.float32).reshape(-1)

    if raw_action.shape != (7,):
        raise ValueError(f"Expected a 7D action, got {raw_action.shape}.")

    # Initialize an empty array
    result = np.zeros(7, dtype=np.float32)

    # Convert the raw actions in the controller's scale
    result[0:3] = raw_action[0:3] * OSC_TRANSLATION_SCALE

    result[3:6] = raw_action[3:6] * OSC_ROTATION_SCALE

    # Fix the difference in gripper's action standard between the project and OpenVLA...
    gripper_command = float(raw_action[6])

    if gripper_command > 1e-6:
        current_gripper_target = 1.0

    elif gripper_command < -1e-6:
        current_gripper_target = 0.0

    # If the command is exactly 0.0, then we pass the current target!

    result[6] = float(current_gripper_target)

    return (result, current_gripper_target)


def save_episode(episode_steps, output_dir: Path, episode_index: int):
    """
    Save one temporary raw RLDS episode.

    The resulting .npy files are consumed by the TFDS builder.
    """

    if not episode_steps:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_path = output_dir / f"episode_{episode_index:06d}.npy"

    np.save(episode_path, np.asarray(episode_steps, dtype=object), allow_pickle=True)

    print(f"Saved RLDS raw episode {episode_index}: {len(episode_steps)} steps -> {episode_path}")


def export_dataset(lerobot_root: Path, output_dir: Path, instruction: str, repo_id: str):
    """
    Export a local LeRobot dataset into temporary episode .npy files
    suitable for the P-Rob3 TFDS / RLDS builder.
    """

    dataset = LeRobotDataset(repo_id=repo_id, root=lerobot_root, return_uint8=True)

    print(f"Loaded LeRobot dataset:\n  root: {lerobot_root}\n  episodes: {dataset.num_episodes}\n  frames: {dataset.num_frames}")

    current_episode_index = None
    current_episode_steps = []

    # P-Grip always starts open at episode reset.
    current_gripper_target = 1.0

    for dataset_index in range(len(dataset)):

        item = dataset[dataset_index]

        episode_index = _scalar_int(item["episode_index"])

        if current_episode_index is None:
            current_episode_index = episode_index
            current_gripper_target = 1.0
        elif episode_index != current_episode_index:
            save_episode(current_episode_steps, output_dir, current_episode_index)
            current_episode_steps = []
            current_episode_index = episode_index
            current_gripper_target = 1.0

        image = _prepare_image(item[IMAGE_KEY])

        state = _to_numpy(item[STATE_KEY]).astype(np.float32).reshape(-1)

        if state.shape != (7,):
            raise ValueError(f"Expected 7D state, got {state.shape}.")

        action, current_gripper_target = convert_action(item[ACTION_KEY], current_gripper_target)

        step = {
            "image": image,
            "state": state,
            "action": action,
            "language_instruction": instruction,
        }

        current_episode_steps.append(step)

    # Save final episode.
    if current_episode_index is not None:

        save_episode(current_episode_steps, output_dir, current_episode_index)

    print("\nConversion completed.")


def main():
    parser = argparse.ArgumentParser(description=("Convert a local LeRobot P-Rob3 dataset to temporary raw RLDS episodes."))

    parser.add_argument("--root", required=True, type=Path, help=("Path to the LeRobot dataset directory."))

    parser.add_argument("--output", required=True, type=Path, help=("Destination folder for episode_XXXXXX.npy files."))

    parser.add_argument("--instruction", default=("Pick up the cube and place it at the center of the table"), help="Language instruction.")

    parser.add_argument("--repo-id", default=None, help=("LeRobot repo id. Defaults to local/<dataset-folder-name>."))

    args = parser.parse_args()

    repo_id = args.repo_id if args.repo_id is not None else f"local/{args.root.name}"

    export_dataset(lerobot_root=args.root, output_dir=args.output, instruction=args.instruction, repo_id=repo_id)

if __name__ == "__main__":
    main()