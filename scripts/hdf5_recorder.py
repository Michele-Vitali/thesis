from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import settings
from scipy.spatial.transform import Rotation as R


class HDF5Recorder:
    """
    Recorder for P-Rob3 simulated demonstrations.

    Each successful episode is stored inside one HDF5 file using a
    LIBERO / OpenVLA-like structure:

        data/
            demo_0/
                obs/
                    agentview_rgb
                    joint_states
                    gripper_states
                    ee_states
                    ee_pos
                    ee_ori

                actions
                actions_openvla
                states
                robot_states
                rewards
                dones

    `actions` contains the exact 7D OSC action sent to robosuite.

    `actions_openvla` contains the same first 6 dimensions, while
    converting the gripper to:

        0 = closed
        1 = open

    Failed demonstrations are never written to disk.
    """

    def __init__(self):
        # The recorder is intentionally LAZY: no directory or HDF5 file
        # is created until the first SUCCESSFUL episode is saved.
        self.root_directory = Path(settings.dataset_root).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_name = f"{settings.dataset_name}_{timestamp}"
        self.root = self.root_directory / self.run_name
        self.hdf5_path = self.root / "demo.hdf5"

        self.file = None
        self.data_group = None
        self.saved_episodes = 0
        self.total_saved_frames = 0
        self._clear_episode_buffer()

    def _open_on_first_save(self):
        """Initialize on-disk storage only when there is data worth saving."""
        if self.file is not None:
            return

        # Do not silently overwrite another run (e.g. process restarted).
        self.root.mkdir(parents=True, exist_ok=False)
        self.file = h5py.File(self.hdf5_path, "x")
        self.data_group = self.file.create_group("data")
        self.data_group.attrs["robot"] = settings.robot
        self.data_group.attrs["env"] = settings.env_name
        self.data_group.attrs["fps"] = settings.dataset_fps
        self.data_group.attrs["task"] = settings.dataset_task
        self.data_group.attrs["camera"] = settings.dataset_camera_name
        print(f"HDF5 dataset: {self.hdf5_path}")

    def _clear_episode_buffer(self):
        """
        Reset all temporary buffers for the current episode.
        """

        self.images = []

        self.joint_states = []
        self.gripper_states = []

        self.ee_states = []
        self.ee_positions = []
        self.ee_orientations = []

        self.actions = []
        self.actions_openvla = []

        self.sim_states = []
        self.robot_states = []

        # P-Grip starts open.
        #
        # OpenVLA convention used here:
        #   0 = closed
        #   1 = open
        self.gripper_target = 1.0

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert robosuite image to HWC uint8.
        """

        image = np.asarray(image)

        # CHW -> HWC if necessary.
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            image = image.astype(np.float32)

            if np.max(image) <= 1.0:
                image *= 255.0

            image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        expected_shape = (settings.dataset_image_height, settings.dataset_image_width, 3)

        if image.shape != expected_shape:
            raise ValueError(f"Unexpected image shape: {image.shape}, expected {expected_shape}.")

        return np.ascontiguousarray(image)

    def _get_gripper_state(self, observation: dict) -> np.ndarray:
        """
        Return physical P-Grip joint positions.

        Usually robosuite exposes both physical finger qpos values.
        """

        gripper_qpos = observation.get("robot0_gripper_qpos", None)

        if gripper_qpos is None:
            return np.zeros(2, dtype=np.float32)

        return np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)

    def _convert_action_for_openvla(self, action: np.ndarray) -> np.ndarray:

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}.")

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains invalid values!")

        if not 0.0 <= action[-1] <= 1.0:
            raise ValueError(f"Invalid gripper target: {action[-1]}")

        return action.copy()

    def record_frame(self, env, action: np.ndarray, observation: dict = None):
        """
        Record observation_t together with action_t.

        This function is called BEFORE env.step(action), which is exactly
        what we want for imitation-learning data:

            observation_t -> action_t
        """

        if observation is None:
            observation = env._get_observations()

        camera_key = settings.dataset_camera_observation_key

        if camera_key not in observation:
            raise KeyError(f"Camera observation '{camera_key}' not found.")

        image = self._prepare_image(observation[camera_key])

        joint_state = np.asarray(env.robots[0]._joint_positions, dtype=np.float32).reshape(-1)[:6]

        gripper_state = self._get_gripper_state(observation)

        eef_pos = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3)

        eef_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(4)

        # Robosuite quaternion:
        # scipy expects [x, y, z, w],
        # which matches robot0_eef_quat.
        eef_axis_angle = R.from_quat(eef_quat).as_rotvec().astype(np.float32)

        eef_state = np.concatenate([eef_pos, eef_axis_angle]).astype(np.float32)

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}.")

        action_openvla = self._convert_action_for_openvla(action)

        sim_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float32)

        # LIBERO-style compact robot state.
        #
        # [gripper physical qpos, eef xyz, eef quaternion]
        robot_state = np.concatenate([gripper_state, eef_pos, eef_quat]).astype(np.float32)

        self.images.append(image)

        self.joint_states.append(joint_state)

        self.gripper_states.append(gripper_state)

        self.ee_states.append(eef_state)

        self.ee_positions.append(eef_pos)

        self.ee_orientations.append(eef_axis_angle)

        self.actions.append(action)

        self.actions_openvla.append(action_openvla)

        self.sim_states.append(sim_state)

        self.robot_states.append(robot_state)

    def save_episode(self):
        """
        Write the buffered successful demonstration to HDF5.
        """

        num_frames = len(self.actions)

        if num_frames == 0:
            print("Episode contains no frames, nothing will be saved.")
            return

        self._open_on_first_save()

        episode_name = f"demo_{self.saved_episodes}"

        episode_group = self.data_group.create_group(episode_name)

        episode_group.attrs["num_samples"] = num_frames

        episode_group.attrs["language_instruction"] = settings.dataset_task

        obs_group = episode_group.create_group("obs")

        obs_group.create_dataset("agentview_rgb", data=np.stack(self.images, axis=0), dtype=np.uint8, compression="gzip", compression_opts=4)

        obs_group.create_dataset("joint_states", data=np.stack(self.joint_states, axis=0), dtype=np.float32)

        obs_group.create_dataset("gripper_states", data=np.stack(self.gripper_states, axis=0), dtype=np.float32)

        obs_group.create_dataset("ee_states", data=np.stack(self.ee_states, axis=0), dtype=np.float32)

        obs_group.create_dataset("ee_pos", data=np.stack(self.ee_positions, axis=0), dtype=np.float32)

        obs_group.create_dataset("ee_ori", data=np.stack(self.ee_orientations, axis=0), dtype=np.float32)

        episode_group.create_dataset("actions", data=np.stack(self.actions, axis=0), dtype=np.float32)

        episode_group.create_dataset("actions_openvla", data=np.stack(self.actions_openvla, axis=0), dtype=np.float32)

        episode_group.create_dataset("states", data=np.stack(self.sim_states, axis=0), dtype=np.float32)

        episode_group.create_dataset("robot_states", data=np.stack(self.robot_states, axis=0), dtype=np.float32)

        rewards = np.zeros(num_frames, dtype=np.uint8)

        dones = np.zeros(num_frames, dtype=np.uint8)

        # Only successful episodes reach save_episode().
        rewards[-1] = 1
        dones[-1] = 1

        episode_group.create_dataset("rewards", data=rewards)

        episode_group.create_dataset("dones", data=dones)

        self.file.flush()

        self.saved_episodes += 1
        self.total_saved_frames += num_frames

        print(f"Episode saved: {episode_name} ({num_frames} frames)")

        self._clear_episode_buffer()

    def discard_episode(self):
        """
        Delete all buffered frames for an unsuccessful demonstration.
        """

        num_frames = len(self.actions)

        if num_frames > 0:
            print(f"Discarding failed episode ({num_frames} frames).")

        self._clear_episode_buffer()

    def finalize(self):
        """
        Flush and close the HDF5 file.
        """

        if self.file is None:
            # No successful episode was saved: nothing was ever created.
            self._clear_episode_buffer()
            return

        self.file.flush()
        self.file.close()

        print("=" * 60)
        print("HDF5 dataset finalized.")
        print(f"Saved episodes: {self.saved_episodes}")
        print(f"Saved frames: {self.total_saved_frames}")
        print(f"Path: {self.hdf5_path}")
        print("=" * 60)

        self.file = None

class NullRecorder:
    """No-op recorder used when settings.record is False.

    This object has the same public methods used by main.py, but it never
    touches the filesystem. It also avoids conditional branches in task loops.
    """

    def __init__(self):
        self.saved_episodes = 0
        self.total_saved_frames = 0
        self.hdf5_path = None
        self.root = None

    def record_frame(self, *args, **kwargs):
        pass

    def save_episode(self):
        pass

    def discard_episode(self):
        pass

    def finalize(self):
        pass
