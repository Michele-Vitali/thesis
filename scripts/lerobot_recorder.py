from datetime import datetime
from pathlib import Path

import numpy as np
import settings
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotRecorder:
    """
    Records P-Rob3 demonstrations directly in LeRobotDataset format.

    Each successfully completed pick-and-place execution is stored as
    one LeRobot episode.

    Failed episodes can be discarded with discard_episode().
    """

    def __init__(self):
        # Grab the scripts directory
        script_directory = Path(__file__).resolve().parent

        # Define the data directory
        data_directory = script_directory / settings.dataset_root

        # Create the directory, if intermediary folders are not present we create them (parents = True), 
        # if the folder already exists we do not launch any error (exist_ok = True)
        data_directory.mkdir(parents=True, exist_ok=True)

        # Grab the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define the dataset name
        self.run_name = (f"{settings.dataset_name}_{timestamp}")

        # Define the dataset directory
        self.root = data_directory / self.run_name

        # LeRobot expects a repository-like identifier even when we only save the dataset locally.
        self.repo_id = (f"local/{self.run_name}")

        self.features = {

            # Defines how images will be stored
            "observation.images.front": {
                "dtype": "image",   # Change to "videos" if we want to see the files in the file explorer as they will be saved differently
                "shape": (
                    settings.dataset_image_height,
                    settings.dataset_image_width,
                    3
                ),
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
            },
            # Defines how observations will be stored
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper_aperture",
                ],
            },
            # Defines how actions will be stored
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_rot_x",
                    "delta_rot_y",
                    "delta_rot_z",
                    "gripper_command",
                ],
            },
        }

        # Create the actual dataset
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root,
            fps=settings.dataset_fps,
            robot_type="PRob3",
            features=self.features,
            use_videos=settings.dataset_use_videos, # Defines the format our images/videos should be saved in
            image_writer_processes=0,   # Save images asynchronously so dataset collection does not slow down the OSC simulation.
            image_writer_threads=2,
        )

        # Define some simulation identifiers
        self.current_episode_frames = 0
        self.saved_episodes = 0
        self.total_saved_frames = 0

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the robosuite camera observation to an HWC uint8 image.
        """

        # Ensure the image is an np.ndarray
        image = np.asarray(image)

        # Ensure the image is in the correct format
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            image = image.astype(np.float32)

            if np.max(image) <= 1.0:
                image = image * 255.0

            image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        if image.shape != (settings.dataset_image_height, settings.dataset_image_width, 3):
            raise ValueError("Unexpected camera image shape.")

        return np.ascontiguousarray(image)

    def _get_gripper_aperture(self, env, observation: dict) -> float:
        """
        Return the current physical gripper aperture normalized to [-1, +1].

        -1 = closed
        +1 = open
        """

        # Grab the gripper position, otherwise None
        gripper_qpos = observation.get("robot0_gripper_qpos", None)

        # Ensure we grabbed the gripper position
        if gripper_qpos is not None:
            # Reshape the array to a 1D array
            gripper_qpos = np.asarray(gripper_qpos, dtype=float).reshape(-1)
            if gripper_qpos.size > 0:
                # Compute the mean and normalize the position of the gripper.
                mean_qpos = float(np.mean(gripper_qpos))
                normalized = (2.0 * (mean_qpos - settings.gripper_q_min) / (settings.gripper_q_max - settings.gripper_q_min) - 1.0)

                return float(np.clip(normalized, -1.0, +1.0))

        # Fallback if robosuite does not expose gripper_qpos.
        # Get the gripper's current action, if we cannot then we default it to a "blank" array (meaning hold position)
        current_action = getattr(env.robots[0].gripper, "current_action", np.array([0.0]))

        return float(np.clip(np.mean(current_action), -1.0, +1.0))

    def record_frame(self, env, action: np.ndarray, observation: dict = None):
        """
        Record observation_t together with action_t.

        LeRobot automatically generates:
            frame_index
            timestamp
            episode_index
            global index
        """

        # Check if an observation was actually passed.
        if observation is None:
            observation = env._get_observations()

        # Define the key for the camera in the obs dict.
        camera_key = settings.dataset_camera_observation_key

        # Check if the camera actually exists.
        if camera_key not in observation:
            raise KeyError(f"Camera observation '{camera_key}' not found.")

        # Format the image as wanted.
        image = self._prepare_image(observation[camera_key])

        # Grab and flatten the robot's actual joint positions
        arm_qpos = np.asarray(env.robots[0]._joint_positions, dtype=np.float32).reshape(-1)

        # Grab the first 6 elements (as the 7th is the gripper!)
        arm_qpos = arm_qpos[:6]

        # Grab the current gripper aperture state
        gripper_aperture = (self._get_gripper_aperture(env, observation))

        # We then concatenate the joints' positions with the gripper state
        state = np.concatenate([arm_qpos, np.array([gripper_aperture], dtype=np.float32)]).astype(np.float32)

        # We re-flatten it.
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # Define the frame
        frame = {
            "observation.images.front": image,
            "observation.state": state,
            "action": action.copy(),
            "task": settings.dataset_task,
        }

        # Add the current frame to the dataset
        self.dataset.add_frame(frame)

        # Increment the number of frames for the current episode
        self.current_episode_frames += 1

    def save_episode(self):
        """
        Save the currently buffered demonstration as one LeRobot episode.
        """

        # Check if there are actually any frames to save
        if self.current_episode_frames == 0:
            print("Episode contains no frames, nothing will be saved.")
            return

        # Grab the number of frames in the episode
        n_frames = self.current_episode_frames

        # Save the episode
        self.dataset.save_episode()

        self.saved_episodes += 1
        self.total_saved_frames += n_frames
        self.current_episode_frames = 0

        print("Episode saved!")

    def discard_episode(self):
        """
        Delete all frames from the current unsuccessful demonstration.
        """

        # If there are no frames, we do not even have anything to discard
        if self.current_episode_frames == 0:
            return

        # Actually discard the episode
        self.dataset.clear_episode_buffer(delete_images=True)

        # Reset the frame counter to 0...
        self.current_episode_frames = 0

    def finalize(self):
        """
        Finalize the LeRobot dataset.

        This must be called before terminating the program so Parquet
        metadata and dataset statistics are correctly written.
        """

        # Never accidentally save an unfinished demonstration.
        if self.current_episode_frames > 0:
            self.discard_episode()

        # Actually finalize the dataset...
        self.dataset.finalize()