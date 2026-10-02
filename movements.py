# Import useful libraries
import numpy as np
import settings
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from scipy.spatial.transform import Rotation as R


class OSCMovementController:

    # These must correspond to the OSC_POSE controller configuration loaded in main_osc_test.py.
    OSC_POSITION_OUTPUT_MAX = np.array([0.05, 0.05, 0.05], dtype=float)
    OSC_ROTATION_OUTPUT_MAX = np.array([0.5, 0.5, 0.5], dtype=float)

    # Top-down rotation matrix.
    R_TOP_GRASP = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    def __init__(self, env: SingleArmEnv, gripper_state: float = 1.0, tracked_body_name: str = "", recorder=None):

        self.env = env
        self.gripper_state = gripper_state

        # Optional LeRobot dataset recorder.
        self.recorder = recorder

        # Tracked object data.
        self.tracked_body_name = tracked_body_name
        self.tracked_body_id = None

        if tracked_body_name is not None:
            self.tracked_body_id = self.env.sim.model.body_name2id(tracked_body_name)

    def _get_eef_pose(self, observation: dict = None):
        """
        Retrieve the current EEF Cartesian position and orientation.

        Args:
            observation (dict, optional):
                Existing robosuite observation. Reusing it avoids rendering
                the camera twice when recording a LeRobot frame.

        Returns:
            tuple:
                - EEF position in world coordinates
                - scipy Rotation describing the EEF orientation
        """

        # To get the eef_pos we must ensure we have a valid observation
        if observation is None:
            observation = (self.env._get_observations())

        # Copy the current eef_pos
        eef_pos = observation["robot0_eef_pos"].copy()

        # Robosuite observations expose the EEF quaternion as: [x, y, z, w]
        # So we convert it to the convention: [w, x, y, z]
        eef_rotation = R.from_quat(observation["robot0_eef_quat"])

        return eef_pos, eef_rotation

    def _step(self, action: np.ndarray, observation: dict = None, record: bool = True):
        """
        Execute one robosuite control step.

        When a LeRobot recorder is attached, observation_t and action_t are
        stored immediately before executing action_t.
        """

        # If the recording is enabled and a recorder was initialized...
        if record and self.recorder is not None:
            # Record the frame.
            self.recorder.record_frame(env=self.env, action=action, observation=observation)

        result = self.env.step(action)

        # If we initialized the env with a renderer we render the action...
        if self.env.has_renderer:
            self.env.render()

        return result

    def _get_top_down_rotation(self, object_quat: np.ndarray) -> R:
        """
        Build exactly the same top-down grasp orientation convention
        currently used by the Mink controller.

        The object quaternion comes from MuJoCo and therefore follows:
            [w, x, y, z]

        Only the object's yaw is preserved. Roll and pitch are discarded
        so that the gripper remains perpendicular to the table.
        """

        # Convert the object quaternion to a np.ndarray
        object_quat = np.asarray(object_quat, dtype=float)

        # MuJoCo quaternion convention: [w, x, y, z]
        object_rotation = R.from_quat(object_quat, scalar_first=True)

        # Convert the rotation to a rotation matrix
        object_matrix = object_rotation.as_matrix()

        # Extract the object yaw...
        object_yaw = np.arctan2(object_matrix[1, 0], object_matrix[0, 0])

        # And the convert that yaw to a rotation matrix with rotation only on that specific yaw!
        cos_yaw = np.cos(object_yaw)
        sin_yaw = np.sin(object_yaw)

        yaw_rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ])

        # Include in the final matrix the rotation for the top-down grasp as well
        target_matrix = (yaw_rotation @ self.R_TOP_GRASP)

        return R.from_matrix(target_matrix)

    def _build_osc_action(self, position_error: np.ndarray, rotation_error: np.ndarray, position_gain: float = 0.25,
        rotation_gain: float = 0.30, max_position_action: float = 0.40, max_rotation_action: float = 0.40, ) -> np.ndarray:
        """
        Convert Cartesian position / orientation errors into normalized
        OSC_POSE actions.

        The OSC controller receives normalized actions in [-1, +1].

        With the current configuration used in main_osc_test.py:

            +/- 1.0 translation -> +/- 0.05 m
            +/- 1.0 rotation    -> +/- 0.5 rad

        We intentionally limit the normalized commands to +/- 0.40:

            translation -> maximum +/- 0.02 m per action
            rotation    -> maximum +/- 0.20 rad per action

        The proportional gains also make the commanded movement progressively
        smaller as the EEF approaches the target, reducing overshoot.
        """

        # Initialized a "blank" action
        action = np.zeros(self.env.action_dim, dtype=float)

        # Compute the normalized position we must reach...
        normalized_position = (position_gain * position_error / self.OSC_POSITION_OUTPUT_MAX)

        # Compute the translation actions over the x, y, z axes.
        action[0:3] = np.clip(normalized_position, -max_position_action, +max_position_action)

        # Compute the normalized rotation we must reach...
        normalized_rotation = (rotation_gain * rotation_error / self.OSC_ROTATION_OUTPUT_MAX)

        # Compute the rotation actions.
        action[3:6] = np.clip(normalized_rotation, -max_rotation_action, +max_rotation_action)

        # Hold the current gripper position while moving the arm.
        action[6] = 0.0

        return action

    def move_eef_to_pose(self, target_pos: np.ndarray, target_rotation: R, max_steps: int = 400, position_tolerance: float = 0.003,
        rotation_tolerance_deg: float = 1.0, record: bool = True) -> bool:
        """
        Closed-loop Cartesian motion using OSC_POSE.
        Every executed OSC action is optionally recorded as a LeRobot frame.

        Args:
            target_pos:
                Desired EEF world position.

            target_rotation:
                Desired EEF world orientation.

            max_steps:
                Maximum number of controller actions.

            position_tolerance:
                Allowed Cartesian position error in meters.

            rotation_tolerance_deg:
                Allowed orientation error in degrees.

        Returns:
            bool: True if the desired pose was reached.
        """

        # Convert the target position to a np.ndarray and copy it.
        target_pos = np.asarray(target_pos, dtype=float).copy()

        # Define the minimum z of the object...
        minimum_object_z = np.inf

        if self.tracked_body_id is not None:
            # Grab the effective object's z...
            minimum_object_z = (self.env.sim.data.body_xpos[self.tracked_body_id][2])

        for step in range(max_steps):
            # Grab the env's observation...
            observation = self.env._get_observations()

            # Grab the current position and rotation of our eef.
            current_pos, current_rotation = self._get_eef_pose(observation)

            # Compute the current position error and its euclidean norm.
            position_error = (target_pos - current_pos)
            position_error_norm = np.linalg.norm(position_error)

            # Compute the needed delta rotation to get closer to the goal rotation... 
            delta_rotation = (target_rotation * current_rotation.inv())

            # As for the position we compute the current rotation error and its norm in degrees (for better readability)
            rotation_error = (delta_rotation.as_rotvec())
            rotation_error_deg = np.degrees(np.linalg.norm(rotation_error))

            # Check if we reached the target position and rotation...
            if position_error_norm < position_tolerance and rotation_error_deg < rotation_tolerance_deg:
                # If yes we return as we finished our goal.
                print(f"OSC target reached in {step} steps!\nPosition error: {position_error_norm * 1000:.2f} mm | Rotation error: {rotation_error_deg:.2f} deg")
                return True

            # If we did not reach the target position/rotation, we compute the next action...
            action = self._build_osc_action(position_error, rotation_error)

            # Save observation_ and action_t for dataset population...
            self._step(action=action, observation=observation, record=record)

            if self.tracked_body_id is not None:
                object_z = (self.env.sim.data.body_xpos[self.tracked_body_id][2])
                minimum_object_z = min(minimum_object_z, object_z)

        # If we arrive here it means we did not reach the target in max_steps => We are sure to have failed...

        # Grab the actual observation...
        observation = (self.env._get_observations())

        # And current position and rotation...
        current_pos, current_rotation = self._get_eef_pose(observation)

        # Also compute last position and rotation errors
        position_error = (target_pos - current_pos)

        delta_rotation = (target_rotation * current_rotation.inv())
        rotation_error_deg = np.degrees(delta_rotation.magnitude())

        # Debug prints...
        print("OSC target NOT reached!")
        print(f"Final position error: {np.linalg.norm(position_error) * 1000:.2f} mm")
        print(f"Final rotation error: {rotation_error_deg:.2f} deg")

        return False

    def move_robot_to_position(self, pos: np.ndarray, quat: np.ndarray, max_steps: int = 400, record : bool = True) -> bool:
        """
        Move the EEF to the Cartesian pose corresponding to the passed
        object/reference position.
        """

        # Ensure the position is a np.ndarray and copy it...
        pos = np.asarray(pos, dtype=float).copy()

        # Compute the target position and rotation our eef should reach.
        # Note: Remember that the actual eef_site is settings.safe_offset_gripper cm from the tip of the gripper!
        target_eef_pos = (pos + np.asarray(settings.safe_offset_gripper, dtype=float))

        target_rotation = self._get_top_down_rotation(quat)

        # Actual movement
        success = self.move_eef_to_pose(target_pos=target_eef_pos, target_rotation=target_rotation, max_steps=max_steps, record=record)

        return success

    def elevate_obj(self, pos: np.ndarray, quat: np.ndarray, max_height: float = settings.ideal_z + settings.safe_offset_gripper[2]) -> np.ndarray | None:
        """
        Raise the object/reference point to max_height using OSC_POSE.
        """

        # Ensure the position is a np.ndarray and copy it...
        target_pos = np.asarray(pos, dtype=float).copy()

        # If the object is already higher than the ideal z, do NOT lower it...
        target_pos[2] = max(target_pos[2], max_height)

        # Actual elevation
        done = self.move_robot_to_position(target_pos, quat)

        if not done:
            # Something went wrong!
            return None

        return target_pos

    def move_to_center(self, current_pos: np.ndarray, quat: np.ndarray) -> np.ndarray | None:
        """
        Transport the grasped object toward the center of the table while
        keeping the SAME reference Z.
        """

        # Ensure position in a np.ndarray
        current_pos = np.asarray(current_pos, dtype=float)

        center_pos = np.array([0.0, 0.0, current_pos[2]])

        # Actual movement
        done = self.move_robot_to_position(center_pos, quat)

        if not done:
            # Error!
            return None

        return center_pos

    def toggle_grab(self, min_steps: int = 70) -> bool:
        """
        Toggle the high-level gripper command.

        Every closing / opening timestep is also recorded in the LeRobot
        dataset, which is important because the VLA must learn when to
        actuate the gripper.

        Arm commands remain zero so the OSC controller keeps its current
        Cartesian goal while the gripper opens / closes.
        """

        # Get the current gripper state
        current_state = self.gripper_state

        # Invert it!
        self.gripper_state = -current_state

        # Initialize a "blank" action
        action = np.zeros(self.env.action_dim, dtype=float)

        # Set the new grip state
        action[6] = self.gripper_state

        # Execute the action for at least "min_steps" so we are sure the state is completely toggled
        for _ in range(min_steps):
            # Apply the action and save it in the dataset
            self._step(action=action, record=True)

        return True

    
    def hold(self, steps: int = 50, record: bool = False):
        """
        Keep the current OSC pose and gripper state for visualization.

        By default we do not record these frames in the dataset as they are not useful.
        """

        # Initialize the "blank" action
        action = np.zeros(self.env.action_dim)

        # Execute the action for at least "steps"
        for _ in range(steps):
            # Apply the action and eventually record it...
            self._step(action=action, record=record)