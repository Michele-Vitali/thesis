# Import useful libraries
import numpy as np
import settings
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from scipy.spatial.transform import Rotation as R


class OSCMovementController:

    # These define the maximum position and rotation outputs our robot is able to erogate
    OSC_POSITION_OUTPUT_MAX = np.array([0.05, 0.05, 0.05], dtype=float)
    OSC_ROTATION_OUTPUT_MAX = np.array([0.5, 0.5, 0.5], dtype=float)

    # Top-down rotation matrix.
    R_TOP_GRASP = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    def __init__(self, env: SingleArmEnv, gripper_state: float = 1.0, recorder = None):
        """Initalizes an OSCMovementController.

        Args:
            env (SingleArmEnv): The simulation environment.
            gripper_state (float, optional): The initial gripper state. Defaults to 1.0.
            recorder (_type_, optional): The HDF5 recorder used to record the episodes. Defaults to None.
        """

        self.env = env
        self.gripper_state = gripper_state

        # Optional HDF5 dataset recorder.
        self.recorder = recorder

    def _get_eef_pose(self, observation: dict | None = None) -> tuple[np.ndarray, R]:
        """
        Retrieve the current EEF Cartesian position and orientation.

        Args:
            observation (dict, optional): The already existing robosuite's current observation.

        Returns:
            tuple[np.ndarray, R]: A tuple containing the eef's position array [x, y, z] and the corresponding Scipy Rotation to the eef's rotation. 
        """

        # To get the eef_pos we must ensure we have a valid observation
        if observation is None:
            observation = (self.env._get_observations())

        # Copy the current eef_pos
        eef_pos = observation["robot0_eef_pos"].copy()

        # Robosuite observations expose the EEF quaternion as: [x, y, z, w]
        # So we convert it to the convention: [w, x, y, z]
        eef_rotation = R.from_quat(observation["robot0_eef_quat"])

        return (eef_pos, eef_rotation)

    def _step(self, action: np.ndarray, observation: dict | None = None, record: bool = True) -> tuple[dict, float, bool, dict]:
        """
        Execute one robosuite control step.

        When an HDF5 recorder is attached, observation_t and action_t are stored immediately before executing action_t.

        Args:
            action (np.ndarray): The action we should execute.
            observation (dict, optional): The current observations retrieved from the environment. Defaults to None.
            record (bool, optional): Whether we should record the episode or not. Defaults to True.
        
        Returns:
            tuple[dict, float, bool, dict]: The environment's step results:
                - (dict): An ordered dict that contains information about the environment.
                - (float): The granted reward from the RL algorithm.
                - (bool): Whether the current episode was completed or not.
                - (dict): Misc information.
        """

        # If the recording is enabled and a recorder was initialized...
        if record and self.recorder is not None:
            # Record the frame.
            self.recorder.record_frame(env=self.env, action=action, observation=observation)

        # Execute the action and store the result in a 4-tuple variable 
        result = self.env.step(action)

        # If we initialized the env with a renderer we render the action...
        if self.env.has_renderer:
            self.env.render()

        return result

    def _get_top_down_rotation(self, object_quat: np.ndarray = None) -> R:
        """
        Build the best rotation we must carry on to achieve the final desired top-down rotation.

        The object quaternion comes from MuJoCo and therefore follows the [w, x, y, z] convention!

        Note: Only the object's yaw is preserved. Roll and pitch are discarded so that the gripper remains perpendicular to the table.

        Args:
            object_quat (np.ndarray, optional): The object's quaternion. Defaults to None.
        
        Returns:
            R: The Scipy Rotation corresponding to the needed rotation that the eef should carry out to achieve the top-down pose.
        """
        
        if object_quat is not None:
            # Convert the object quaternion to a np.ndarray
            object_quat = np.asarray(object_quat, dtype=float)

            # MuJoCo quaternion convention: [w, x, y, z]
            object_rotation = R.from_quat(object_quat, scalar_first=True)

            # Convert the rotation to a rotation matrix
            object_matrix = object_rotation.as_matrix()
        else:
            # If no quaternion was passed, then we define the rotation matrix as the identity matrix (basically no rotation)
            object_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0 ,1.0, 0.0],
                [0.0, 0.0, 1.0]
                ])

        # Extract the object yaw...
        object_yaw = np.arctan2(object_matrix[1, 0], object_matrix[0, 0])

        # Get the current eef rotation
        _, current_rotation = self._get_eef_pose()

        candidate_rotations = []

        # The gripper is parallel => It has a 180° symmetry!
        for candidate_yaw in (object_yaw, object_yaw + np.pi):

            # Convert the candidate yaw to a rotation matrix
            cos_yaw = np.cos(candidate_yaw)
            sin_yaw = np.sin(candidate_yaw)

            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw,  cos_yaw, 0.0],
                [0.0,      0.0,     1.0],
            ])

            # Compute the needed candidate rotation to achieve a top-down pose
            candidate_rotation = R.from_matrix(yaw_rotation @ self.R_TOP_GRASP)

            # Compute the magnitude of the distance between the current rotation and the candidate rotation
            rotation_distance = (candidate_rotation * current_rotation.inv()).magnitude()

            # Add the rotation distance and the candidate rotation to the candidates
            candidate_rotations.append((rotation_distance, candidate_rotation))

        # Sort the candidates list by the first element of each candidate tuple (meaning sort by the distance magnitude)
        candidate_rotations.sort(key=lambda candidate: candidate[0])

        # Return the candidate rotation of the first sorted candidate!
        return candidate_rotations[0][1]

    def _build_osc_action(self, position_error: np.ndarray, rotation_error: np.ndarray, position_gain: float = settings.position_gain, rotation_gain: float = settings.rotation_gain,
                          max_position_action: float = settings.max_position_action, max_rotation_action: float = settings.max_rotation_action) -> np.ndarray:
        """
        Convert Cartesian position / orientation errors into normalized OSC_POSE actions.

        The OSC controller receives normalized actions in the [-1, +1] range.

        With the current configuration used:
            +/- 1.0 translation -> +/- 0.05 m
            +/- 1.0 rotation    -> +/- 0.5 rad

        We intentionally limit the normalized commands to +/- 0.40:
            translation -> maximum +/- 0.02 m per action
            rotation    -> maximum +/- 0.05 rad per action

        The proportional gains also make the commanded movement progressively smaller as the EEF approaches the target, reducing overshoot.

        Args:
            position_error (np.ndarray): The current position error between the eef position and the target position.
            rotation_error (np.ndarray): The current rotation error between the eff rotation and the target rotation.
            position_gain (float): The gain used for the translation movements. A factor used to define how strong should the translation action be. Defaults to the value chosen in the 'settings' file.
            rotation_gain (float): The gain used for the rotation movements. A factor used to define how strong should the rotation action be. Defaults to the value chosein in the 'settings' file.
            max_position_action (float): The maximum translational value our controller should output in the case of a translational action. Default to the value chosen in the 'settings' file.
            max_rotation_action (float): The maximum rotational value our controlle should output in the case of a rotational action. Defaults to the value chosen in the 'settings' file.

        Returns:
            np.ndarray: The built osc action.
        """

        # Initialized a "blank" action
        action = np.zeros(self.env.action_dim, dtype=float)

        # We compute the normalized translation action needed to minimize the position error.
        normalized_position = position_gain * position_error / self.OSC_POSITION_OUTPUT_MAX

        # We clip the translation action between the maximum values (for safety).
        action[0:3] = np.clip(normalized_position, -max_position_action, +max_position_action)

        normalized_rotation = rotation_gain * rotation_error / self.OSC_ROTATION_OUTPUT_MAX

        action[3:6] = np.clip(normalized_rotation, -max_rotation_action, +max_rotation_action)

        # Hold the current gripper position while moving the arm.
        action[6] = 0.0

        return action

    def move_eef_to_pose(self, target_pos: np.ndarray, target_rotation: R, max_steps: int = 400, position_tolerance: float = settings.position_tolerance,
        rotation_tolerance_deg: float = settings.rotation_tolerance_deg, payload_compensation: bool = False, record: bool = True) -> tuple[bool, int]:
        """
        Closed-loop Cartesian motion using OSC_POSE.
        Every executed OSC action is optionally recorded as an HDF5 frame.

        Args:
            target_pos (np.ndarray): Desired EEF world position.
            target_rotation (R): Desired EEF world orientation.
            max_steps (int, optional): Maximum number of controller actions. Defaults to 400
            position_tolerance (float, optional): Allowed Cartesian position error in meters. Defaults to the value chosen in the 'settings' file.
            rotation_tolerance_deg (float, optional): Allowed orientation error in degrees. Default to the value chosen in the 'settings' file.
            payload_compensation (bool, optional): Whether the function should also use the payload compensation feature to compensate inertia. Defaults to False.
            record (bool, optional): Whether we should record the episode or not. Defaults to True.

        Returns:
            tuple[bool, int]: A tuple containing a boolean that indicates whether the desired pose was reached or not, and an integer indicating the number of steps that were carried out.
        """

        # Convert the target position to a np.ndarray and copy it.
        target_pos = np.asarray(target_pos, dtype=float).copy()

        # Integral component variables...
        dt = 1.0 /float(settings.dataset_fps)
        integral_z = 0.0

        for step in range(max_steps):
            # Grab the env's observations...
            try:
                # If it is the first step, then we force update (this fixes the error of the first frame having not randomized features yet!)
                observation = self.env._get_observations(force_update=(step==0))
            except TypeError:
                # Fallback for older robosuite's versions...
                observation = self.env._get_observations()

            # Grab the current position and rotation of our eef.
            current_pos, current_rotation = self._get_eef_pose(observation)

            # Compute the current position error and its euclidean norm.
            position_error = (target_pos - current_pos)

            # Compute the error solely on the Z-axis.
            z_error = float(position_error[2])

            # Check if the absolute value of the error falls within the threshold of using the integral component...
            if abs(z_error) > settings.z_integral_deadband:
                # Clip the value of the integral on the z axis between the defined limits
                integral_z = float(np.clip(integral_z, -settings.z_integral_limit, +settings.z_integral_limit))

            # Compute the needed delta rotation to get closer to the goal rotation... 
            delta_rotation = (target_rotation * current_rotation.inv())

            # As for the position we compute the current rotation error and its norm in degrees (for better readability)
            rotation_error = (delta_rotation.as_rotvec())
            rotation_error_deg = np.degrees(np.linalg.norm(rotation_error))

            position_error_norm = np.linalg.norm(position_error)

            # Check if we reached the target position and rotation...
            if position_error_norm < position_tolerance and rotation_error_deg < rotation_tolerance_deg:
                # If yes we return as we finished our goal.
                #print(f"OSC target reached in {step} steps!\nPosition error: {position_error_norm * 1000:.2f} mm | Rotation error: {rotation_error_deg:.2f} deg")
                return (True, step)

            # If we did not reach the target position/rotation, we compute the next action...
            action = self._build_osc_action(position_error, rotation_error)

            # Actual integral Z component's code.
            if payload_compensation:
                if abs(z_error) > settings.z_integral_deadband:
                    # Compute the new value for the Z integral component and clip it between the bounds.
                    integral_z += z_error * dt
                    integral_z = float(np.clip(integral_z, -settings.z_integral_limit, +settings.z_integral_limit))
                # Integral Z action
                integral_action_z = settings.z_integral_gain * integral_z / self.OSC_POSITION_OUTPUT_MAX[2]
                integral_action_z = float(np.clip(integral_action_z, -settings.z_integral_action_limit, +settings.z_integral_action_limit))

                # Add the integral Z action to the normal Z action 
                action[2] += integral_action_z
                action[2] = float(np.clip(action[2], -0.40, +0.40))

            # Save observation_ and action_t for dataset population...
            self._step(action=action, observation=observation, record=record)

        # If we arrive here it means we did not reach the target in max_steps => We are sure to have failed!
        print("OSC target NOT reached!")

        """DEBUG/PRINT CODE

        # Grab the actual observation...
        observation = (self.env._get_observations())

        # And current position and rotation...
        current_pos, current_rotation = self._get_eef_pose(observation)

        # Also compute last position and rotation errors
        position_error = (target_pos - current_pos)

        delta_rotation = (target_rotation * current_rotation.inv())
        rotation_error_deg = np.degrees(delta_rotation.magnitude())

        # Debug prints...
        with np.printoptions(suppress=True):
            print(f"Final position: {current_pos}, Desired position: {target_pos}")
            print(f"Final position error: {np.linalg.norm(position_error) * 1000:.2f} mm")
            print(f"Final rotation error: {rotation_error_deg:.2f} deg")

        
        joint_positions = self.env.robots[0]._joint_positions
        joint_limits = [2.96, 1.91, 2.0, 2.95, 2.0, 2.96]

        with np.printoptions(suppress=True):
            print(f"Joint positions: {joint_positions}")
        margin = 0.01
        for i, joint_pos in enumerate(joint_positions):
            real_joint_index = i+1
            if joint_pos < -joint_limits[i] + margin or joint_pos > joint_limits[i] - margin:
                print(f"Joint [{real_joint_index}]: Reached its limit! Limit: {joint_limits[i]}, Joint Position: {joint_positions[i]}")

        position_ok = np.linalg.norm(position_error) < position_tolerance
        rotation_ok = rotation_error_deg < rotation_tolerance_deg

        print(f"Position OK: {position_ok} (tol={position_tolerance * 1000:.2f} mm)")
        print(f"Rotation OK: {rotation_ok} (tol={rotation_tolerance_deg:.2f} deg)")
        """

        return (False, step)
    
    def move_robot_to_position(self, pos: np.ndarray, quat: np.ndarray = None, max_steps: int = 400, position_tolerance: float = settings.position_tolerance, rotation_tolerance_deg: float = settings.rotation_tolerance_deg, payload_compensation: bool = False, record: bool = True) -> tuple[bool, int]:
        """
        Move the EEF directly to the Cartesian pose corresponding to the passed position.

        Args:
            pos (np.ndarray): The position we want to reach.
            quat (np.ndarray, optional): The orientation we want to reach. Defaults to None.
            max_steps (int, optional): The maximum number of steps our algorithm should take. Defaults to 400.
            position_tolerance (float, optional): The maximum allowed position tolerance in our translational movements. Default to the value chosen in the 'settings' file.
            rotation_tolerance_deg (float, optional): The maximum allowe rotation tolerance in degrees in our rotational movements. Defaults to the value chosen in the 'settings' file.
            payload_compensation (bool, optional): Whether we should also use the integral action to compensate the inertia. Defaults to False.
            record (bool, optional): Whether we should record the episode or not. Defaulto to True.

        Returns:
            tuple[bool, int]: A tuple containing a boolean that indicates whether the desired position and rotation were reached, and an int indicating the number of total steps were carried out.

        """

        # Ensure the position in a np array.
        pos = np.asarray(pos, dtype=float).copy()

        # Compute the target position and rotation.
        target_eef_pos = pos + np.asarray(settings.safe_offset_gripper, dtype=float)
        target_rotation = self._get_top_down_rotation(quat)

        # Return the tuple that we receive from the move_eef_to_pose function!
        return self.move_eef_to_pose(target_pos=target_eef_pos, target_rotation=target_rotation, max_steps=max_steps, position_tolerance=position_tolerance, rotation_tolerance_deg=rotation_tolerance_deg, payload_compensation=payload_compensation, record=record)

    def toggle_grab(self, max_steps: int = 50, settle_steps: int = 4, record: bool = True) -> bool:
        """
        Toggles the current gripper's state. If the action was holding the current pose, we keep holding it!

        If:
            - +1 (Open) => -1 (Closed)
            - -1 (Closed) => +1 (Open)
            - 0.0 (Hold) => 0.0 (Hold)

        Args:
            max_steps (int, optional): The maximum number of steps the toggling action should take. Defaults to 50.
            settle_steps (int, optional): The number of steps our simulation should wait for the fingers to settle in the correct position. Defaults to 4.
            record (bool, optional): Whether we should record the episode or not. Defaults to True.

        Returns:
            bool: Whether the togglin action was completed or not.
        """

        # Invert the current gripper state!
        self.gripper_state = -self.gripper_state
        target_state = float(self.gripper_state)

        # Initialize a "blank" action
        action = np.zeros(self.env.action_dim, dtype=float)

        # Set the new grip state
        action[6] = target_state

        # Execute the action for at max "max_steps" until we are sure the state is completely toggled
        reached = False
        for _ in range(max_steps):
            # Grab the current observation.
            observation = self.env._get_observations()

            # Execute the step and eventually record it.
            self._step(action=action, observation=observation, record=record)

            # Grab the current action the gripper is executing and flatten it.
            current_command = np.asarray(self.env.robots[0].gripper.current_action, dtype=float).reshape(-1)

            # Condition to early-stop the toggling if the target is already reached!
            if current_command.size > 0 and abs(float(current_command[0]) - target_state) < 0.02:
                reached = True
                break

        # Execute an hold action to give time to the fingers to settle.
        hold_action = np.zeros(self.env.action_dim, dtype=float)
        for _ in range(settle_steps):
            self._step(action=hold_action, record=record)

        if not reached:
            print("Gripper did not fully toggle!")

        return True