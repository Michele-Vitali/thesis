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

    def _get_top_down_rotation(self, object_quat: np.ndarray = None) -> R:
        """
        Build exactly the same top-down grasp orientation convention
        currently used by the Mink controller.

        The object quaternion comes from MuJoCo and therefore follows:
            [w, x, y, z]

        Only the object's yaw is preserved. Roll and pitch are discarded
        so that the gripper remains perpendicular to the table.
        """
        if object_quat is not None:
            # Convert the object quaternion to a np.ndarray
            object_quat = np.asarray(object_quat, dtype=float)

            # MuJoCo quaternion convention: [w, x, y, z]
            object_rotation = R.from_quat(object_quat, scalar_first=True)

            # Convert the rotation to a rotation matrix
            object_matrix = object_rotation.as_matrix()
        else:
            object_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0 ,1.0, 0.0],
                [0.0, 0.0, 1.0]
                ])

        # Extract the object yaw...
        object_yaw = np.arctan2(object_matrix[1, 0], object_matrix[0, 0])

        _, current_rotation = self._get_eef_pose()

        candidate_rotations = []

        # The gripper is parallel => It has a 180° symmetry!
        for candidate_yaw in (object_yaw, object_yaw + np.pi):

            # And the convert that yaw to a rotation matrix with rotation only on that specific yaw!
            cos_yaw = np.cos(candidate_yaw)
            sin_yaw = np.sin(candidate_yaw)

            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw,  cos_yaw, 0.0],
                [0.0,      0.0,     1.0],
            ])

            # Compute the candidate needed rotation to achieve top-down
            candidate_rotation = R.from_matrix(yaw_rotation @ self.R_TOP_GRASP)

            # Compute the magnitude of the distance between the current rotation and the candidate rotation
            rotation_distance = (candidate_rotation * current_rotation.inv()).magnitude()

            # Add the rotation to the candidates
            candidate_rotations.append((rotation_distance, candidate_rotation))

        # Sort the candidates list by the first element of each candidate tuple (meaning sort by the distance magnitude)
        candidate_rotations.sort(key=lambda candidate: candidate[0])

        # Return the candidate rotation of the first sorted candidate!
        return candidate_rotations[1][1]

    def _build_osc_action(self, position_error: np.ndarray, rotation_error: np.ndarray, integral_error: np.ndarray = None, derivative_error: np.ndarray = None, use_xyz_pid: bool = False) -> np.ndarray:
        """
        Convert Cartesian position / orientation errors into normalized
        OSC_POSE actions.

        The OSC controller receives normalized actions in [-1, +1].

        With the current configuration used in main_osc_test.py:

            +/- 1.0 translation -> +/- 0.05 m
            +/- 1.0 rotation    -> +/- 0.5 rad

        We intentionally limit the normalized commands to +/- 0.40:

            translation -> maximum +/- 0.02 m per action
            rotation    -> maximum +/- 0.05 rad per action

        The proportional gains also make the commanded movement progressively
        smaller as the EEF approaches the target, reducing overshoot.
        """

        # Initialized a "blank" action
        action = np.zeros(self.env.action_dim, dtype=float)

        position_error = np.asarray(position_error, dtype=float)
        rotation_error = np.asarray(rotation_error, dtype=float)

        position_error_norm = float(np.linalg.norm(position_error))

        if position_error_norm > settings.position_far_threshold:
            position_gain = settings.position_gain_far
            max_position_action = settings.position_action_far
        elif position_error_norm > settings.position_mid_threshold:
            position_gain = settings.position_gain_mid
            max_position_action = settings.position_action_mid
        else:
            position_gain = settings.position_gain_near
            max_position_action = settings.position_action_near

        proportional_action = position_gain * position_error / self.OSC_POSITION_OUTPUT_MAX
        translation_action = proportional_action.copy()

        if use_xyz_pid:
            if integral_error is None:
                integral_error = np.zeros(3, dtype=float)

            if derivative_error is None:
                derivative_error = np.zeros(3, dtype=float)

            integral_action = settings.xyz_integral_gain * integral_error / self.OSC_POSITION_OUTPUT_MAX
            integral_action = np.clip(integral_action, -settings.xyz_integral_action_limit, +settings.xyz_integral_action_limit)

            integral_action[2] = max(0.0, float(integral_action[2]))

            derivative_action = settings.xyz_derivative_gain * derivative_error / self.OSC_POSITION_OUTPUT_MAX
            derivative_action = np.clip(derivative_action, -settings.xyz_derivative_action_limit, +settings.xyz_derivative_action_limit)

            translation_action += integral_action + derivative_action

        action[0:3] = np.clip(translation_action, -max_position_action, +max_position_action)

        rotation_error_deg = float(np.degrees(np.linalg.norm(rotation_error)))

        if rotation_error_deg > settings.rotation_far_threshold_deg:
            rotation_gain = settings.rotation_gain_far
            max_rotation_action = settings.rotation_action_far

        elif rotation_error_deg > settings.rotation_mid_threshold_deg:
            rotation_gain = settings.rotation_gain_mid
            max_rotation_action = settings.rotation_action_mid

        else:
            rotation_gain = settings.rotation_gain_near
            max_rotation_action = settings.rotation_action_near

        normalized_rotation = rotation_gain * rotation_error / self.OSC_ROTATION_OUTPUT_MAX

        action[3:6] = np.clip(normalized_rotation, -max_rotation_action, +max_rotation_action)

        # Hold the current gripper position while moving the arm.
        action[6] = 0.0

        return action

    def move_eef_to_pose(
    self,
    target_pos: np.ndarray,
    target_rotation: R,
    max_steps: int = 400,
    position_tolerance: float = settings.position_tolerance,
    rotation_tolerance_deg: float = settings.rotation_tolerance_deg,
    payload_compensation: bool = False,
    slow_descent: bool = False,
    record: bool = True,
) -> bool:
        """
        Closed-loop Cartesian motion using OSC_POSE.

        When payload_compensation=True, a full XYZ Cartesian PID correction
        is enabled.

        X / Y integral:
            signed.

        Z integral:
            positive-only, because it compensates payload gravity.

        XY integral accumulation is frozen while q5 is near its upper
        joint limit, preventing integral wind-up against a kinematic
        constraint.
        """

        target_pos = np.asarray(
            target_pos,
            dtype=float,
        ).copy()

        dt = (
            1.0
            / float(
                settings.dataset_fps
            )
        )

        # ================================================================
        # XYZ PID STATE
        # ================================================================

        integral_error = np.zeros(
            3,
            dtype=float,
        )

        previous_position_error = None

        filtered_derivative = np.zeros(
            3,
            dtype=float,
        )

        for step in range(max_steps):

            observation = (
                self.env._get_observations()
            )

            current_pos, current_rotation = (
                self._get_eef_pose(
                    observation
                )
            )

            # ================================================================
            # POSITION ERROR
            # ================================================================

            position_error = (
                target_pos
                - current_pos
            )

            position_error_norm = float(
                np.linalg.norm(
                    position_error
                )
            )

            # ================================================================
            # ROTATION ERROR
            # ================================================================

            delta_rotation = (
                target_rotation
                * current_rotation.inv()
            )

            rotation_error = (
                delta_rotation.as_rotvec()
            )

            rotation_error_deg = float(
                np.degrees(
                    np.linalg.norm(
                        rotation_error
                    )
                )
            )

            # ================================================================
            # SUCCESS
            # ================================================================

            if (
                position_error_norm
                < position_tolerance
                and
                rotation_error_deg
                < rotation_tolerance_deg
            ):

                print(
                    f"OSC target reached in {step} steps!\n"
                    f"Position error: "
                    f"{position_error_norm * 1000:.2f} mm | "
                    f"Rotation error: "
                    f"{rotation_error_deg:.2f} deg"
                )

                return True

            # ================================================================
            # XYZ PID
            # ================================================================

            if payload_compensation:

                # ------------------------------------------------------------
                # Check q5.
                # ------------------------------------------------------------

                joint_positions = (
                    self.env.robots[0]._joint_positions
                )

                q5 = float(
                    joint_positions[4]
                )

                freeze_xy_integral = (
                    q5
                    >= settings.xyz_pid_q5_freeze_threshold
                )

                # ------------------------------------------------------------
                # Integral.
                # ------------------------------------------------------------

                for axis in range(3):

                    # Prevent XY wind-up while q5 is saturated.
                    if (
                        axis < 2
                        and
                        freeze_xy_integral
                    ):
                        continue

                    axis_error = float(
                        position_error[axis]
                    )

                    if (
                        abs(axis_error)
                        <= settings.xyz_integral_deadband[axis]
                    ):

                        integral_error[axis] *= 0.95
                        continue

                    if integral_error[axis] != 0.0 and np.sign(axis_error) != np.sign(integral_error[axis]):
                        integral_error[axis] *= 0.80
                    
                    integral_error[axis] += axis_error * dt

                # Signed X integral.
                integral_error[0] = float(
                    np.clip(
                        integral_error[0],
                        -settings.xyz_integral_limit[0],
                        +settings.xyz_integral_limit[0],
                    )
                )

                # Signed Y integral.
                integral_error[1] = float(
                    np.clip(
                        integral_error[1],
                        -settings.xyz_integral_limit[1],
                        +settings.xyz_integral_limit[1],
                    )
                )

                # Positive-only Z integral.
                #
                # Negative Z error can discharge accumulated compensation,
                # but the integral itself can never become negative.
                integral_error[2] = float(
                    np.clip(
                        integral_error[2],
                        0.0,
                        settings.xyz_integral_limit[2],
                    )
                )

                # ------------------------------------------------------------
                # Derivative.
                # ------------------------------------------------------------

                if previous_position_error is None:

                    raw_derivative = np.zeros(
                        3,
                        dtype=float,
                    )

                else:

                    raw_derivative = (
                        position_error
                        - previous_position_error
                    ) / dt

                alpha = float(
                    settings.xyz_derivative_filter_alpha
                )

                filtered_derivative = (
                    alpha
                    * raw_derivative
                    +
                    (1.0 - alpha)
                    * filtered_derivative
                )

            # ================================================================
            # BUILD ACTION
            # ================================================================

            action = self._build_osc_action(
                position_error=position_error,
                rotation_error=rotation_error,
                integral_error=integral_error,
                derivative_error=filtered_derivative,
                use_xyz_pid=payload_compensation,
            )

            # ================================================================
            # SLOW FINAL DESCENT
            # ================================================================

            if slow_descent:

                # Strongly limit only downward Z motion.
                action[2] = float(
                    np.clip(
                        action[2],
                        -0.10,
                        +0.20,
                    )
                )

            # ================================================================
            # EXECUTE
            # ================================================================

            self._step(
                action=action,
                observation=observation,
                record=record,
            )

            previous_position_error = (
                position_error.copy()
            )

        # ================================================================
        # FAILURE REPORT
        # ================================================================

        observation = (
            self.env._get_observations()
        )

        current_pos, current_rotation = (
            self._get_eef_pose(
                observation
            )
        )

        position_error = (
            target_pos
            - current_pos
        )

        delta_rotation = (
            target_rotation
            * current_rotation.inv()
        )

        rotation_error_deg = float(
            np.degrees(
                delta_rotation.magnitude()
            )
        )

        with np.printoptions(
            suppress=True
        ):

            print(
                "OSC target NOT reached!"
            )

            print(
                f"Final position: "
                f"{current_pos}, "
                f"Desired position: "
                f"{target_pos}"
            )

            print(
                f"Final XYZ error [mm]: "
                f"{position_error * 1000}"
            )

            print(
                f"Final position error: "
                f"{np.linalg.norm(position_error) * 1000:.2f} mm"
            )

            print(
                f"Final rotation error: "
                f"{rotation_error_deg:.2f} deg"
            )

            print(
                f"PID integral state: "
                f"{integral_error}"
            )

        joint_positions = (
            self.env.robots[0]._joint_positions
        )

        joint_limits = [
            2.96,
            1.91,
            2.0,
            2.95,
            2.0,
            2.96,
        ]

        with np.printoptions(
            suppress=True
        ):

            print(
                f"Joint positions: "
                f"{joint_positions}"
            )

        margin = 0.01

        for i, joint_pos in enumerate(
            joint_positions
        ):

            real_joint_index = (
                i + 1
            )

            if (
                joint_pos
                < -joint_limits[i] + margin
                or
                joint_pos
                > joint_limits[i] - margin
            ):

                print(
                    f"Joint [{real_joint_index}]: "
                    f"Reached its limit! "
                    f"Limit: {joint_limits[i]}, "
                    f"Joint Position: "
                    f"{joint_positions[i]}"
                )

        return False
    
    def move_robot_to_position(self, pos: np.ndarray, quat: np.ndarray = None, max_steps: int = 400, position_tolerance: float = settings.position_tolerance, rotation_tolerance_deg: float = settings.rotation_tolerance_deg, payload_compensation: bool = False, slow_descent: bool = False, record: bool = True) -> bool:
        """
        Move the EEF directly to the Cartesian pose corresponding to the
        passed object / reference position.
        """

        pos = np.asarray(pos, dtype=float).copy()

        if position_tolerance is None:
            position_tolerance = settings.position_tolerance

        target_eef_pos = pos + np.asarray(settings.safe_offset_gripper, dtype=float)

        target_rotation = self._get_top_down_rotation(quat)

        return self.move_eef_to_pose(target_pos=target_eef_pos, target_rotation=target_rotation, max_steps=max_steps, position_tolerance=position_tolerance, rotation_tolerance_deg=rotation_tolerance_deg, payload_compensation=payload_compensation, slow_descent=slow_descent, record=record)

    """
    def elevate_obj(self, pos: np.ndarray, quat: np.ndarray, max_height: float = settings.ideal_z + settings.safe_offset_gripper[2]) -> np.ndarray | None:
        
        Raise the object/reference point to max_height using OSC_POSE.
        

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
    """

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

    def toggle_grab(self, max_steps: int = 50, settle_steps: int = 4, record: bool = True) -> bool:
        """
        Toggle the high-level gripper command.

        Every closing / opening timestep is also recorded in the LeRobot
        dataset, which is important because the VLA must learn when to
        actuate the gripper.

        Arm commands remain zero so the OSC controller keeps its current
        Cartesian goal while the gripper opens / closes.
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
            observation = self.env._get_observations()

            self._step(action=action, observation=observation, record=record)
            current_command = np.asarray(self.env.robots[0].gripper.current_action, dtype=float).reshape(-1)

            if current_command.size > 0 and abs(float(current_command[0]) - target_state) < 0.02:
                reached = True
                break

        # Execute an hold action to give time to the fingers to settle.
        hold_action = np.zeros(self.env.action_dim, dtype=float)
        for _ in range(settle_steps):
            self._step(action=hold_action, record=record)

        if not reached:
            print("Griiper did not fully toggle!")

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