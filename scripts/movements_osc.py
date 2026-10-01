import numpy as np
import settings

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from scipy.spatial.transform import Rotation as R


class OSCMovementController:

    # These must correspond to the OSC_POSE controller configuration
    # loaded in main_osc_test.py.
    OSC_POSITION_OUTPUT_MAX = np.array([0.05, 0.05, 0.05], dtype=float)
    OSC_ROTATION_OUTPUT_MAX = np.array([0.5, 0.5, 0.5], dtype=float)

    # Same top-down orientation convention currently used by Mink.
    R_TOP_GRASP = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    def __init__(
        self,
        env: SingleArmEnv,
        gripper_state: float = 1.0,
        tracked_body_name: str = None,
    ):
        self.env = env
        self.gripper_state = gripper_state

        self.tracked_body_name = tracked_body_name
        self.tracked_body_id = None

        if tracked_body_name is not None:
            self.tracked_body_id = self.env.sim.model.body_name2id(
                tracked_body_name
            )

    def _get_eef_pose(self):
        """
        Retrieve the current EEF Cartesian position and orientation.

        Returns:
            tuple:
                - EEF position in world coordinates
                - scipy Rotation describing the EEF orientation
        """

        obs = self.env._get_observations()

        eef_pos = obs["robot0_eef_pos"].copy()

        # robosuite observations expose the EEF quaternion as:
        # [x, y, z, w]
        eef_rotation = R.from_quat(
            obs["robot0_eef_quat"]
        )

        return eef_pos, eef_rotation

    def _get_top_down_rotation(self, object_quat: np.ndarray) -> R:
        """
        Build exactly the same top-down grasp orientation convention
        currently used by the Mink controller.

        The object quaternion comes from MuJoCo and therefore follows:
            [w, x, y, z]

        Only the object's yaw is preserved. Roll and pitch are discarded
        so that the gripper remains perpendicular to the table.
        """

        object_quat = np.asarray(
            object_quat,
            dtype=float
        )

        # MuJoCo quaternion convention:
        # [w, x, y, z]
        object_rotation = R.from_quat(
            object_quat,
            scalar_first=True
        )

        object_matrix = object_rotation.as_matrix()

        # Extract yaw exactly as in IKController._define_target().
        object_yaw = np.arctan2(
            object_matrix[1, 0],
            object_matrix[0, 0]
        )

        cos_yaw = np.cos(object_yaw)
        sin_yaw = np.sin(object_yaw)

        yaw_rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ])

        target_matrix = (
            yaw_rotation
            @ self.R_TOP_GRASP
        )

        return R.from_matrix(target_matrix)

    def _build_osc_action(
        self,
        position_error: np.ndarray,
        rotation_error: np.ndarray,
        position_gain: float = 0.8,
        rotation_gain: float = 0.8,
    ) -> np.ndarray:
        """
        Convert Cartesian position / orientation errors into normalized
        OSC_POSE actions.

        OSC_POSE receives normalized actions in [-1, +1].
        With the configuration used in main_osc_test.py:

            +/- 1 translation -> +/- 0.05 m
            +/- 1 rotation    -> +/- 0.5 rad
        """

        action = np.zeros(
            self.env.action_dim,
            dtype=float
        )

        normalized_position = (
            position_gain
            * position_error
            / self.OSC_POSITION_OUTPUT_MAX
        )

        normalized_rotation = (
            rotation_gain
            * rotation_error
            / self.OSC_ROTATION_OUTPUT_MAX
        )

        action[0:3] = np.clip(
            normalized_position,
            -1.0,
            +1.0
        )

        action[3:6] = np.clip(
            normalized_rotation,
            -1.0,
            +1.0
        )

        # Hold the current gripper position while moving the arm.
        action[6] = 0.0

        return action

    def move_eef_to_pose(
        self,
        target_pos: np.ndarray,
        target_rotation: R,
        max_steps: int = 400,
        position_tolerance: float = 0.005,
        rotation_tolerance_deg: float = 1.0,
    ) -> bool:
        """
        Closed-loop Cartesian motion using OSC_POSE.

        Unlike the JOINT_POSITION version, here every control step is
        computed directly from the current EEF Cartesian error.

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

        target_pos = np.asarray(
            target_pos,
            dtype=float
        ).copy()

        minimum_object_z = np.inf

        if self.tracked_body_id is not None:
            minimum_object_z = (
                self.env.sim.data.body_xpos[
                    self.tracked_body_id
                ][2]
            )

        for step in range(max_steps):

            current_pos, current_rotation = \
                self._get_eef_pose()

            # -----------------------------------------------------
            # POSITION ERROR
            # -----------------------------------------------------
            position_error = (
                target_pos - current_pos
            )

            position_error_norm = np.linalg.norm(
                position_error
            )

            # -----------------------------------------------------
            # ROTATION ERROR
            # -----------------------------------------------------
            # OSC rotation commands are axis-angle deltas.
            # target * current^-1 gives the rotation required to
            # move the current EEF orientation toward the target.
            delta_rotation = (
                target_rotation
                * current_rotation.inv()
            )

            rotation_error = (
                delta_rotation.as_rotvec()
            )

            rotation_error_deg = np.degrees(
                np.linalg.norm(rotation_error)
            )

            # -----------------------------------------------------
            # SUCCESS CHECK
            # -----------------------------------------------------
            if (
                position_error_norm
                < position_tolerance
                and
                rotation_error_deg
                < rotation_tolerance_deg
            ):

                print(
                    f"OSC target reached in {step} steps! "
                    f"Position error: "
                    f"{position_error_norm * 1000:.2f} mm | "
                    f"Rotation error: "
                    f"{rotation_error_deg:.2f} deg"
                )

                if (
                    self.tracked_body_id
                    is not None
                ):
                    print(
                        "Minimum object Z during movement: "
                        f"{minimum_object_z:.5f}"
                    )

                return True

            # -----------------------------------------------------
            # COMPUTE OSC ACTION
            # -----------------------------------------------------
            action = self._build_osc_action(
                position_error,
                rotation_error
            )

            self.env.step(action)

            if self.env.has_renderer:
                self.env.render()

            # -----------------------------------------------------
            # MONITOR OBJECT HEIGHT
            # -----------------------------------------------------
            if self.tracked_body_id is not None:

                object_z = (
                    self.env.sim.data.body_xpos[
                        self.tracked_body_id
                    ][2]
                )

                minimum_object_z = min(
                    minimum_object_z,
                    object_z
                )

        # ---------------------------------------------------------
        # FAILED TO REACH TARGET
        # ---------------------------------------------------------
        current_pos, current_rotation = \
            self._get_eef_pose()

        position_error = (
            target_pos - current_pos
        )

        delta_rotation = (
            target_rotation
            * current_rotation.inv()
        )

        rotation_error_deg = np.degrees(
            delta_rotation.magnitude()
        )

        print(
            "OSC target NOT reached!"
        )

        print(
            "Final position error: "
            f"{np.linalg.norm(position_error) * 1000:.2f} mm"
        )

        print(
            "Final rotation error: "
            f"{rotation_error_deg:.2f} deg"
        )

        if self.tracked_body_id is not None:
            print(
                "Minimum object Z during movement: "
                f"{minimum_object_z:.5f}"
            )

        return False

    def move_robot_to_position(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        max_steps: int = 400,
    ) -> bool:
        """
        Move the EEF to the Cartesian pose corresponding to the passed
        object/reference position.

        We preserve the same convention used by the Mink pipeline:
        safe_offset_gripper is added to the reference point to obtain
        the target grip-site position.
        """

        pos = np.asarray(
            pos,
            dtype=float
        ).copy()

        target_eef_pos = (
            pos
            + np.asarray(
                settings.safe_offset_gripper,
                dtype=float
            )
        )

        target_rotation = \
            self._get_top_down_rotation(quat)

        print(
            "OSC target reference position:",
            np.round(pos, 5)
        )

        print(
            "OSC target EEF position:",
            np.round(target_eef_pos, 5)
        )

        return self.move_eef_to_pose(
            target_pos=target_eef_pos,
            target_rotation=target_rotation,
            max_steps=max_steps
        )

    def elevate_obj(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        max_height: float = settings.ideal_z,
    ) -> np.ndarray | None:
        """
        Raise the object/reference point to max_height using OSC_POSE.
        """

        target_pos = np.asarray(
            pos,
            dtype=float
        ).copy()

        # If the object is already higher than the requested transport
        # height, do not force it downward here.
        target_pos[2] = max(
            target_pos[2],
            max_height
        )

        print(
            f"OSC elevation target: {target_pos}"
        )

        done = self.move_robot_to_position(
            target_pos,
            quat
        )

        if not done:
            return None

        return target_pos

    def move_to_center(
        self,
        current_pos: np.ndarray,
        quat: np.ndarray,
    ) -> np.ndarray | None:
        """
        Transport the grasped object toward the center of the table while
        keeping the SAME reference Z.

        Because OSC works directly on the Cartesian EEF error, Z is
        continuously corrected during the entire motion instead of merely
        being equal at the two joint-space endpoints.
        """

        current_pos = np.asarray(
            current_pos,
            dtype=float
        )

        center_pos = np.array([
            0.0,
            0.0,
            current_pos[2]
        ])

        print(
            f"OSC center target: {center_pos}"
        )

        done = self.move_robot_to_position(
            center_pos,
            quat
        )

        if not done:
            return None

        return center_pos

    def toggle_grab(
        self,
        min_steps: int = 50
    ) -> bool:
        """
        Toggle the high-level gripper command.

        Arm commands remain zero so the OSC controller keeps its current
        Cartesian goal while the gripper opens / closes.
        """

        current_state = self.gripper_state

        self.gripper_state = (
            -current_state
        )

        action = np.zeros(
            self.env.action_dim
        )

        action[6] = self.gripper_state

        for _ in range(min_steps):

            self.env.step(action)

            if self.env.has_renderer:
                self.env.render()

        return True

    def hold(
        self,
        steps: int = 50
    ):
        """
        Keep the current OSC pose and gripper state for visualization.
        """

        action = np.zeros(
            self.env.action_dim
        )

        for _ in range(steps):

            self.env.step(action)

            if self.env.has_renderer:
                self.env.render()