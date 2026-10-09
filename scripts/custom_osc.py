import numpy as np
import robosuite.controllers.controller_factory as controller_factory
import robosuite.controllers.osc as osc_module
from robosuite.controllers.osc import (
    OperationalSpaceController as RobosuiteOperationalSpaceController,
)


class PRob3OperationalSpaceController(RobosuiteOperationalSpaceController):
    """
    Custom OSC_POSE controller for P-Rob3.

    It behaves exactly like robosuite's original OperationalSpaceController,
    but adds an upper joint-limit avoidance term for joint 5.

    The avoidance torque is projected into the null-space of the Cartesian
    POSITION task, so the controller can redistribute motion across the
    remaining joints while trying to preserve EEF XYZ.

    The normal OSC orientation controller remains active.
    """

    def __init__(self, *args, q5_index=4, q5_avoidance_start=1.90, q5_avoidance_full=1.985, q5_avoidance_max_torque=10.0, q5_avoidance_damping=1.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.q5_index = int(q5_index)

        self.q5_avoidance_start = float(q5_avoidance_start)

        self.q5_avoidance_full = float(q5_avoidance_full)

        self.q5_avoidance_max_torque = float(q5_avoidance_max_torque)

        self.q5_avoidance_damping = float(q5_avoidance_damping)

    def _joint5_position_nullspace_avoidance(self) -> np.ndarray:
        """
        Compute an upper-limit avoidance torque for q5.

        The torque is projected into the null-space of J_pos, therefore
        it tries to modify the robot posture without affecting Cartesian
        EEF position XYZ.

        Returns:
            np.ndarray:
                Joint torque correction.
        """

        n_joints = len(self.joint_pos)

        if self.q5_index >= n_joints:
            return np.zeros(n_joints, dtype=float)

        q5 = float(self.joint_pos[self.q5_index])

        q5_velocity = float(self.joint_vel[self.q5_index])

        if q5 <= self.q5_avoidance_start:
            return np.zeros(n_joints, dtype=float)

        denominator = (self.q5_avoidance_full - self.q5_avoidance_start)

        if denominator <= 0.0:
            raise ValueError("q5_avoidance_full must be greater than q5_avoidance_start.")

        alpha = (q5 - self.q5_avoidance_start) / denominator

        alpha = np.clip(alpha, 0.0, 1.0)

        # Smoothstep interpolation:
        # avoids introducing a sudden torque discontinuity.
        alpha = (alpha * alpha * (3.0 - 2.0 * alpha))

        avoidance_torque = np.zeros(n_joints, dtype=float)

        avoidance_torque[self.q5_index] = (-self.q5_avoidance_max_torque * alpha)

        # If q5 is still travelling toward the positive limit,
        # oppose that velocity slightly.
        if q5_velocity > 0.0:
            avoidance_torque[self.q5_index] -= (self.q5_avoidance_damping * alpha * q5_velocity)

        mass_matrix_inverse = np.linalg.pinv(self.mass_matrix, rcond=1e-8)

        lambda_position_inverse = (self.J_pos @ mass_matrix_inverse @ self.J_pos.T)

        lambda_position = np.linalg.pinv(lambda_position_inverse, rcond=1e-6)

        dynamically_consistent_inverse = (mass_matrix_inverse @ self.J_pos.T @ lambda_position)

        position_nullspace = (np.eye(n_joints, dtype=float) - dynamically_consistent_inverse @ self.J_pos)

        projected_torque = (position_nullspace.T @ avoidance_torque)

        return projected_torque

    def run_controller(self):
        """
        Run standard robosuite OSC first, then add our P-Rob3-specific
        q5 joint-limit avoidance.
        """

        # Standard robosuite OSC_POSE.
        torques = super().run_controller()

        # Our posture correction.
        avoidance_torque = (
            self._joint5_position_nullspace_avoidance()
        )

        self.torques = (torques + avoidance_torque)

        # Respect the physical actuator torque limits already provided
        # to the robosuite controller.
        self.torques = self.clip_torques(self.torques)

        return self.torques


def install_prob3_custom_osc():
    """
    Replace robosuite's OperationalSpaceController reference at runtime.

    No robosuite source file is modified.

    robosuite's arm_controller_factory instantiates:

        arm_controllers.OperationalSpaceController(...)

    so replacing that module attribute is sufficient.
    """

    controller_factory.OperationalSpaceController = (PRob3OperationalSpaceController)

    print("[P-Rob3] Custom OSC_POSE controller installed.")