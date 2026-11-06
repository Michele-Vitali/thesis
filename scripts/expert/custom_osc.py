import numpy as np
import robosuite.controllers.osc as osc_module
from robosuite.controllers import controller_factory
from robosuite.controllers.osc import (
    OperationalSpaceController as RobosuiteOperationalSpaceController,
)


class PRob3OperationalSpaceController(RobosuiteOperationalSpaceController):
    """
    Custom OSC_POSE controller for P-Rob3.

    It behaves exactly like robosuite's original OperationalSpaceController, but adds an upper joint-limit avoidance term for joint 5.

    The avoidance torque is projected into the null-space of the Cartesian POSITION task, so the controller can redistribute motion across the remaining joints while trying to preserve EEF XYZ.

    The normal OSC orientation controller remains active.
    """

    def __init__(self, *args, q5_index=4, q5_avoidance_start=1.90, q5_avoidance_full=1.985, q5_avoidance_max_torque=10.0, q5_avoidance_damping=1.5, **kwargs):
        """Initializes the PRob3OperationalSpaceController.

        Args:
            q5_index (int, optional): The index of the q5 joint. Defaults to 4.
            q5_avoidance_start (float, optional): The threshold value after which we should start using the joint limit avoidance. Defaults to 1.90.
            q5_avoidance_full (float, optional): the threshold value after which we should apply the maximum joint limit avoidance. Defaults to 1.985.
            q5_avoidance_max_torque (float, optional): The maximum torque allowed for the joint limit avoidance. Defaults to 10.0.
            q5_avoidance_damping (float, optional): The damping used for the joint limit avoidance. Defaults to 1.5.
        """
        super().__init__(*args, **kwargs)

        self.q5_index = int(q5_index)
        self.q5_avoidance_start = float(q5_avoidance_start)
        self.q5_avoidance_full = float(q5_avoidance_full)
        self.q5_avoidance_max_torque = float(q5_avoidance_max_torque)
        self.q5_avoidance_damping = float(q5_avoidance_damping)

    def _joint5_position_nullspace_avoidance(self) -> np.ndarray:
        """
        Compute an upper-limit avoidance torque for q5.

        The torque is projected into the null-space of J_pos, therefore it tries to modify the robot posture without affecting Cartesian EEF position XYZ.

        Returns:
            np.ndarray: Joint torque correction.
        """

        # Grab the number of joints we have.
        n_joints = len(self.joint_pos)

        # If the passed index of the q5 joint is not valid we return an empty array (meaning no avoidance) 
        if self.q5_index >= n_joints:
            return np.zeros(n_joints, dtype=float)

        # Grab the current q5's position and velocity.
        q5 = float(self.joint_pos[self.q5_index])
        q5_velocity = float(self.joint_vel[self.q5_index])

        # If we did not yet surpass the avoidance start threshold, we return an empty array (no avoidance)
        if q5 <= self.q5_avoidance_start:
            return np.zeros(n_joints, dtype=float)

        # Check if the passed parameters are valid.
        denominator = (self.q5_avoidance_full - self.q5_avoidance_start)

        if denominator <= 0.0:
            raise ValueError("q5_avoidance_full must be greater than q5_avoidance_start.")

        # Compute the strength factor of the avoidance.
        alpha = (q5 - self.q5_avoidance_start) / denominator
        alpha = np.clip(alpha, 0.0, 1.0)

        # Smoothstep interpolation: avoids introducing a sudden torque discontinuity.
        alpha = (alpha * alpha * (3.0 - 2.0 * alpha))

        # Compute the physical avoidance torque
        avoidance_torque = np.zeros(n_joints, dtype=float)
        avoidance_torque[self.q5_index] = (-self.q5_avoidance_max_torque * alpha)

        # If q5 is still travelling toward the positive limit, oppose that velocity slightly.
        if q5_velocity > 0.0:
            avoidance_torque[self.q5_index] -= (self.q5_avoidance_damping * alpha * q5_velocity)

        # Formulas to compute the needed torque to avoid rreaching the joint limit...
        mass_matrix_inverse = np.linalg.pinv(self.mass_matrix, rcond=1e-8)
        lambda_position_inverse = (self.J_pos @ mass_matrix_inverse @ self.J_pos.T)
        lambda_position = np.linalg.pinv(lambda_position_inverse, rcond=1e-6)
        dynamically_consistent_inverse = (mass_matrix_inverse @ self.J_pos.T @ lambda_position)
        position_nullspace = (np.eye(n_joints, dtype=float) - dynamically_consistent_inverse @ self.J_pos)
        projected_torque = (position_nullspace.T @ avoidance_torque)

        return projected_torque

    def run_controller(self):
        """
        Run standard robosuite OSC first, then add our P-Rob3-specific q5 joint-limit avoidance.

        Returns:
            np.ndarray: The computed torques.
        """

        # Standard robosuite OSC_POSE.
        torques = super().run_controller()

        # Our posture correction.
        avoidance_torque = self._joint5_position_nullspace_avoidance()

        # Compute the "corrected" torques.
        self.torques = (torques + avoidance_torque)

        # Respect the physical actuator torque limits already provided to the robosuite controller.
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