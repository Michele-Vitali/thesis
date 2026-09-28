from mink import Configuration, FrameTask, PostureTask, ConfigurationLimit, SE3, SO3
from robosuite.utils import transform_utils as T
import numpy as np

class IKController:

    def __init__(self, env):
        # Mink setup
        self.env = env
        # Get the MuJoco model for Mink to do further computations
        self.mj_model = self.env.sim.model._model
        self.configuration = Configuration(self.mj_model)
        self.configuration.update(env.sim.data.qpos.copy())

        # Define the two tasks
        # FrameTask handles the reaching of a specific 3D point in space
        self.eef_task = FrameTask(
            frame_name="gripper0_grip_site",
            frame_type="site",
            position_cost=1.0,  # We do not favour traslation over orientation or viceversa
            orientation_cost=1.0
        )

        # PostureTask handles how our robotic arm reaches that point (it's pose/posture!)
        self.posture_task = PostureTask(
            model=self.mj_model,
            cost=1e-2   # As per common default for soft regularization via docs
        )

        # Define the physical joint limits of our robotic arm
        self.joint_limits = [ConfigurationLimit(self.mj_model)]

        # Define a base matrix for top-down grasp.
        self.R_TOP_GRASP = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])

    def define_target(self, pos, rot, offset_pos=[0.0, 0.0, 0.10]):
        """Define the target position we want our robot to reach to then compute the needed IK.

        Args:
            pos (np.array): The array of the 3D position (x, y, z) of the target
            rot (np.array): The rotational matrix of the target objects (3x3 matrix)
            offset_pos (np.array): The array of the position offsets we want in each dimension.
        """

        target_pos = pos + offset_pos
        r_target = rot @ self.R_TOP_GRASP

        return (target_pos, r_target)

    def make_mink_target(self, pos, rot_mat):
        """Generate the target transformation in a usable format for mink.

        Args:
            pos (np.array): The array of the 3D positions of the target point.
            rot_mat (np.array): The rotation matrix we want to achieve.
        """

        return SE3.from_rotation_and_translation(
            SO3.from_matrix(rot_mat),
            pos
        )