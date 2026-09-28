import mink
from mink import Configuration, ConfigurationLimit, FrameTask, PostureTask, SE3, SO3
from robosuite.utils import transform_utils as T
import numpy as np
import mujoco

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

        self.joint_names = np.array([
                "robot0_joint1",
                "robot0_joint2",
                "robot0_joint3",
                "robot0_joint4",
                "robot0_joint5",
                "robot0_joint6",
            ])

    def define_target(self, pos, rot, offset_pos=(0.0, 0.0, 0.10)):
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

    def solve_target_pose(self, q_start, pose, dt=0.01, max_iterations=500, pos_tolerance=1e-3, rot_tolerance=1e-3):

        self.configuration.update(np.asarray(q_start, dtype=float))

        # Set the target pose
        self.posture_task.set_target_from_configuration(self.configuration)
        self.eef_task.set_target(pose)

        tasks = [self.eef_task, self.posture_task]

        # Now iterating over differential IK we find the final q positions.
        for _ in range(max_iterations):

            vel = mink.solve_ik(configuration=self.configuration, tasks=tasks, dt=dt, solver="daqp")

            self.configuration.integrate_inplace(vel, dt)

            # Current task error
            error = self.eef_task.compute_error(self.configuration)

            position_error = np.linalg.norm(error[:3])
            rotation_error = np.linalg.norm(error[3:])

            if position_error < pos_tolerance and rotation_error < rot_tolerance:
                break

        # Return the found q positions
        return self.configuration.q.copy()

    def get_arm_qpos_indices(self):

        indices = []

        for joint_name in self.joint_names:

            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if joint_id == -1:
                raise ValueError(f"Joint name: {joint_name} not found in MJ Model!")

            qpos_index = self.mj_model.jnt_qposadr[joint_id]
            indices.append(qpos_index)

        return np.array(indices, dtype=int)