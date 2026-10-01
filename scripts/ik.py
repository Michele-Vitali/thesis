import mink
import mujoco
import numpy as np
import settings
from mink import SE3, SO3, Configuration, ConfigurationLimit, FrameTask, PostureTask
from robosuite.utils import transform_utils as T


class IKController:

    def __init__(self, env):
        # Mink setup
        self.env = env
        # Get the MuJoco model for Mink to do further computations
        self.mj_model = self.env.sim.model._model
        self.configuration = Configuration(self.mj_model)
        self.configuration.update(env.sim.data.qpos.copy())

        # Position-only task, useful as first stage of the IK.
        self.position_task = FrameTask(
            frame_name="gripper0_grip_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0
        )

        # Full pose task.
        self.eef_task = FrameTask(
            frame_name="gripper0_grip_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0
        )

        # Soft preference toward the actual robot posture.
        self.posture_task = PostureTask(
            model=self.mj_model,
            cost=1e-3
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

    def _define_target(self, pos, rot=None):
        """Define the target position we want our robot to reach to then compute the needed IK.

        Args:
            pos (np.array): The array of the 3D position (x, y, z) of the target
            rot (np.array): The rotational matrix of the target objects (3x3 matrix)
            offset_pos (np.array): The array of the position offsets we want in each dimension.
        """

        target_pos = np.asarray(pos, dtype=float).copy()

        if rot is None:
            r_target = self.R_TOP_GRASP.copy()
        else:
            # We only care about the object's yaw.
            # Roll and pitch must not tilt our top-down grasp.
            object_yaw = np.arctan2(rot[1, 0], rot[0, 0])

            cos_yaw = np.cos(object_yaw)
            sin_yaw = np.sin(object_yaw)

            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw,  cos_yaw, 0.0],
                [0.0,      0.0,     1.0]
            ])

            r_target = yaw_rotation @ self.R_TOP_GRASP

        return (target_pos, r_target)

    def _make_mink_target(self, pos, rot_mat):
        """Generate the target transformation in a usable format for mink.

        Args:
            pos (np.array): The array of the 3D positions of the target point.
            rot_mat (np.array): The rotation matrix we want to achieve.
        """

        return SE3.from_rotation_and_translation(
            SO3.from_matrix(rot_mat),
            pos
        )

    def _get_arm_joint_limits(self):
        """Retrieve every joint's angular movement limit.

        Raises:
            ValueError: Raised if the joint with that specific name cannot be found.

        Returns:
            np.ndarray, np.ndarray: The first array contains the lower limits, the second the higher limits.
        """
        lower_limits = []
        upper_limits = []

        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )

            if joint_id == -1:
                raise ValueError(f"Joint name: {joint_name} not found in MJ Model!")

            lower, upper = self.mj_model.jnt_range[joint_id]

            lower_limits.append(lower)
            upper_limits.append(upper)

        return np.array(lower_limits), np.array(upper_limits)

    def _get_normalized_joint_margins(self, q_solution):
        """Compute the normalized distance of every arm joint from its closest limit.

        Args:
            q_solution (np.ndarray): Full MuJoCo configuration containing the proposed solution.

        Returns:
            np.ndarray: Normalized margin of every arm joint from its closest joint limit.
        """

        # Retrieve the indices corresponding to the robot arm joints.
        arm_indices = self.get_arm_qpos_indices()

        # Retrieve the physical limits of every arm joint.
        lower_limits, upper_limits = self._get_arm_joint_limits()

        # Extract only the arm joint positions from the complete configuration.
        solution_q = q_solution[arm_indices]

        # Compute the total movement range of every joint.
        joint_range = upper_limits - lower_limits

        # Compute the normalized distance from both limits.
        lower_margin = (solution_q - lower_limits) / joint_range
        upper_margin = (upper_limits - solution_q) / joint_range

        # The real margin is the distance from the closest limit.
        normalized_margin = np.minimum(lower_margin, upper_margin)

        return normalized_margin

    def _generate_seeds(self, q_start, n_local_seeds=4, n_global_seeds=8):        
        """This functions generates some partially fixed seeds and some random seeds 
        close to the actual robot posture and some that are truly random.

        Args:
            q_start (np.ndarray): The starting joints' positions.
            n_local_seeds (int, optional): Defines how many random seeds close to the actual robot posture should be generated.
                                           Defaults to 4.
            n_global_seeds (int, optional): Defines how many truly random seeds within the robot joint space should be generated. 
                                           Defaults to 8.

        Returns:
            list: The list containing all of the generated seeds.
        """

        # Retrieve the joints' indices in the robot 
        arm_indices = self.get_arm_qpos_indices()
        # Retrieve the joints' limits
        lower_limits, upper_limits = self._get_arm_joint_limits()

        # Get the current joints situation
        current_arm_q = q_start[arm_indices].copy()

        # Some helper values to compute useful seeds
        joint_center = (lower_limits + upper_limits) / 2.0
        joint_range = upper_limits - lower_limits

        # (We define these safe values, to avoid hitting the real joints' limits)
        safety_margin = 0.05 * joint_range
        safe_lower = lower_limits + safety_margin
        safe_upper = upper_limits - safety_margin

        seeds = []

        # Actual current configuration.
        seeds.append(q_start.copy())

        # Halfway between current and joint center.
        half_seed = q_start.copy()
        half_seed[arm_indices] = 0.5 * current_arm_q + 0.5 * joint_center
        seeds.append(half_seed)

        # Joint-center configuration.
        center_seed = q_start.copy()
        center_seed[arm_indices] = joint_center
        seeds.append(center_seed)

        # Mirrored configuration around the joint centers.
        mirror_seed = q_start.copy()
        mirror_arm_q = 2.0 * joint_center - current_arm_q
        mirror_seed[arm_indices] = np.clip(mirror_arm_q, safe_lower, safe_upper)
        seeds.append(mirror_seed)

        # Deterministic random generator, so the experiment is repeatable.
        rng = np.random.default_rng(0)

        # Local seeds around the current robot pose.
        for _ in range(n_local_seeds):
            seed = q_start.copy()

            # We randomly perturbate, so every seed is really random
            perturbation = rng.uniform(-0.20, 0.20, size=len(arm_indices)) * joint_range

            arm_q = current_arm_q + perturbation
            # Normalize the joints' positions between the safe lower and upper limits we pre-defined
            seed[arm_indices] = np.clip(arm_q, safe_lower, safe_upper)

            seeds.append(seed)

        # Global seeds distributed across the valid joint space.
        for _ in range(n_global_seeds):
            seed = q_start.copy()

            seed[arm_indices] = rng.uniform(
                safe_lower,
                safe_upper
            )

            seeds.append(seed)

        return seeds

    def _generate_preview_seeds(self, q_start, n_local_seeds=settings.preview_local_seeds):
        """Generate a small set of local seeds for fast IK preview.

        Unlike the robust multi-start strategy, this function assumes that
        the target pose is close to the current / hypothetical robot pose.

        Args:
            q_start (np.ndarray): Complete MuJoCo starting configuration.
            n_local_seeds (int): Number of additional local seeds.

        Returns:
            list: Local seeds used for fast IK preview.
        """

        # Retrieve the arm joint indices.
        arm_indices = self.get_arm_qpos_indices()

        # Retrieve the arm joint limits.
        lower_limits, upper_limits = self._get_arm_joint_limits()

        current_arm_q = q_start[arm_indices].copy()

        joint_range = upper_limits - lower_limits

        # Keep a small safety margin from the physical joint limits.
        safety_margin = 0.05 * joint_range
        safe_lower = lower_limits + safety_margin
        safe_upper = upper_limits - safety_margin

        seeds = []

        # The most important seed is exactly the current / hypothetical
        # robot configuration.
        seeds.append(q_start.copy())

        # Deterministic generator so results remain reproducible.
        rng = np.random.default_rng(1)

        # Add only a few SMALL local perturbations.
        for _ in range(n_local_seeds):

            seed = q_start.copy()

            perturbation = rng.uniform(-settings.preview_seed_perturbation, settings.preview_seed_perturbation, size=len(arm_indices)) * joint_range

            seed[arm_indices] = np.clip(current_arm_q + perturbation, safe_lower, safe_upper)

            seeds.append(seed)

        return seeds

    def _score_solution(self, q_start, q_solution):
        """This function returns a score for the given solution. In particular it check how far it is from 
        the current robot posture and how far from their limits the joints will be after the transformation.

        Args:
            q_start (np.ndarray): The starting joints' positions.
            q_solution (np.ndarray): The solution proposed joints' positions

        Returns:
            float, float: The first is the score computed on the solution, the second is the lowest margin of each joint.
        """

        # Get the indices corresponding to the robot joints
        arm_indices = self.get_arm_qpos_indices()
        lower_limits, upper_limits = self._get_arm_joint_limits()

        # Get the current joints' positions and the proposed solutions ones
        current_q = q_start[arm_indices]
        solution_q = q_solution[arm_indices]

        # Compute the totale range of movement of every joint
        joint_range = upper_limits - lower_limits

        # Prefer solutions close to the actual robot configuration.
        normalized_delta = (solution_q - current_q) / joint_range

        distance_score = np.linalg.norm(normalized_delta)
        max_joint_movement = np.max(np.abs(normalized_delta))

        # Prefer configurations far from the joint limits.
        lower_margin = (solution_q - lower_limits) / joint_range
        upper_margin = (upper_limits - solution_q) / joint_range
        normalized_margin = np.minimum(lower_margin, upper_margin)

        # Only penalize joints that are in the outer 8% of their range.
        limit_penalty = np.sum(np.maximum(0.0, 0.08 - normalized_margin) ** 2) * 50.0

        score = distance_score + 0.5 * max_joint_movement + limit_penalty

        return score, np.min(normalized_margin)

    def solve_target_pose(self, pose, q_start_arm=None, preview=False,
                      dt=0.01, max_iterations=500,
                      pos_tolerance=0.005,
                      rot_tolerance=0.0175): # The rot tolerance is in rad (0.0175 rad = 1.0 deg)

        # Always start from the complete REAL MuJoCo configuration.
        q_start = self.env.sim.data.qpos.copy()
        q_start = np.asarray(q_start, dtype=float).copy()

        arm_indices = self.get_arm_qpos_indices()

        # If an hypothetical arm configuration is provided, replace only
        # the arm joints inside the complete MuJoCo configuration.
        if q_start_arm is not None:

            q_start_arm = np.asarray(q_start_arm, dtype=float).reshape(-1)

            if q_start_arm.size != len(arm_indices):
                raise ValueError(
                    "q_start_arm contains a different number of values "
                    "than the number of robot arm joints!"
                )

            q_start[arm_indices] = q_start_arm

        # The posture preference corresponds to the actual starting
        # configuration used for this IK problem.
        self.configuration.update(q_start)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Both tasks share the same Cartesian target.
        self.position_task.set_target(pose)
        self.eef_task.set_target(pose)

        # -------------------------------------------------------------
        # Choose between robust global IK and fast local preview IK.
        # -------------------------------------------------------------
        if preview:
            seeds = self._generate_preview_seeds(q_start)

            position_iterations = settings.preview_position_iterations
            pose_iterations = settings.preview_pose_iterations

        else:
            seeds = self._generate_seeds(q_start)

            position_iterations = 150
            pose_iterations = max_iterations

        valid_solutions = []

        # Keep track of the best failed solution for diagnostic purposes.
        best_failed_seed = None
        best_failed_position_error = np.inf
        best_failed_rotation_error = np.inf

        for seed_index, seed in enumerate(seeds):

            # Update the configuration with the current seed.
            self.configuration.update(seed)

            # ---------------------------------------------------------
            # STAGE 1: position-only convergence.
            # ---------------------------------------------------------
            for _ in range(position_iterations):

                vel = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=[self.position_task, self.posture_task],
                    dt=dt,
                    solver="daqp",
                    limits=self.joint_limits
                )

                self.configuration.integrate_inplace(vel, dt)

                position_error_stage1 = np.linalg.norm(
                    self.position_task.compute_error(self.configuration)[:3]
                )

                if position_error_stage1 < 0.02:
                    break

            # ---------------------------------------------------------
            # STAGE 2: complete position + orientation convergence.
            # ---------------------------------------------------------
            converged = False

            for _ in range(pose_iterations):

                vel = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=[self.eef_task, self.posture_task],
                    dt=dt,
                    solver="daqp",
                    limits=self.joint_limits
                )

                self.configuration.integrate_inplace(vel, dt)

                error = self.eef_task.compute_error(self.configuration)

                position_error = np.linalg.norm(error[:3])
                rotation_error = np.linalg.norm(error[3:])

                if position_error < pos_tolerance and rotation_error < rot_tolerance:
                    converged = True
                    break

            if not converged:

                if position_error < best_failed_position_error:
                    best_failed_seed = seed_index + 1
                    best_failed_position_error = position_error
                    best_failed_rotation_error = rotation_error

                continue

            # If we found a solution, retrieve it.
            solution = self.configuration.q.copy()

            # Check that the solution respects the arm joint limits.
            lower_limits, upper_limits = self._get_arm_joint_limits()
            solution_arm_q = solution[arm_indices]

            if np.any(solution_arm_q < lower_limits) or np.any(solution_arm_q > upper_limits):
                continue

            # Score the found solution.
            score, min_margin = self._score_solution(q_start, solution)

            valid_solutions.append(
                (
                    score,
                    solution,
                    position_error,
                    rotation_error,
                    min_margin
                )
            )

        # No valid IK solution found.
        if len(valid_solutions) == 0:
            return None

        # Prefer accurate Cartesian solutions before applying the
        # joint-space score.
        preferred_position_tolerance = 0.002 # 2 mm

        precise_solutions = [
            candidate
            for candidate in valid_solutions
            if candidate[2] <= preferred_position_tolerance
        ]

        if len(precise_solutions) > 0:
            candidate_solutions = precise_solutions
        else:
            candidate_solutions = valid_solutions

        # Choose the best remaining candidate according to the
        # existing joint-space score.
        candidate_solutions.sort(
            key=lambda candidate: candidate[0]
        )

        _, best_solution, _, _, _ = candidate_solutions[0]

        if best_solution is None:
            print("No valid IK solution... Robot will not move!")
            return None

        # Return only the 6 arm joints.
        q_final = best_solution[arm_indices].copy()

        return q_final

    def get_arm_qpos_indices(self):

        indices = []

        for joint_name in self.joint_names:

            joint_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )

            if joint_id == -1:
                raise ValueError(f"Joint name: {joint_name} not found in MJ Model!")

            qpos_index = self.mj_model.jnt_qposadr[joint_id]
            indices.append(qpos_index)

        return np.array(indices, dtype=int)

    def quat_mj_to_mat(self, quat_mj: np.array) -> np.array:
        # First convert the mj quaternion (w, x, y, z) to a robosuite quaternion (x, y, z, w)
        quat_mj = np.asarray(quat_mj, dtype=float)  # Ensure the quaternion is in the correct accepted type
        quat_robosuite = T.convert_quat(quat_mj, to="xyzw")

        # The convert the quaternion to a matrix
        quat_matrix = T.quat2mat(quat_robosuite)

        return quat_matrix

    def create_mink_target(self, pos: np.ndarray, quat_matrix: np.ndarray = None):

        # Define the transformation needed for mink
        # Add the offest of the gripper site!
        pos = np.asarray(pos + np.asarray(settings.safe_offset_gripper), dtype=float)
        target_position, r_matrix = self._define_target(pos, quat_matrix)

        # Create the mink target in a suitable format
        target_pose = self._make_mink_target(target_position, r_matrix)

        return target_pose