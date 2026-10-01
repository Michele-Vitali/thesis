import math

import numpy as np
import settings
from ik import IKController
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models import objects
from scipy.spatial.transform import Rotation as R


class MovementController:

    def __init__(self, env: SingleArmEnv, gripper_state: float = 1.0):
        self.env = env
        self.gripper_state = gripper_state
        self.steps = 0
        self.reward = 0.0
        self.q_joints = self.get_robot_joints_pos()
        self.MAX_JOINT_DELTA = 0.15
        self.ik_ctrl = IKController(self.env)

    def move_joints_to_pose(self, q_final: np.ndarray, max_steps: int = 150, joint_err_tol: float = 0.01) -> bool:

        if q_final is None:
            print("Cannot move robot. No solution had been found!")
            return False

        # Get the current joint positions
        q_current = self.get_robot_joints_pos()
        
        joint_target_reached = False

        for _ in range(max_steps):
            # Compute the joint command needed for the movement
            joint_command = (q_final - q_current)
            #Normalize it so we do not make too large movements
            joint_command = np.clip(joint_command / self.MAX_JOINT_DELTA, -1.0, +1.0)

            # Generate the action
            action = np.zeros(self.env.action_dim)
            action[:6] = joint_command
            action[6] = 0.0 # Hold the current state

            self.env.step(action)
            self.env.render()

            # Get the current joint positions
            q_current = self.get_robot_joints_pos()
            # Compute the error of each joint with respect to the desired pose
            joint_error = q_final - q_current

            # If the maximum error among the joint errors is below the tolerance we consider the task finished
            if np.max(np.abs(joint_error)) < joint_err_tol:
                print("Joint target reached!")
                joint_target_reached = True
                break     

        if not joint_target_reached:
            print("Joint target not reached!")

        return joint_target_reached

    def check_eef_pos(self, target_pos: np.ndarray, cartesian_err_tol: float = 0.005) -> bool:

        cartesian_position_reached = False

        obs = self.env._get_observations()
        # Remember to use .copy() otherwise we would get a view that will change overtime
        final_eef_pos = obs["robot0_eef_pos"].copy()

        # Check the error with the norm of the distance between the target and actual position
        cartesian_error = np.linalg.norm(target_pos - final_eef_pos)

        if cartesian_error < cartesian_err_tol:
            cartesian_position_reached = True

        return cartesian_position_reached

    def get_robot_joints_pos(self):
        joint_positions = self.env.robots[0]._joint_positions

        return joint_positions

    def complex_traslation(self, target_pos: np.ndarray, act_descr: str, stop_on_contact_geom_ids: list | None = None,
                            allowed_body_ids: list | None = None, max_steps: int = 200, tolerance: float = 0.005) -> None:
        """
        This functions makes the eef moves from its initial position to the target position.
        It uses the function move_on_single_axis to implement one-axis movements to avoid collisions.

        Args:
            env (SingleArmEnv): The simulation environment.
            target_pos (np.ndarray): The target position we want to reach.
            act_descr (str): The description of the current action.
            stop_on_contact_geom_ids (list, optional): The list of geometry IDs that should stop the movement if touched. Defaults to None.
            allowed_body_ids (list, optional): The list of body IDs that are allowed to be touched. Defaults to None.
            max_steps (int, optional): The maximum number of steps the moving should take. Defaults to 150.
            tolerance (float, optional): The tolerance we accept for the traslation error. Defaults to 0.005
        """

        # Ensure to use a copy of the passed position (so the original object doesn't get affected!)
        target_pos = np.asarray(target_pos, dtype=float).copy()

        obs = self._obs_init()
        new_target_pos = obs["robot0_eef_pos"].copy()

        current_steps = 0
        current_reward = 0.0

        # Implement one-axis movements to avoid collisions, we move on the z and then x, y axes separately.
        for i in range(target_pos.size):
            new_target_pos[target_pos.size - i - 1] = target_pos[target_pos.size - i - 1]
            steps, reward = self.simple_traslation(new_target_pos, act_descr, stop_on_contact_geom_ids, allowed_body_ids, max_steps, tolerance)
            current_steps += steps
            current_reward += reward

        self.steps += current_steps
        self.reward += current_reward

        # print(f"Action: {act_descr}, Target reached within {current_steps} steps. Reward: {current_reward: .4f}!")


    def simple_traslation(self, target_pos: np.ndarray, act_descr: str,
                            stop_on_contact_geom_ids: list | None = None, allowed_body_ids: list | None = None,
                            max_steps: int = 200, tolerance: float = 0.005) -> tuple[int, float]:
        """
        Moves the eef from its initial position to the target position.

        Args:
            env (SingleArmEnv): The simulation environment.
            target_pos (np.ndarray): The array representing the target position
            act_descr (str): The description of the current action
            stop_on_contact_geom_ids (list, optional): The list of geometry IDs that should stop the movement if touched. Defaults to None.
            allowed_body_ids (list, optional): The list of body IDs that are allowed to be touched. Defaults to None.
            max_steps (int, optional): The maximum number of step the algorithm should take, othwerwirse it would go on indefinitely.
                Defaults to 150.
            tolerance (float, optional): The tolerance we can accept for positioning. Defaults to 0.005.
        """

        # Fake action just to initalize obs.
        action = np.zeros(self.env.action_dim)
        action[-1] = self.gripper_state
        obs, _, _, _ = self.env.step(action)

        total_steps = 0
        total_reward = 0.0

        # Now we start to move.
        for _ in range(max_steps):
            # eef stands for End-Effector, which in our case is the gripper!
            # 1. Take the gripper position.
            eef_pos = (0,0,0) if not obs else obs["robot0_eef_pos"]

            # 2. Compute the distance.
            delta_pos = target_pos - eef_pos

            # np.linalg.norm just computes the euclidean distance between target_pos and eef_pos.
            distance = np.linalg.norm(delta_pos)

            if distance < tolerance:
                # We reached our target.
                break

            # 3.a If not reached the target, compute proportional action
            action[0:3] = np.clip(settings.translation_k * delta_pos, -0.7, 0.7) # We proportionally move, clipped to abs 0.7 to not move at max speed
            #action[3:6] = 0.0  Already zero as per definition in the start
            #action[-1] = _get_grabber_state(env)

            # 4. Execute
            obs, reward, _, _ = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

            total_reward += reward
            total_steps += 1

            """
            Check for any unexpected contact, if yes stop!
            If no body is allowed we pass an empty set()
            if stop_on_contact_geom_ids is not None and self._unexpected_contact(stop_on_contact_geom_ids, allowed_body_ids or set()):
                print("Touched an unexpected body, halt the movement!")
                return
            """

        # Eventually after 'max_steps' steps the loop finishes...
        # print(f"Action: {act_descr}, Target not reached even in {total_steps} steps. Reward: {total_reward: .4f}")

        return (total_steps, total_reward)


    def _obs_init(self) -> dict:
        """Initializes the observation dictionary.

        Args:
            env (SingleArmEnv): The simulation environment

        Returns:
            dict: The initialized observation dictionary
        """

        # Fake action to initialize the obs dict
        action = np.zeros(self.env.action_dim)

        # Take the gripper state so that we do not release objects
        action[-1] = self.gripper_state

        obs, _, _, _ = self.env.step(action)

        return obs


    def determine_n_faces(self, obj, square_tolerance: float = 0.1) -> int:
        """
        Decide quante orientazioni di presa sono davvero equivalenti per questo oggetto,
        guardando le sue dimensioni orizzontali (x, y).

        Un oggetto con base quadrata ha 4 facce equivalenti (ogni 90°): ruotare di 90°
        porta comunque a una presa valida.
        Un oggetto con base rettangolare (non quadrata) ha solo 2 orientazioni valide
        (ogni 180°): ruotare di 90° significherebbe afferrarlo lungo il lato sbagliato.

        Args:
            obj: L'oggetto BoxObject di cui valutare la forma.
            square_tolerance: differenza relativa massima tra i due lati per considerarli "uguali".

        Returns:
            4 se la base è quadrata, 2 altrimenti.
        """

        if isinstance(obj, objects.BoxObject):
            # Get the sizes of the object in both dimensions
            size_x = obj.size[0]
            size_y = obj.size[1]

            # Get which side is the biggest
            larger_side = max(size_x, size_y)

            # Compute a metric to measure how much the object differs from a cube...
            difference = abs(size_x - size_y)
            relative_difference = difference / larger_side

            if relative_difference <= square_tolerance:
                return 4 # Considered as cube
            else:
                return 2 # Considered as rectangle
        else:
            return 100  # Point of grasp is not relevant


    def optimal_eef_rotation(self, obj_quat: np.ndarray, eef_quat: np.ndarray, n_faces: int = 4) -> R:
        """
        Return the optimal rotation that the eef needs to reach for grabbing
        the passed object.

        Args:
            env: The current simulation environment.
            obj_quat: The quaternion of the object on which we want to compute
                    the optimal eef rotation for grabbing it (w, x, y, z).
            eef_quat: The quaternion of the eef.
            n_faces: **(Optional)** The number of faces the geometry has.

        Returns:
            The optimal rotations the eef needs to execute
        """

        # Grab the object yaw
        object_rotation = R.from_quat(obj_quat, scalar_first=True)
        object_yaw = object_rotation.as_euler('zyx')[0]

        # Compute the faces' angular width
        face_width = (2 * np.pi) / n_faces
        #half_face_width = face_width / 2.0

        # Find the yaw of the nearest face
        #shortest_yaw = (object_yaw + half_face_width) % face_width - half_face_width

        # Compute the needed rotation
        #yaw_rotation = R.from_euler('z', shortest_yaw)
        downward_rotation = R.from_euler('y', np.pi)    # To ensure the gripper is perpendicular to the table
        #target_rotation = yaw_rotation * downward_rotation

        # This quat is already in the [x ,y, z, w] convention
        eef_rotation = R.from_quat(eef_quat)

        best_rotation = None
        best_angle = None

        # Loop through all object's faces
        for face_index in range(n_faces):
            # Compute the yaw of every face
            face_yaw = object_yaw + face_index * face_width

            # Check whether it is better
            for gripper_flip in (0.0, np.pi):
                # Compute the rotation needed to reach that face's yaw
                candidate_yaw = face_yaw + gripper_flip
                candidate_rotation = R.from_euler('z', candidate_yaw) * downward_rotation

                # Compute the difference between the needed rotation and actual eef rotation
                difference = candidate_rotation * eef_rotation.inv()
                angle_needed = difference.magnitude()

                if best_angle is None or angle_needed < best_angle:
                    best_angle = angle_needed
                    best_rotation = candidate_rotation

        return best_rotation


    def yaw_rotation(self, object_quat: np.ndarray, act_descr: str, n_faces: int = 4) -> None:
        """Executes a gripper yaw-rotation (around the z-axis) reaching the target rotation passed.

        Args:
            env (SingleArmEnv): The current simulation environment
            object_quat (np.ndarray): The quaternion of the object passed (w, x, y, z)
            act_descr (str): The description of the current action
            n_faces (int, optional): The number of faces of the object. Defaults to 4.
        """

        # Initialize the obs dictionary
        obs = self._obs_init()
        eef_quat = obs["robot0_eef_quat"]   # [x, y, z, w]

        # Compute the optimal eef rotation
        target_rotation = self.optimal_eef_rotation(object_quat, eef_quat, n_faces)

        # Execute the rotation
        current_steps, current_reward = self.rotation(obs, target_rotation, act_descr)

        self.steps += current_steps
        self.reward += current_reward


    def _delta_rotation(self, obs: dict, target_rotation: R) -> np.ndarray:
        """
        Computes the rotation the eef needs to make to reach the passed target rotation.
        The value returned is the delta between the current eef rotation and the target rotation.

        Args:
            obs (dict): The observation dictionary
            target_rotation (scipy.spatial.transform.Rotation): The rotation the eef should reach

        Returns:
            The rotational difference the eef needs to execute to reach the target rotation
        """

        # First grab the eef rotation
        eef_quat = obs["robot0_eef_quat"]
        eef_rotation = R.from_quat(eef_quat)

        # Compute the rotation difference between the scipy Rotation objects (via matrixes)
        delta_rotation = target_rotation * eef_rotation.inv()

        # Convert the difference in the rotational vector required by the OSC_POSE controller
        delta_rotvec = delta_rotation.as_rotvec()

        return delta_rotvec


    def rotation(self, obs: dict, target_rot: np.ndarray, act_descr: str, max_steps: int = 200, tolerance: float = 1.0) -> tuple[int, float]:
        """Bring the eef from its initial rotation to the target rotation passed.

        Args:
            env (SingleArmEnv): The current simulation environment
            obs (dict): The observation dictionary
            target_rot (np.ndarray): The target rotation we want to achieve
            act_descr (str): The description of the current action
            max_steps (int, optional): The maximum steps the algorithm should take, otherwise it would go on indefinitely. Defaults to 150.
            tolerance (float, optional): The tolerance we can accept for the alignment (in degrees). Defaults to 1.0.
        """

        # Grab the eef initial position for active correction
        init_eef_pos = obs["robot0_eef_pos"].copy()

        # Initialize the action
        action = np.zeros(self.env.action_dim)

        total_steps = 0
        total_reward = 0.0

        for _ in range(max_steps):
            delta_rotvec = self._delta_rotation(obs, target_rot)

            # Compute the magnitude of the error from the final rotation
            error_magnitude = np.linalg.norm(delta_rotvec)
            degree_difference = (error_magnitude * 180) / np.pi

            # Note: We put abs(...) because the difference in rotation can also be negative!
            if degree_difference < tolerance:
                # print(f"Action: {act_descr}, Aligned within {total_steps} steps! Final difference: {degree_difference:.2f}°. Reward: {total_reward: .4f}")
                return (total_steps, total_reward)

            # Introducing active correction...
            current_eef_pos = obs["robot0_eef_pos"]
            delta_pos = init_eef_pos - current_eef_pos

            # Generate the action
            current_rot_limit = settings.max_rot_speed * (total_steps + 1) / settings.rotation_ramp_steps
            current_rot_limit = min(current_rot_limit, settings.max_rot_speed)

            action[0:3] = np.clip(settings.stationary_k * delta_pos, -0.05, 0.05)
            action[3:6] = np.clip(settings.rotation_k * delta_rotvec, - current_rot_limit, current_rot_limit)
            action[-1] = self.gripper_state

            # Execute
            obs, reward, _, _ = self.env.step(action)
            if self.env.has_renderer:
                self.env.render()

            total_reward += reward
            total_steps += 1

        # Eventually after 'max_steps' steps the loop finishes...
        # print(f"Action: {act_descr}, Target not reached even in {total_steps} steps...")

        return (total_steps, total_reward)


    def toggle_grab(self, min_steps: int = 50):
        """Toggles the current gripper's grab state.

        If it is the first time we call this function we use init_grab to ensure
        the eef stays open for grabbing procedure.

        Args:
            env (SingleArmEnv): The simulation environment.
            min_steps (int, optional): The minimum steps we run the opening or closing action to
                before going on with the next movements.. Defaults to 30.
        """

        # Grab the current gripper state from the controller's internal state variable...
        current_state = self.gripper_state

        # Invert it!
        self.gripper_state = - current_state

        # Initialize the action, the first 6 elements stay at 0 (meaning no movements)
        action = np.zeros(self.env.action_dim)
        
        # Set the 7th to the wanted gripper state.
        action[6] = self.gripper_state

        # Apply the action for at least 'min_steps' to ensure the grip are completely closed/open
        for _ in range(min_steps):
            self.env.step(action)
            self.env.render()

    def init_robot_pose(self):
        self.move_joints_to_pose(np.asarray(settings.starting_pose, dtype=float))

    def build_pose(self, pos: np.array, quat: np.array = None) -> SE3:
        """Build a robot pose for reaching the specified position and rotation.

        Args:
            pos (np.ndarray): The position we want to reach.
            quat (np.ndarray, optional): The rotation we wanto to reach. Defaults to None.

        Returns:
            SE3: The trasnformation needed to reach the wanted position and rotation.
        """

        quat_matrix = None
        if quat is not None:
            # Convert the quaternion to robosuite convention (x, y, z, w)
            quat_matrix = self.ik_ctrl.quat_mj_to_mat(quat)
    
        target_pose = self.ik_ctrl.create_mink_target(pos, quat_matrix)
        
        return target_pose
    
    def move_robot_to_position(self, pos: np.array, quat: np.array):

        target_pose = self.build_pose(pos, quat)
        
        # Solve the IK problem
        q_solution = self.ik_ctrl.solve_target_pose(target_pose)
    
        done = self.move_joints_to_pose(q_solution)

        return done

    def elevate_obj(self, pos: np.ndarray, quat: np.ndarray = None, elevation: float = 0.10): #10 cm
    
        target_z = pos[2] + elevation

        needed_steps = math.ceil(settings.desired_elevation / settings.elevation_step)
        max_retries = settings.elevation_max_retries

        current_pos = pos.copy()

        for _ in range(needed_steps):

            solution_found = False
            n_retry = 0
            perturbation_strength = 0
            # Compute the next quote we want to reach...
            next_z = min(pos[2] + settings.elevation_step, target_z)
            perturbated_pos = pos.copy()
            q_solution = []

            while not solution_found and n_retry < max_retries + 1: # +1 Since we need to include the first legit try with no perturbation
                # Build the perturbated pose
                noise = np.random.uniform(0.0, 1.0, size=pos.shape - 1) # Uniformly pick a vlaue between 0.0 and 1.0 for the xy values
                perturbated_pos = pos[:3] + (perturbation_strength * noise)    # Perturbate the position
                print(f"Perturbated position: {perturbated_pos}")
                target_pose = self.build_pose(perturbated_pos, quat)

                current_q = self.get_robot_joints_pos()
                # If we really perturbated the initial position...
                if n_retry != 0:
                    partial_solution = self._check_for_solution(target_pose)

                    if partial_solution is not None:
                        # First move to the new xy location.
                        #self.move_joints_to_pose(partial_solution)
                        current_q = partial_solution
                        target_pose = self.build_pose(perturbated_pos)
                    else:
                        # If we cannot move to that specific xy, pass to the next perturbation try
                        continue 


                if not solution_found:
                    n_retry += 1
                    perturbation_strength += 0.01

            # If no perturbation was needed, just move.
            if solution_found and n_retry == 0:
                self.move_joints_to_pose(q_solution)
            elif solution_found and n_retry > 0:
                # We first move to the perturbated xy position and then elevate
                continue

    def _check_for_solution(self, pose, q_start = None):

        # Check if there is a valid IK solution for the current xy configuration
        q_solution = self.ik_ctrl.solve_target_pose(pose, q_start)
        if q_solution is not None:
            print("A solution for the position exists!")

            # Retrieve the real joint limits.
            lower_limits, upper_limits = self.ik_ctrl._get_arm_joint_limits()

            # Compute the normalized distance of each joint from its closest limit.
            joint_range = upper_limits - lower_limits
            lower_margin = (q_solution - lower_limits) / joint_range
            upper_margin = (upper_limits - q_solution) / joint_range
            joint_margins = np.minimum(lower_margin, upper_margin)

            min_joint_margin = np.min(joint_margins)

            print("Joint margins: ")
            for i in range(len(q_solution)):
                print(f"\t[Joint {i+1}] {joint_margins[i]}")

            print(f"Minimum joint margin: {min_joint_margin}")

            if min_joint_margin < 1e-3: # If we have a maring less than 0.001 we are in a critic situation
                print("The solution had a joint margin too low!")
                return None
            else:
                print("A correct solution was found!")
                return q_solution
        else:
            print("No solution found for the position!")
            return None