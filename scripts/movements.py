import numpy as np
from scipy.spatial.transform import Rotation as R
import settings
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

def move_to_target(env: "SingleArmEnv", target_pos: np.ndarray, str: str, max_steps: int = 150, tolerance: float = 0.005):
    """Moves the eef from its initial position to the target position.

    Args:
        env (SingleArmEnv): The simulation environment.
        target_pos (np.ndarray): The array representing the target position
        str (str): The description of the current action
        max_steps (int, optional): The maximum number of step the algorithm should take, othwerwirse it would go on indefinitely. 
            Defaults to 150.
        tolerance (float, optional): The tolerance we can accept for positioning. Defaults to 0.005.
    """
    
    # Fake action just to initalize obs.
    action = np.zeros(env.action_dim)
    action[-1] = _get_grabber_state(env)
    obs, reward, done, info = env.step(action)

    steps = 0

    # Now we start to move.
    for step in range(max_steps):
        # eef stands for End-Effector, which in our case is the gripper!
        # 1. Take the gripper position.
        eef_pos = (0,0,0) if not obs else obs["robot0_eef_pos"]

        # 2. Compute the distance.
        delta_pos = target_pos - eef_pos

        # np.linalg.norm just computes the euclidean distance between target_pos and eef_pos.
        distance = np.linalg.norm(delta_pos)

        if distance < tolerance:
            # We reached our target.
            print(f"Action: {str}, Target reached within {steps} steps! Final distance: {distance:.2f}cm")
            return 

        # 3.a If not reached the target, compute proportional action
        action[0:3] = np.clip(settings.translation_k * delta_pos, -0.7, 0.7) # We proportionally move, clipped to abs 0.7 to not move at max speed
        #action[3:6] = 0.0  Already zero as per definition in the start
        action[-1] = _get_grabber_state(env)

        # 4. Execute
        obs, reward, done, info = env.step(action)
        env.render()

        steps += 1

    # Eventually after 'max_steps' steps the loop finishes...
    print(f"Action: {str}, Target not reached even in {steps} steps...")

def _get_grabber_state(env: "SingleArmEnv") -> float:
    """Retrieves the current gripper state (open or closed).

    Args:
        env (SingleArmEnv): The simulation environment.

    Returns:
        (float): -1.0 if the gripper is open or 1.0 if the gripper is closed
    """
    # Take the list of all objects in the env
    all_objects = []
    for obj in env.objects:
        if isinstance(obj.contact_geoms, list):
            all_objects.extend(obj.contact_geoms)
        else:
            all_objects.append(obj.contact_geoms)

    # Get the current gripper state
    grabbing = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=all_objects)
    return 1.0 if grabbing else -1.0

def _obs_init(env: "SingleArmEnv") -> dict:
    """Initializes the observation dictionary.

    Args:
        env (SingleArmEnv): The simulation environment

    Returns:
        dict: The initialized observation dictionary
    """
    # Fake action to initialize the obs dict
    action = np.zeros(env.action_dim)
    # Take the gripper state so that we do not release objects
    action[-1] = _get_grabber_state(env)
    obs, reward, done, info = env.step(action)

    return obs

def optimal_eef_rotation(env: SingleArmEnv, obj_quat: np.ndarray, n_faces: int = 4) -> R:
    """
    Return the optimal rotation that the eef needs to reach for grabbing
    the passed object.

    Args:
        env: The current simulation environment.
        obj_quat: The quaternion of the object on which we want to compute 
                the optimal eef rotation for grabbing it (w, x, y, z).
        n_faces: **(Optional)** The nuumber of faces of the object.

    Returns: 
        The optimal rotations the eef needs to execute
    """

    # Grab the object yaw
    object_rotation = R.from_quat(obj_quat, scalar_first=True)
    object_yaw = object_rotation.as_euler('zyx')[0]

    # Compute the faces' angular width
    face_width = (2 * np.pi) / n_faces
    half_face_width = face_width / 2.0

    # Find the yaw of the nearest face
    shortest_yaw = (object_yaw + half_face_width) % face_width

    # Compute the needed rotation
    yaw_rotation = R.from_euler('z', shortest_yaw)
    downward_rotation = R.from_euler('y', np.pi)    # To ensure the gripper is perpendicular to the table
    target_rotation = yaw_rotation * downward_rotation

    return target_rotation

def yaw_rotation(env, object_quat, str, n_faces=4):
    """Executes a gripper yaw-rotation (around the z-axis) reaching the target rotation passed.

    Args:
        env (SingleArmEnv): The simulation environment
        object_quat (np.ndarray): The quaternion of the object passed (w, x, y, z)
        str (str): The description of the current action
        n_faces (int, optional): The number of faces of the object. Defaults to 4.
    """

    #MAX_FACES = 20  # Number of faces after which we consider the point of grabbing not relevant

    if True:#n_faces <= MAX_FACES:
        # Initialize the obs dictionary
        obs = _obs_init(env)
        # Compute the optimal eef rotation
        target_rotation = optimal_eef_rotation(env, object_quat)
        # Execute the rotation
        rotation(env, obs, target_rotation, str, n_faces)

def _delta_rotation(obs: dict, target_rotation: R) -> np.ndarray:
    """
    Computes the rotation the eef needs to make to reach the passed target rotation.
    The value returned is the delta between the current eef rotation and the target rotation.

    Args:
        obs (dict): The simulation observation dictionary
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

def rotation(env, obs, target_rot, str, max_steps=150, tolerance=1.0):
    """Bring the eef from its initial rotation to the target rotation passed.

    Args:
        env (SingleArmEnv): The simulation environment
        obs (dict): The observation dictionary
        target_rot (np.ndarray): The target rotation we want to achieve
        str (str): The description of the current action
        max_steps (int, optional): The maximum steps the algorithm should take, otherwise it would go on indefinitely. Defaults to 150.
        tolerance (float, optional): The tolerance we can accept for the alignment (in degrees). Defaults to 1.0.
    """
    # Grab the eef initial position for active correction
    init_eef_pos = obs["robot0_eef_pos"]

    # Initialize the action
    action = np.zeros(env.action_dim)

    steps = 0

    for step in range(max_steps):
        delta_rotvec = _delta_rotation(obs, target_rot)

        # Compute the magnitude of the error from the final rotation
        error_magnitude = np.linalg.norm(delta_rotvec)
        degree_difference = (error_magnitude * 180) / np.pi

        if steps % 10 == 0:
            print(f"Action: {str}, Step: {steps}, Degree difference: {degree_difference:.1f}")

        # Note: We put abs(...) because the difference in rotation can also be negative!
        if degree_difference < tolerance:
            print(f"Action: {str}, Aligned within {steps} steps! Final difference: {degree_difference:.2f}°")
            return

        # Introducing active correction...
        current_eef_pos = obs["robot0_eef_pos"]
        delta_pos = init_eef_pos - current_eef_pos

        # Generate the action
        action[0:3] = np.clip(settings.stationary_k * delta_pos, -0.05, 0.05)
        action[3:6] = np.clip(settings.rotation_k * delta_rotvec, -0.3, 0.3)
        action[-1] = _get_grabber_state(env)

        # Execute
        obs, reward, done, info = env.step(action)
        env.render()
        
        steps += 1
        
    # Eventually after 'max_steps' steps the loop finishes...
    print(f"Action: {str}, Target not reached even in {steps} steps...")
        
def toggle_grab(env: "SingleArmEnv", init_grab: bool = False, min_steps: int = 30):
    """Toggles the current gripper's grab state.

    If it is the first time we call this function we use init_grab to ensure 
    the eef stays open for grabbing procedure.

    Args:
        env (SingleArmEnv): The simulation environment.
        init_grab (bool, optional): Whether it's the first time we call the function or not.. Defaults to False.
        min_steps (int, optional): The minimum steps we run the opening or closing action to 
            before going on with the next movements.. Defaults to 30.
    """
    # Initialize the obs dictionary
    obs = _obs_init(env)

    # Grab the initial eef position
    init_eef_pos = obs["robot0_eef_pos"]

    # Initialize the action
    action = np.zeros(env.action_dim)
    action[-1] = _get_grabber_state(env)
    # We first run some steps to ensure enough time has passed from the previouse stages
    for i in range(settings.hold_steps):
        env.step(action)
        env.render()

    # Decide the gripper value and keep it for the entire loop
    if init_grab:
        action[-1] = - 1.0
    else:
        action[-1] = - (_get_grabber_state(env))

    # Keep closing for 'min_steps' otherwise the gripper won't completely close itself and miss the object.
    for i in range(min_steps):
        # Active correction
        current_eef_pos = obs["robot0_eef_pos"]
        delta_pos = init_eef_pos - current_eef_pos

        action[0:3] = np.clip(settings.stationary_k * delta_pos, -0.05, 0.05)
        #action[3:6] = 0.0   Already zero as per definition

        obs, reward, done, info = env.step(action)
        env.render()