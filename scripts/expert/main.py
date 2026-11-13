# Import our custom env and robot
import scripts.expert.custom_env as custom_env  # noqa: F401
import scripts.expert.custom_robot as custom_robot  # noqa: F401

# Import some libraries
import mujoco
import numpy as np
import robosuite as suite

# Import the settings file
import scripts.expert.settings as settings
from scripts.expert.custom_osc import install_prob3_custom_osc
from scripts.dataset_utils.hdf5_recorder import HDF5Recorder, NullRecorder

# Import useful classes
from scripts.expert.homing import prepare_episode
from scripts.expert.movements import OSCMovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper
import random

from scipy.stats import qmc
from scipy.spatial.transform import Rotation as R

current_seed = None

def set_global_seed(seed: int):
    # First set the randomization seed for reproducibility!
    random.seed(seed)
    np.random.seed(seed)
    global current_seed
    current_seed = seed

def get_global_seed() -> int:
    if current_seed is None:
        raise ValueError("Global seed not initialized yet!")
    else:
        return current_seed

def generate_sobol_conditions(n_samples: int, seed: int) ->list[dict]:

    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    # Potenza di 2 abbastanza ampia da compensare gli scarti nella zona centrale vietata
    m = max(12, int(np.ceil(np.log2(max(n_samples * 12, 1)))))
    raw_u = sampler.random_base2(m=m)

    table_half_width = settings.table_width / 2.0
    table_top_z = settings.table_height + 0.025  # 0.05 / 2 in TableArena
    robot_xy = np.array([-settings.rob_x, 0.0])
    #forbidden_xy = np.array(settings.forbidden_spawning_pos, dtype=float)

    conditions = []
    for u in raw_u:
        # 1. Dimensioni 3D (half-sizes)
        half_sizes = settings.object_half_size_min + u[3:6] * (
            settings.object_half_size_max - settings.object_half_size_min
        )
        obj_radius = float(np.hypot(half_sizes[0], half_sizes[1]))

        # 2. Posizione planare (x, y) dentro table_bounds con margine per ingombro oggetto
        x = -table_half_width + u[0] * (2.0 * table_half_width)
        y = -table_half_width + u[1] * (2.0 * table_half_width)

        if abs(x) + obj_radius > table_half_width or abs(y) + obj_radius > table_half_width:
            continue

        xy = np.array([x, y])

        # Vincolo anello radiale raggiungibile dal robot
        dist_robot = float(np.linalg.norm(xy - robot_xy))
        if not (settings.radial_range[0] + obj_radius <= dist_robot <= settings.radial_range[1] - obj_radius):
            continue

        # Vincolo zona centrale vietata (drop point)
        #dist_drop = float(np.linalg.norm(xy - forbidden_xy))
        #if dist_drop < settings.forbidden_spawning_radius + obj_radius:
            #continue

        # 3. Orientamento yaw [-pi, pi] -> quaternione MuJoCo [w, x, y, z]
        yaw = -np.pi + u[2] * (2.0 * np.pi)
        quat_wxyz = R.from_euler("z", yaw).as_quat(scalar_first=True)

        conditions.append({
            "init_obj_size": half_sizes.tolist(),
            "init_obj_pos": [float(x), float(y), float(table_top_z + half_sizes[2])],
            "init_obj_quat": quat_wxyz.tolist(),
        })

        if len(conditions) == n_samples:
            break

    if len(conditions) < n_samples:
        raise RuntimeError(f"Generate solo {len(conditions)}/{n_samples} condizioni valide.")

    # Mescola l'ordine di presentazione senza alterare la copertura dello spazio 6D
    rng = np.random.default_rng(seed)
    rng.shuffle(conditions)
    return conditions

def main():
    target_seed = settings.base_env_rand_collect_seed
    set_global_seed(target_seed)

    # Load our custom OSC_POSE controller!
    install_prob3_custom_osc()

    # Load the controllers
    joint_config = load_controller_config(default_controller="JOINT_POSITION")
    osc_config = load_controller_config(default_controller="OSC_POSE")

    # Update the config with the OSC controller configurations from the settings file...
    osc_config.update(settings.osc_config)

    # Initialize the HDF5 recorder and set the recording to the value in the settings file
    record = bool(settings.record)
    recorder = HDF5Recorder() if record else NullRecorder()

    # Create the env
    

    try:
        task_loop = True
        while task_loop:
            # Ask how many episodes we want to simulate
            n_tasks = int(input("How many times do you want to simulate the OSC task? "))

            # Variables for debugging and reporting
            n_simulated_tasks = 0
            successes = 0   # Also works as incremental ID of tasks...
            cond_id = 0
            retries = 0
            max_retries_per_condition = 2
            total_retries = 0
            results = []

            pool_size = int(np.ceil(n_tasks * 1.20))
            sobol_conditions = generate_sobol_conditions(pool_size, target_seed)

            # Loop to ensure we get N successes (and not N simulations, as failures should not be included or the model will learn on bad episodes...)
            while successes < n_tasks:
                if cond_id >= len(sobol_conditions):
                    sobol_conditions.extend(generate_sobol_conditions(pool_size, target_seed + n_simulated_tasks))

                current_seed = target_seed + n_simulated_tasks
                set_global_seed(current_seed)
                env = create_randomized_env(joint_config, sobol_conditions[cond_id])
                prepare_episode(env, joint_config, osc_config)
                #print(type(env.robots[0].controller))
                configure_task(env)
                (error, steps) = execute_task(successes, env, recorder, record)
                success = check_task_success(error)

                # We append the tuple to the results list
                results.append((error, steps))
            
                # Increase the success counter if the task was successfull
                if success:
                    successes += 1
                    cond_id += 1
                    retries = 0
                else:
                    retries += 1
                    if retries >= max_retries_per_condition:
                        print(f"[Condition-{cond_id}] Failed {retries} times (probably a joint limit!)... Skip the condition.")
                        cond_id += 1
                        retries = 0
                    else:
                        print("Task was not a success, retrythe same sobol condition!")
                        total_retries += 1
            
                # Increase the counter of the total number of simulated tasks
                n_simulated_tasks += 1
                env.close()
                env = None

            # Print the debug report
            print_final_report(successes, n_simulated_tasks, total_retries, recorder, results)

            # Menu code...
            choice = input("Want to repeat the tasks? (Y/N): ")
            task_loop = choice.upper() == "Y"

    except KeyboardInterrupt:
        # Make the user understand he killed the process...
            print("\n")
            print("="*60)
            print("Simulation killed by user...")  
            print("="*60)   
    except Exception:
        import traceback
        traceback.print_exc()    
    finally:
        # If an environment was initialized we also make sure to close it!
        if env is not None:
            env.close()

        # Make sure to finalize the HDF5 dataset...
        recorder.finalize()
    
        
def check_task_success(error_norm: float) -> bool:
    """Checks for task success by comparing the passed task error with the tolerance defined in the 'settings' file.

    Args:
        error_norm (float): The norm of the error computed considering the XYZ axes.

    Returns:
        bool: Whether the task was successfull or not.
    """

    # First check if a valid error was actually passed!
    if error_norm is not None:
        # We tag an episode succesfull if its final error_norm was lower than the final allowed tolerance
        success = error_norm <= settings.phase_tolerances['drop'][0]
    else:
        # If the passed error_norm was not valid, the task gets tagged as unsuccessfull
        success = False

    print(f"The task was a: {'SUCCESS' if success else 'FAIL'}")

    return success

def execute_task(task_number: int, env: "SingleArmEnv", recorder: HDF5Recorder, record: bool = True) -> tuple[float, int]:
    """Actually executes the task by calling the correct functions in the movements controller.

    Args:
        task_number (int): The numerical ID that identifies the task.
        env (SingleArmEnv): The simulation environment.
        recorder (HDF5Recorder): The HDF5 recorder used for recording the episodes.
        record (bool, optional): Boolean to indicate if the simulated episodes should be recorder or not. Defaults to True.

    Returns:
        tuple[float, int]: A tuple containing the final placement error norm and the number of total steps the simulations has lasted.
    """

    # Print the task number for debug/visualization
    print(f"[Task - {task_number}]")

    # The final placing error and number of steps of our simulations are returned
    error, steps = pick_and_place_task(env, record=record, recorder=recorder)

    return (error, steps)
    

def print_final_report(n_successes: int, n_total_tasks: int, total_retries: int, recorder: HDF5Recorder, results: list[tuple[float, int]]) -> None:
    """This function print the final report after executing N tasks.

    Args:
        n_successes (int): The number of successes we achieved.
        n_total_tasks (int): The number of total tasks we simulated.
        recorder (HDF5Recorder): The HDF5 recorder.
        results (list[tuple[float, int]]): A list of 'n_total_tasks' tuples, where each tuple contains the final error norm of the executed task and the number of total steps it has lasted.
    """

    print("=" * 60)
    print(f"OSC final report: {n_successes}/{n_total_tasks - total_retries} successful tasks.")
    print(f"Dataset episodes currently saved: {recorder.saved_episodes}")
    print("=" * 60)
    np_results = np.array(results)

    valid = [(e,s) for e,s in results if e is not None]
    if not valid:
        print("No valid episodes.")
        return

    errors = np_results[:, 0]
    steps = np_results[:, 1]

    valid_mask = (errors != None)

    filtered_errors = errors[valid_mask]
    filtered_steps = steps[valid_mask]

    average_err = np.mean(filtered_errors)
    average_steps = np.mean(filtered_steps)
    print(f"Average final error from the drop point: {average_err: 0.5f}")
    print(f"Average number of steps per episode: {average_steps: 0.5f}")


def configure_task(env: "SingleArmEnv"):
    """    
    Prepares the task for the simulation.

    Args:
        env (SingleArmEnv): The simulation environment.
    """

    '''# Grab the model instance.
    model = env.sim.model

    # Modify the camera FOV
    camera_id = model.camera_name2id(settings.dataset_camera_name)
    #model.cam_fovy[camera_id] = settings.dataset_camera_fovy'''

    env.sim.forward()

def create_randomized_env(config: dict | None, init_conditions: dict | None = None) -> DomainRandomizationWrapper:
    """This function creates and initializes the simulation environment with the configurations chosen in the 'settings' file.
        In particular we also use a DomainRandomizationWrapper which wraps our robosuite environment and applies the Domain Randomization features to it.

    Args:
        config (dict): The environment's configuration dictionary.

    Returns:
        DomainRandomizationWrapper: The environment with the Domain Randomization active.
    """

    # As the docs say, we use this so that entire geom groups are randomized as a whole
    macros.USING_INSTANCE_RANDOMIZATION = True

    # Create the base environment
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=settings.render_on_screen, # On-screen renderer, usefule for debug, but slow for dataset gathering...
        has_offscreen_renderer=True, # Necessary for the dataset!
        use_camera_obs=True, # Also for dataset...
        render_camera=settings.dataset_external_camera_name,    # De-comment this line to see the simulation from the wrist camera
        camera_names=settings.dataset_camera_names, # Note: We changed the value to the list of cameras!
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1500, # So we are sure that all the pick and places terminate
        reward_shaping=True, # Not useful in our case but had implemented it anyway...
        init_conditions=init_conditions,
    )

    """
    # Place the robot in the intial pose defined in the settings file (empirically and theoretically found!)
    robot_init_qpos = np.asarray(env.robots[0].init_qpos, dtype=float).copy()
    robot_init_qpos[:6] = np.asarray(settings.starting_pose, dtype=float)
    env.robots[0].init_qpos = (robot_init_qpos)
    """

    # We use domain randomization to create a more robust dataset
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False,
        randomize_lighting=False,
        randomize_camera=False, # We first set this to False so we don't randomize the grip camera as well...
        randomize_dynamics=False, # Seems useful but for the first trainings we avoid using this...
        randomize_on_reset=True, # So the arena gets re-randomized every time we reset the env
        randomize_every_n_steps=0, # Randomization must not happen during the episode
    )

    env.reset()

    '''
    # Enable shadows
    model = env.sim.model
    if model.nlight > 0:
        model.light_castshadow[:] = int(settings.camera_shadows)
        active_lights = model.light_active.astype(bool)

        model.light_ambient[active_lights] = settings.camera_shadow_ambient
        model.light_diffuse[active_lights] = settings.camera_shadow_diffuse
    '''

    # Disable the rangefinders rays...
    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)

    if offscreen_context is not None:
        offscreen_context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    env.sim.forward()

    return env


def debug_wrist_camera(env, label=""):
    model = env.sim.model
    data = env.sim.data

    env.sim.forward()

    obj = env.objects[0]
    obj_id = model.body_name2id(obj.root_body)
    obj_pos = data.body_xpos[obj_id].copy()

    for cam_id in range(model.ncam):
        cam_name = model.camera_id2name(cam_id)

        if "wristview" not in cam_name:
            continue

        parent_id = int(model.cam_bodyid[cam_id])
        parent_name = model.body_id2name(parent_id)

        cam_pos = data.cam_xpos[cam_id].copy()

        rotation = data.cam_xmat[cam_id].reshape(3, 3)

        # MuJoCo cameras look along local -Z.
        forward = -rotation[:, 2]

        to_object = obj_pos - cam_pos
        distance = np.linalg.norm(to_object)
        to_object /= max(distance, 1e-8)

        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(forward, to_object),
                    -1.0, 1.0
                )
            )
        )

        print(f"\n[WRIST CAMERA] {label}")
        print("Camera:", cam_name)
        print("Attached body:", parent_name)
        print("Camera XYZ:", np.round(cam_pos, 4))
        print("Object XYZ:", np.round(obj_pos, 4))
        print("Viewing direction:", np.round(forward, 4))
        print("Angle to object:", round(angle, 2), "deg")

        return cam_pos


def pick_and_place_task(env: "SingleArmEnv", recorder: HDF5Recorder, record: bool = True) -> tuple[float, int]:
    """This function executes the pick and place tas.

    Args:
        env (SingleArmEnv): the simulation environment.
        recorder (HDF5Recorder): The HDF5 recorder.
        record (bool, optional): Whether we should record the episode or not. Defaults to True.

    Returns:
        tuple[float, int]: A tuple containing the final placing error norm and the number of total steps the simulation has lasted.
    """

    try:
        # This avoids the "first recorder frame is not randomized" error...
        #env.reset()
        env.sim.forward()
        try:
            # We force the update of the env
            env._get_observations(force_update=True)
        except TypeError:
            # Fallback for older robosuite's versions...
            env._get_observations()

        # Error flag and counter variables...
        error = None
        steps = 0

        # We loop on every present object, most of the times it's only 1, but whatever...
        for obj in env.objects:
            # Get the object name and its id in the simulation
            obj_name = obj.root_body
            obj_id = env.sim.model.body_name2id(obj_name)
            # Grab the object's position and quaternion
            pos = env.sim.data.body_xpos[obj_id].copy()     # Copy! Otherwise we get a reference to the original array and 
            quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!

            # The final error and steps are returned...
            error, steps = pick_and_place_action(env, obj, pos, quat, recorder, record=record)

            # If the action was successfull we save it...
            if error is not None and error <= settings.phase_tolerances['drop'][0]:
                recorder.save_episode(get_global_seed())
            else:
                recorder.discard_episode()
    except Exception:
        # Never keep an interrupted / corrupted demonstration.
        recorder.discard_episode()
        raise

    return (error, steps)

def pick_and_place_action(env: "SingleArmEnv", obj, pos: np.ndarray, quat: np.ndarray, recorder: HDF5Recorder, record: bool = True) -> tuple[float, int]:
    """This function defines how the pick and place task should be carried out.

    Args:
        env (SingleArmEnv): The simulation environment.
        obj (_type_): The object we are currently trying to pick and place.
        pos (np.ndarray): The object's position.
        quat (np.ndarray): The object's quaternion.
        recorder (HDF5Recorder): The HDF5 recorder.
        record (bool, optional): Whether we should record the episode or not. Defaults to True.

    Returns:
        tuple[float, int]: _description_
    """

    # Grab the object id for later
    obj_id = env.sim.model.body_name2id(obj.root_body)

    # Initialize the movement controller
    movements_ctrl = OSCMovementController(env, recorder=recorder)

    # Note: We usually copy np.ndarray because otherwise numpy just references the memory location
    # resulting in a change in the variable everywhere in the code

    # Define the grasp position as the object position
    grasp_pos = np.asarray(pos, dtype=float).copy()

    # Define the height from the TABLE and not the cube (otherwise bigger cubes would go higher!)
    table_z = float(env.mujoco_arena.table_top_abs[2])
    object_half_height = max(grasp_pos[2] - table_z, 0.0)
    initial_cube_z = float(grasp_pos[2])

    # Define the position OVER the object
    above_object_pos = grasp_pos.copy()
    above_object_pos[2] += object_half_height + settings.pregrasp_top_clearance

    # Define the transport position, this was found empirically with some precisely focused tests!
    transport_z = max(table_z + settings.transport_bottom_clearance + object_half_height, settings.maximum_transport_z)
    transport_pos = grasp_pos.copy()
    transport_pos[2] = transport_z

    # Position directly ABOVE the central drop point.
    above_drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], transport_z], dtype=float)

    # Final drop height.
    drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], grasp_pos[2] + settings.drop_clearance], dtype=float)

    # Total steps for debug/report
    total_steps = 0

    '''
    obs = env._get_observations(force_update=True)
    from PIL import Image
    img_upright = np.flipud(obs[settings.dataset_external_camera_observation_key])
    Image.fromarray(img_upright).save("check_frontview_256.png")
    '''
    # Move over the object.
    recorder.phase = "approach"
    tol_p, tol_r = settings.phase_tolerances['approach']
    done, steps = movements_ctrl.move_robot_to_position(above_object_pos, quat, record=record, position_tolerance=tol_p, rotation_tolerance_deg=tol_r, position_gain=settings.gains['position']['approach'], rotation_gain=settings.gains['rotation']['align'])
    total_steps += steps

    if not done:
        print(f"OSC failed during approach. [Steps: {steps}]")
        return (None, total_steps)

    # Descend to the object.
    recorder.phase = "descend"
    tol_p, tol_r = settings.phase_tolerances['descend']
    done, steps = movements_ctrl.move_robot_to_position(grasp_pos, quat, record=record, position_tolerance=tol_p, rotation_tolerance_deg=tol_r, position_gain=settings.gains['position']['descend'])
    total_steps += steps

    if not done:
        print(f"OSC failed during grasp descent. [Steps: {steps}]")
        return (None, total_steps)

    # Close the gripper.
    recorder.phase = "grasp"
    done, steps = movements_ctrl.set_gripper(target=0.0, record=record, obj=obj)
    total_steps += steps

    #print(f"Closing steps: {steps}")

    if not done:
        print(f"OSC failed while closing the gripper. [Steps: {steps}]")
        return (None, total_steps)

    # Lift the object vertically to the transport height.
    # Note: We use payload compensation to achieve higher precision in some particular movements
    recorder.phase = "lift"
    tol_p, tol_r = settings.phase_tolerances['lift']
    done, steps = movements_ctrl.move_robot_to_position(transport_pos, quat, payload_compensation=True, position_tolerance=tol_p, rotation_tolerance_deg=tol_r, record=record, position_gain=settings.gains['position']['lift'])
    total_steps += steps

    if not done:
        print(f"OSC failed during vertical lift. [Steps: {steps}]")
        return (None, total_steps)

    lifted_cube_z = float(env.sim.data.body_xpos[obj_id][2])
    if lifted_cube_z - initial_cube_z < 0.025:
        print(f"[INVALID] Cube did not lift with gripper: delta_z={(lifted_cube_z - initial_cube_z)*1000:.1f} mm")
        return (None, total_steps)
    
    # Move diagonally above the central drop point.
    recorder.phase = "transport"
    tol_p, tol_r = settings.phase_tolerances['transport']
    done, steps = movements_ctrl.move_robot_to_position(above_drop_pos, payload_compensation=True, record=record, position_tolerance=tol_p, rotation_tolerance_deg=tol_r, position_gain=settings.gains['position']['transport'], rotation_gain=settings.gains['rotation']['align'])
    total_steps += steps

    if not done:
        print(f"OSC failed during transport. [Steps: {steps}]")
        return (None, total_steps)

    # Now just align the cube!
    #recorder.phase = "align"
    #done, steps = movements_ctrl.move_robot_to_position(above_drop_pos, payload_compensation=True, record=record, position_tolerance=transport_tol, rotation_tolerance_deg=rotation_tol, rotation_gain=settings.gains['rotation']['align'])
    #total_steps += steps

    #if not done:
        #print("OSC failed during alignment.")
        #return (None, total_steps)

    # Descend to drop position.
    recorder.phase = "lower"
    tol_p, tol_r = settings.phase_tolerances['lower']
    done, steps = movements_ctrl.move_robot_to_position(drop_pos, record=record, position_tolerance=tol_p, rotation_tolerance_deg=tol_r, position_gain=settings.gains['position']['lower'])
    total_steps += steps

    if not done:
        print(f"OSC failed during drop descent. [Steps: {steps}]")
        return (None, total_steps)
    
    # Open the gripper.
    recorder.phase = "release"
    done, steps = movements_ctrl.set_gripper(target=1.0, record=record, obj=obj)
    total_steps += steps

    #print(f"Opening steps: {steps}")

    if not done:
        print(f"OSC failed while opening the gripper. [Steps: {steps}]")

    # Some debug/report code...
    final_object_pos = (env.sim.data.body_xpos[obj_id].copy())
    center_error = np.linalg.norm(final_object_pos[:2] - np.asarray(settings.drop_point, dtype=float))

    print(f"Final error between drop point ad actual position: {center_error: .5f}m")

    return (center_error, total_steps)


if __name__ == "__main__":
    main()