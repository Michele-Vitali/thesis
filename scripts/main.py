# Import our custom env and robot
import custom_env  # noqa: F401
import custom_robot  # noqa: F401

# Import some libraries
import mujoco
import numpy as np
import robosuite as suite

# Import the settings file
import settings
from custom_osc import install_prob3_custom_osc
from hdf5_recorder import HDF5Recorder, NullRecorder

# Import useful classes
from homing import prepare_episode
from movements import OSCMovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():
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
    env = create_randomized_env(joint_config)

    try:
        task_loop = True
        while task_loop:
            # Ask how many episodes we want to simulate
            n_tasks = int(input("How many times do you want to simulate the OSC task? "))

            # Variables for debugging and reporting
            n_simulated_tasks = 0
            successes = 0   # Also works as incremental ID of tasks...
            results = []

            # Loop to ensure we get N successes (and not N simulations, as failures should not be included or the model will learn on bad episodes...)
            while successes < n_tasks:
                prepare_episode(env, joint_config, osc_config)
                configure_task(env)
                (error, steps) = execute_task(successes, env, recorder, record)
                success = check_task_success(error)

                # We append the tuple to the results list
                results.append((error, steps))
            
                # Increase the success counter if the task was successfull
                if success:
                    successes += 1
                else:
                    print("Task was not a success, retry!")
            
                # Increase the counter of the total number of simulated tasks
                n_simulated_tasks += 1

            # Print the debug report
            print_final_report(successes, n_simulated_tasks, recorder, results)

            # Menu code...
            choice = input("Want to repeat the tasks? (Y/N): ")
            task_loop = choice.upper() == "Y"

    except KeyboardInterrupt:
        # Make the user understand he killed the process...
        print("\n")
        print("="*60)
        print("Simulation killed by user...")  
        print("="*60)      

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
        success = error_norm <= settings.final_tolerance
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
    

def print_final_report(n_successes: int, n_total_tasks: int, recorder: HDF5Recorder, results: list[tuple[float, int]]) -> None:
    """This function print the final report after executing N tasks.

    Args:
        n_successes (int): The number of successes we achieved.
        n_total_tasks (int): The number of total tasks we simulated.
        recorder (HDF5Recorder): The HDF5 recorder.
        results (list[tuple[float, int]]): A list of 'n_total_tasks' tuples, where each tuple contains the final error norm of the executed task and the number of total steps it has lasted.
    """

    print("=" * 60)
    print(f"OSC final report: {n_successes}/{n_total_tasks} successful tasks.")
    print(f"Dataset episodes currently saved: {recorder.saved_episodes}")
    print("=" * 60)
    np_results = np.array(results)
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

    # Grab the model instance.
    #model = env.sim.model

    # Modify the camera FOV
    #camera_id = model.camera_name2id(settings.)
    #model.cam_fovy[camera_id] = settings.dataset_camera_fovy

    env.sim.forward()

def create_randomized_env(config: dict | None) -> DomainRandomizationWrapper:
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
        camera_names=settings.dataset_camera_names, # Note: We changed the value to the list of cameras!
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1500, # So we are sure that all the pick and places terminate
        reward_shaping=True, # Not useful in our case but had implemented it anyway...
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
        randomize_color=True,
        randomize_lighting=True,
        randomize_camera=False, # We first set this to False so we don't randomize the grip camera as well...
        randomize_dynamics=False, # Seems useful but for the first trainings we avoid using this...
        randomize_on_reset=True, # So the arena gets re-randomized every time we reset the env
        randomize_every_n_steps=0, # Randomization must not happen during the episode
    )

    env.reset()

    # Enable shadows
    model = env.sim.model
    if model.nlight > 0:
        model.light_castshadow[:] = int(settings.camera_shadows)
        active_lights = model.light_active.astype(bool)

        model.light_ambient[active_lights] = settings.camera_shadow_ambient
        model.light_diffuse[active_lights] = settings.camera_shadow_diffuse 

    # Disable the rangefinders rays...
    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)

    if offscreen_context is not None:
        offscreen_context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    env.sim.forward()

    return env

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
            if error is not None and error <= settings.final_tolerance:
                recorder.save_episode()
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

    # Post-grasp tolerances
    post_grasp_pos_tolerance = settings.transport_position_tolerance
    post_grasp_rot_tolerance = settings.transport_rotation_tolerance

    # Total steps for debug/report
    total_steps = 0
    
    # Move over the object.
    recorder.phase = "approach"
    done, steps = movements_ctrl.move_robot_to_position(above_object_pos, quat, record=record, position_tolerance=settings.transport_position_tolerance, rotation_tolerance_deg=settings.transport_rotation_tolerance)
    total_steps += steps

    if not done:
        print("OSC failed during approach.")
        return (None, total_steps)

    # Descend to the object.
    recorder.phase = "descend"
    done, steps = movements_ctrl.move_robot_to_position(grasp_pos, quat, record=record)
    total_steps += steps

    if not done:
        print("OSC failed during grasp descent.")
        return (None, total_steps)

    # Capture actual open finger positions before grasping the cube.
    finger_indices = env.robots[0]._ref_gripper_joint_pos_indexes
    open_qpos = np.asarray(env.sim.data.qpos[finger_indices], dtype=float).copy()

    # Close the gripper.
    recorder.phase = "grasp"
    done, steps = movements_ctrl.set_gripper(target=0, record=record, obj=obj)
    total_steps += steps

    if not done:
        print("OSC failed while closing the gripper.")
        return (None, total_steps)

    # Change the controller kp and damping ratio to compensate object's inertia...
    #set_osc_gains(env, kp=settings.grasp_kp, damping_ratio=settings.grasp_damping)

    # Lift the object vertically to the transport height.
    # Note: We use payload compensation and some tolerances to achieve higher precision in some particular movements
    recorder.phase = "lift"
    done, steps = movements_ctrl.move_robot_to_position(transport_pos, quat, payload_compensation=True, position_tolerance=post_grasp_pos_tolerance, rotation_tolerance_deg=post_grasp_rot_tolerance, record=record)
    total_steps += steps

    if not done:
        print("OSC failed during vertical lift.")
        return (None, total_steps)

    lifted_cube_z = float(env.sim.data.body_xpos[obj_id][2])
    if lifted_cube_z - initial_cube_z < 0.025:
        print(f"[INVALID] Cube did not lift with gripper: delta_z={(lifted_cube_z - initial_cube_z)*1000:.1f} mm")
        return (None, total_steps)
    
    # Move diagonally above the central drop point.
    recorder.phase = "transport"
    done, steps = movements_ctrl.move_robot_to_position(above_drop_pos, quat, payload_compensation=True, record=record, position_tolerance=settings.transport_position_tolerance, rotation_tolerance_deg=settings.transport_rotation_tolerance)
    total_steps += steps

    if not done:
        print("OSC failed during transport.")
        return (None, total_steps)

    # Now just align the cube!
    recorder.phase = "align"
    done, steps = movements_ctrl.move_robot_to_position(above_drop_pos, payload_compensation=True, record=record, position_tolerance=settings.transport_position_tolerance, rotation_tolerance_deg=settings.transport_rotation_tolerance)
    total_steps += steps

    if not done:
        print("OSC failed during alignment.")
        return (None, total_steps)

    # Descend to drop position.
    recorder.phase = "lower"
    done, steps = movements_ctrl.move_robot_to_position(drop_pos, payload_compensation=False, record=record)
    total_steps += steps

    if not done:
        print("OSC failed during drop descent.")
        return (None, total_steps)
    
    # Open the gripper.
    recorder.phase = "release"
    done, steps = movements_ctrl.set_gripper(target=1, record=record, obj=obj)
    total_steps += steps

    if not done:
        print("OSC failed while opening the gripper.")

    # Reset controller configuration...
    #set_osc_gains(env, kp=settings.free_kp, damping_ratio=settings.free_damping)

    # Some debug/report code...
    final_object_pos = (env.sim.data.body_xpos[obj_id].copy())
    center_error = np.linalg.norm(final_object_pos[:2] - np.asarray(settings.drop_point, dtype=float))
    z_error = abs(float(final_object_pos[2] - initial_cube_z))
    released_qpos = np.asarray(env.sim.data.qpos[finger_indices], dtype=float).copy()
    gripper_reopened = bool(np.all(np.abs(released_qpos - open_qpos) < 0.08))

    print(f"Final error between drop point ad actual position: {center_error: .5f}m")

    if z_error > 0.015 or not gripper_reopened:
        print("[INVALID] Cube not resting on table or gripper not open.")
        return (None, total_steps)

    return (center_error, total_steps)

'''
Commented out as we should not be able to change controller settings at runtime on the real robot...

def set_osc_gains(env, kp: np.ndarray, damping_ratio: np.ndarray) -> None:
    """This function is used to change the OSC controller's kp and damping ration configurations for grasping to compensate the grasped object's inertia.

    Args:
        env (_type_): The simulation environment.
        kp (np.ndarray): The final desired kp array.
        damping_ratio (np.ndarray): The final desired damping ration array.
    """

    # Grab the controller
    robot = env.robots[0]
    osc = robot.controller

    # Make sure the kp and dampin_ratio arrays are in the correct format (1-D array with 6 values)
    if kp.ndim == 0:
        kp = np.full(6, float(kp))

    if damping_ratio.ndim == 0:
        damping_ratio = np.full(6, float(damping_ratio))

    # Update the controller configurations...
    osc.kp = kp
    osc.kd = 2.0 * np.sqrt(kp) * damping_ratio
'''
    

if __name__ == "__main__":
    main()