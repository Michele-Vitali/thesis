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

# Import useful classes
from lerobot_recorder import LeRobotRecorder
from movements import OSCMovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():
    # Load our custom OSC_POSE controller!
    #install_prob3_custom_osc()

    # Create the controller with OSC_POSE mode
    config = load_controller_config(default_controller="OSC_POSE")

    # Update the config with the OSC controller configurations from the settings file...
    config.update(settings.osc_config)

    recorder = LeRobotRecorder()
    record = False

    try:
        task_loop = True
        while task_loop:
            n_tasks = int(input("How many times do you want to simulate the OSC task? "))
            successes = 0
            for i in range(n_tasks):
                success = pick_and_place_task(config=config, task_index=i, record=record, recorder=recorder)
                string = "YES" if success else "NO"
                print(f"Success? {string}")

                if success:
                    successes += 1

            print("=" * 60)
            print(f"OSC final report: {successes}/{n_tasks} successful tasks.")
            print(f"LeRobot episodes currently saved: {recorder.saved_episodes}")
            print("=" * 60)

            choice = input("Want to repeat the tasks? (Y/N): ")
            task_loop = choice.upper() == "Y"
    finally:

        # Absolutely necessary for LeRobot v3, as this closes the Parquet writers and writes metadata/statistics.
        recorder.finalize()


def create_randomized_env(config: dict) -> DomainRandomizationWrapper:

    # As the docs say, we use this so that entire geom groups are randomized as a whole
    macros.USING_INSTANCE_RANDOMIZATION = True

    # Create the base environment
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True, # On-screen renderer
        has_offscreen_renderer=True, # Necessary for the dataset!
        use_camera_obs=True, # Also for dataset...
        camera_names=settings.dataset_camera_name,
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1000, # So we are sure that all the pick and places terminate
        reward_shaping=True, # So we enable RL success check
    )

    robot_init_qpos = np.asarray(env.robots[0].init_qpos, dtype=float).copy()

    robot_init_qpos[:6] = np.asarray(settings.starting_pose, dtype=float)

    env.robots[0].init_qpos = (robot_init_qpos)

    # We use domain randomization to create a more robust dataset
    env = DomainRandomizationWrapper(
        env,
        randomize_color=True,
        randomize_lighting=True,
        randomize_camera=True,
        randomize_dynamics=False, # Breaks the whole robot, but seems useful for the future...
        randomize_on_reset=True,
        randomize_every_n_steps=0, # Randomization must not happen during the episode
    )

    env.reset()

    offscreen_context = getattr(env.sim, "_render_context_offscreen", None)

    if offscreen_context is not None:
        offscreen_context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    return env

def pick_and_place_task(config: dict, task_index: int, recorder: LeRobotRecorder, record: bool = True) -> bool:

    # Create the env
    env = create_randomized_env(config)

    success = False

    try:
        for obj in env.objects:
            obj_name = obj.root_body
            obj_id = env.sim.model.body_name2id(obj_name)
            # Grab the position
            pos = env.sim.data.body_xpos[obj_id].copy()
            # Grab the rotation (quaternion)                # Copy! Otherwise we get a reference to the original array and 
            quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!
            
            success = pick_and_place_action(env, obj, pos, quat, task_index, recorder, record=record)
            #success = test(env, obj, pos, quat, task_index, recorder, record=record)

            # If the action was successfull we save it...
            if success:
                recorder.save_episode()
            else:
                recorder.discard_episode()
    except Exception:
        # Never keep an interrupted / corrupted demonstration.
        recorder.discard_episode()
        raise
    finally:
        env.close()

    return success

def test(env, obj, pos, quat, task_index, recorder, record=True):

    movements_ctrl  =OSCMovementController(env, tracked_body_name=obj.root_body, recorder=recorder)
    test_pos = np.array([0.0, 0.0, 0.30])
    done = movements_ctrl.move_robot_to_position(test_pos, record=record)

    return done


def pick_and_place_action(env: "SingleArmEnv", obj, pos: np.ndarray, quat: np.ndarray, task_index: int, recorder: LeRobotRecorder, record: bool = True) -> bool:

    # Grab the object id for later
    obj_id = env.sim.model.body_name2id(obj.root_body)

    # Initialize the movement controller
    movements_ctrl = OSCMovementController(env, tracked_body_name=obj.root_body, recorder=recorder)

    # Note: We usually copy np.ndarray because otherwise numpy just references the memory location
    # resulting in a change in the variable everywhere in the code

    # Define the grasp position as the object position
    grasp_pos = np.asarray(pos, dtype=float).copy()

    # Define the height from the TABLE and not the cube (otherwise bigger cubes would go higher!)
    table_z = float(env.mujoco_arena.table_top_abs[2])
    object_half_height = max(grasp_pos[2] - table_z, 0.0)

    # Define the position OVER the object
    above_object_pos = grasp_pos.copy()
    above_object_pos[2] += object_half_height + settings.pregrasp_top_clearance

    # Define the transport position. It has to be at MAXIMUM 18cm (+ offset_gripper = 30cm approximately)
    transport_z = max(table_z + settings.transport_bottom_clearance + object_half_height, settings.maximum_transport_z)
    transport_pos = grasp_pos.copy()
    transport_pos[2] = transport_z

    # Position directly ABOVE the central drop point.
    above_drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], transport_z], dtype=float)
    print(f"Position above drop: {above_drop_pos}")

    # Final drop height.
    drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], grasp_pos[2] + settings.drop_clearance], dtype=float)

    # Move over the object.
    done = movements_ctrl.move_robot_to_position(above_object_pos, quat, record=record)

    if not done:
        print("OSC failed while moving above the object.")
        return False

    # Descend to the object.
    done = movements_ctrl.move_robot_to_position(grasp_pos, quat, record=record)

    if not done:
        print("OSC failed during grasp descent.")
        return False

    # Close the gripper.
    movements_ctrl.toggle_grab(record=record)

    # Lift the object vertically to the transport height.
    done = movements_ctrl.move_robot_to_position(transport_pos, quat, payload_compensation=False, record=record)

    if not done:
        print("OSC failed during vertical lift.")
        return False
    
    # Move diagonally above the central drop point.
    done = movements_ctrl.move_robot_to_position(above_drop_pos, record=record)

    if not done:
        print("OSC failed during transport.")
        return False

    # Align the cube!
    neutral_quat = [1, 0, 0, 0]
    done = movements_ctrl.move_robot_to_position(above_drop_pos, neutral_quat, record=record)
            
    # Descend to drop position.
    done = movements_ctrl.move_robot_to_position(drop_pos, record=record)

    if not done:
        print("OSC failed during drop descent.")
        return False

    # Open the gripper.
    movements_ctrl.toggle_grab(record=record)

    final_object_pos = (env.sim.data.body_xpos[obj_id].copy())

    center_error = np.linalg.norm(final_object_pos[:2] - np.asarray(settings.drop_point, dtype=float))

    print(f"Final error between drop point ad actual position: {center_error: .5f}m")

    # Only demonstrations that actually complete the task are stored.
    # We define the task completed and successfull if the object is within 5mm from the drop point.
    return center_error < settings.final_tolerance

if __name__ == "__main__":
    main()