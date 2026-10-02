# Import our custom env and robot
import custom_env  # noqa: F401
import custom_robot  # noqa: F401

# Import some libraries
import mujoco
import numpy as np
import robosuite as suite

# Import the settings file
import settings

# Import useful classes
from lerobot_recorder import LeRobotRecorder
from movements import OSCMovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():
    # Create the controller with OSC_POSE mode
    config = load_controller_config(default_controller="OSC_POSE")

    # Some OSC configurations...

    # Fixed impedance OSC.
    config["impedance_mode"] = "fixed"

    config["kp"] = 150
    config["damping_ratio"] = 1

    # Cartesian actions are interpreted as deltas from the current EEF pose.
    config["control_delta"] = True

    # Explicit normalization of inputs/outputs
    config["input_min"] = -1
    config["input_max"] = +1

    config["output_min"] = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]

    config["output_max"] = [+0.05, +0.05, +0.05, +0.5, +0.5, +0.5]

    recorder = LeRobotRecorder()

    try:
        task_loop = True
        while task_loop:
            n_tasks = int(input("How many times do you want to simulate the OSC task? "))
            results = []
            for i in range(n_tasks):
                success = pick_and_place_task(config=config, task_index=i, recorder=recorder)

                results.append(success)

            #successes = sum(results)

            print("=" * 60)
            #print(f"OSC final report: {successes}/{n_tasks} successful tasks.")
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
        has_renderer=False, # On-screen renderer
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

def pick_and_place_task(config: dict, task_index: int, recorder: LeRobotRecorder) -> bool:

    # Create the env
    env = create_randomized_env(config)

    try:
        for obj in env.objects:
            obj_name = obj.root_body
            obj_id = env.sim.model.body_name2id(obj_name)
            # Grab the position
            pos = env.sim.data.body_xpos[obj_id].copy()
            # Grab the rotation (quaternion)                # Copy! Otherwise we get a reference to the original array and 
            quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!

            success = pick_and_place_action(env, obj, pos, quat, task_index, recorder)

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


def pick_and_place_action(env: "SingleArmEnv", obj, pos: np.ndarray, quat: np.ndarray, task_index: int, recorder: LeRobotRecorder) -> bool:

    # Grab the object id for later
    obj_id = env.sim.model.body_name2id(obj.root_body)

    # Initialize the movement controller
    movements_ctrl = OSCMovementController(env, tracked_body_name=obj.root_body, recorder=recorder)

    # Note: We usually copy np.ndarray because otherwise numpy just references the memory location
    # resulting in a change in the variable everywhere in the code

    # Define the grasp position as the object position
    grasp_pos = np.asarray(pos, dtype=float).copy()

    # Define the position OVER the object
    above_object_pos = grasp_pos.copy()
    above_object_pos[2] = settings.ideal_z

    # Position directly ABOVE the central drop point.
    above_drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], settings.ideal_z], dtype=float)

    # Final drop height.
    drop_pos = np.array([settings.drop_point[0], settings.drop_point[1], grasp_pos[2] + settings.drop_clearance], dtype=float)

    # Move over the object.
    done = movements_ctrl.move_robot_to_position(above_object_pos, quat)

    if not done:
        print("OSC failed while moving above the object.")
        return False

    # Descend to the object.
    done = movements_ctrl.move_robot_to_position(grasp_pos, quat)

    if not done:
        print("OSC failed during grasp descent.")
        return False

    # Close the gripper.
    movements_ctrl.toggle_grab()

    # Elevate the object.
    done = movements_ctrl.move_robot_to_position(above_object_pos, quat)

    if not done:
        print("OSC failed during vertical lift.")
        return False

    # Move OVER the drop point.
    done = movements_ctrl.move_robot_to_position(above_drop_pos, quat)

    if not done:
        print("OSC failed during horizontal transport.")
        return False

    # Descend to drop position.
    done = movements_ctrl.move_robot_to_position(drop_pos, quat)

    if not done:
        print("OSC failed during drop descent.")
        return False

    # Open the gripper.
    movements_ctrl.toggle_grab()

    final_object_pos = (env.sim.data.body_xpos[obj_id].copy())

    center_error = np.linalg.norm(final_object_pos[:2] - np.asarray(settings.drop_point, dtype=float))

    print(f"Final error between drop point ad actual position: {center_error: .5f}")

    # Only demonstrations that actually complete the task are stored.
    # We define the task completed and successfull if the object is within 3cm from the drop point.
    return center_error < 0.03

if __name__ == "__main__":
    main()