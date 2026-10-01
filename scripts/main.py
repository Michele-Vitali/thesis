import time

import custom_env  # noqa: F401
import custom_robot  # noqa: F401
import mujoco
import numpy as np
import robosuite as suite

# Import the settings file
import settings
from ik import IKController

# Import utilities files
from movements import MovementController
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():
    # Create the CompositeController JOINT_POSITION modality
    config = load_controller_config(default_controller="JOINT_POSITION")
    config["output_min"] = -0.15
    config["output_max"] = 0.15

    task_loop = True
    while task_loop:
        n_tasks = int(input("How many times do you want to simulate the task? "))
        results = [[], [], []]
        for i in range(n_tasks):
            steps, reward, accuracy = pick_and_place_task(config, i)
            results[0].append(steps)
            results[1].append(reward)
            results[2].append(accuracy)

        averages = [sum(sublist) / len(sublist) for sublist in results]

        print("="*60)
        print(f"Final report.\n - Average steps: {averages[0]: .2f}.\n - Average reward: {averages[1]: .2f}.\n - Average accuracy over {n_tasks} iterations: {averages[2]: .4f}")
        print("="*60)

        choice = input("Want to repeat the tasks? (Y/N): ")
        task_loop = choice.upper() == "Y"

def create_randomized_env(config: dict) -> DomainRandomizationWrapper:
    # As the docs say, we use this so that entire geom groups are randomized as a whole
    macros.USING_INSTANCE_RANDOMIZATION = True
    # Create the base environment
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True, 
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20, # Limits the robot to 20 actions/second (5 for testing in lab...)
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1000, # So we are sure that all the pick and places terminate
        reward_shaping=True, # So we enable RL success check
    )

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

    if env.has_renderer:
        env.sim._render_context_offscreen.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        env.render()

    return env

def pick_and_place_task(config: dict, task_index: int) -> tuple[int, float, float]:
    # PickAndPlace situation

    env = create_randomized_env(config)

    for obj in env.objects:
        obj_name = obj.root_body
        obj_id = env.sim.model.body_name2id(obj_name)
        # Grab the position
        pos = env.sim.data.body_xpos[obj_id].copy()
        # Grab the rotation (quaternion)                # Copy! Otherwise we get a reference to the original array and 
        quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!

        steps, reward, accuracy = pick_and_place_action(env, obj, pos, quat, task_index)

        #success = env.env._check_success()
        #print(f"{obj_name}: {'Success' if success else 'Failed'}!")

        return (0, 0, 0)#(steps, reward, accuracy)

    # Close the env for any cleanup
    env.close()


def pick_and_place_action(env: "SingleArmEnv", obj, pos: np.array, quat: np.array, task_index: int) -> tuple[int, float, float]:
    """
    Define the position over the cube, which is object_pos + 10cm on the z-axis
    target_pos:
        - [0], positions (x,y,z)
        - [1], rotations (qw, qx, qy, qz)
    """

    print(f"Initial object position: {env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)]}")


    # Initialize movements controller.
    movements_ctrl = MovementController(env)

    movements_ctrl.init_robot_pose()
    # First render the still robot
    env.render()

    # Now start the pick and place
    # Define a boolean for knowing if some steps went wrong
    move_on = False

    # Move the robot to the final pre-grasping position
    move_on = movements_ctrl.move_robot_to_position(pos, quat)
    
    # Verify the real Cartesian position after JOINT_POSITION execution
    # create_mink_target() adds safe_offset_gripper, so we must check the same EEF target here.
    # expected_eef_pos = pos + np.asarray(settings.safe_offset_gripper, dtype=float)
    # move_on = movements_ctrl.check_eef_pos(expected_eef_pos)

    # Close the gripper
    move_on = movements_ctrl.toggle_grab()

    # Start ascending
    elevated_pos = movements_ctrl.elevate_obj(pos, quat)

    # Now move to the center of the table
    center_pos = np.array([0.0, 0.0, elevated_pos[2]], dtype=float)
    print(f"Desired position: {center_pos} and rotation {quat}")
    move_on = movements_ctrl.move_robot_to_position(center_pos, quat)


    time.sleep(5)


    print(f"Final object position: {env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)]}")

    # Loop to visualize results and not close the sim instantly
    for _ in range(100):
        action = np.zeros(env.action_dim)
        env.step(action)
        env.render()

    env.close()

    return (0,0,0)
    
if __name__ == "__main__":
    main()