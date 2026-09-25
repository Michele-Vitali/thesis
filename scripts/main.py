import time

import mujoco
import numpy as np
import robosuite as suite
from robosuite import load_controller_config, macros
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper

import custom_env  # noqa: F401
import custom_robot  # noqa: F401

# Import utilities files
import movements

# Import the settings file
import settings


def main():
    # Create the CompositeController with OSC POSE and JOINT_POSITION modality
    config = load_controller_config(default_controller="OSC_POSE")

    # Menu
    menu_loop = True
    while menu_loop:
        input_good = False
        print("\n\nChoose the task you want to simulate: ")
        print("1 - Pick and Place")
        while not input_good:
            choice = input("Choice: ")
            try:
                num_choice = int(choice)
                if num_choice < 1 or num_choice > 1:
                    raise ValueError
                else:
                    input_good = True
            except ValueError:
                print("Invalid choice... Retry.")

        task_loop = True
        while task_loop:
            match num_choice:
                case 1:
                    n_tasks = int(input("How many times? "))
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

            choice = input("Want to repeat the task? (Y/N): ")
            task_loop = choice.upper() == "Y"

        choice = input("Want to terminate? (Otherwise choose a task later...) (Y/N): ")
        menu_loop = choice.upper() != "Y"

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
        control_freq=2, # Limits the robot to 20 actions/second (5 for testing in lab...)
        controller_configs=config,
        hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
        horizon=1000, # So we are sure that all the pick and places terminate
        reward_shaping=True, # So we enable RL success check
    )

    # We use domain randomization to create a more robust dataset
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False,
        randomize_lighting=True,
        randomize_camera=False,
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
        obj_pos_rot = (pos, quat)

        test_movement(env)
        #steps, reward, accuracy = pick_and_place_action(env, obj, obj_pos_rot, task_index)

        #success = env.env._check_success()
        #print(f"{obj_name}: {'Success' if success else 'Failed'}!")

        return (0, 0, 0)#(steps, reward, accuracy)

    # Close the env for any cleanup
    env.close()

def test_movement(env):

    action = np.zeros(env.action_dim)

    """for _ in range(50):
        action[-1] = 1.0
        env.step(action)
        env.render()
    """
    """for _ in range(50):
        action[-1] = 0.0
        env.step(action)
        env.render()
    """
    for _ in range(50):
        action[-1] = 1.0
        env.step(action)
        env.render()

        
def pick_and_place_action(env: "SingleArmEnv", obj, obj_pos_rot: tuple, task_index: int) -> tuple[int, float, float]:
    """
    Define the position over the cube, which is object_pos + 10cm on the z-axis
    target_pos:
        - [0], positions (x,y,z)
        - [1], rotations (qx, qy, qz, qw)
    """
    obj_pos, obj_rot = obj_pos_rot[0], obj_pos_rot[1]

    # A z of 30cm is the chosen ideal quote
    over_obj_pos = obj_pos + np.array([0, 0, 0.30])

    # Initialize the movements controller
    movement_ctrl = movements.MovementController(env)

    n_faces = movement_ctrl.determine_n_faces(obj)

    """
    Let's print the rewards to see if the heuristic is correct (reward should be growing in a monothonic way.)
    """

    # 1. Move over the target
    movement_ctrl.complex_traslation(over_obj_pos, "Move over")

    # 2. Orientate the gripper as the cube
    movement_ctrl.yaw_rotation(obj_rot, "Rotate", n_faces)

    # 2. Start the descent, we just reuse the same function...
    #    but we ensure the gripper is initially open!
    movement_ctrl.complex_traslation(obj_pos, "Descent")


    # 3. Grab the cube and elevate it!
    movement_ctrl.toggle_grab()
    movement_ctrl.complex_traslation(over_obj_pos, "Elevate")


    # 4. Go in the middle, rotate, go down and drop!
    over_final_pos = np.array([0, 0, over_obj_pos[2]])
    movement_ctrl.complex_traslation(over_final_pos, "Move center")

    # We align the cube with the system axes by putting the target as 
    # a quaternion with w=1 (scalar value) and rotations around the axes at 0
    movement_ctrl.yaw_rotation([1, 0, 0, 0], "Final rotation", n_faces) 
    drop_position = np.array([0, 0, obj_pos[2] + 0.02]) # Keep the same z as the original (on table surface) plus a margin
    movement_ctrl.complex_traslation(drop_position, "Final descent")
    movement_ctrl.toggle_grab()

    # 5. Go back to neutral position
    neutral_pos = np.array([0, 0, obj_pos[2] + 0.30])
    movement_ctrl.complex_traslation(neutral_pos, "Neutral position")

    total_steps = movement_ctrl.steps
    total_reward = movement_ctrl.reward
    accuracy = total_reward/total_steps

    print(f"[Task-{task_index}] Report:\n - Final total steps: {total_steps}\n - Final total reward: {total_reward: .4f}\n - Final accuracy: {accuracy: .2f}")

    return (total_steps, total_reward, accuracy)

if __name__ == "__main__":
    main()

    """

Still under construction...

def stacking_action(env, obj_pos_rot, obj, table_body_id):
    obj_pos = obj_pos_rot[0]
    obj_rot = obj_pos_rot[1]

    over_obj_pos = obj_pos + np.array([0, 0, 0.10])

    # Save the contact geoms of the object we are manipulating
    gripper = env.robots[0].gripper
    geom_names = [name for names in gripper.important_geoms.values() for name in names]
    gripper_geom_ids = {env.sim.model.geom_name2id(name) for name in geom_names}
    gripper_body_ids = {env.sim.model.geom_bodyid[geom_id] for geom_id in gripper_geom_ids}

    held_geom_ids = {env.sim.model.geom_name2id(g) for g in obj.contact_geoms}

    monitored_geom_ids = held_geom_ids | gripper_geom_ids

    # 1. Move over the target
    movements.complex_traslation(env, over_obj_pos, "Move over")

    # 2. Orientate the gripper as the cube
    movements.yaw_rotation(env, obj_rot, "Rotate")

    # 2. Start the descent, we just reuse the same function...
    #    but we ensure the gripper is initially open!
    movements.complex_traslation(env, obj_pos, "Descent")

    # 3. Grab the cube and elevate it!
    movements.toggle_grab(env)
    movements.complex_traslation(env, over_obj_pos, "Elevate")

    # 4. Go in the middle, rotate, go down and drop!
    over_final_pos = np.array([0, 0, over_obj_pos[2]])
    movements.complex_traslation(env, over_final_pos, "Move center")

    # We align the cube with the system axes by putting the target as 
    # a quaternion with w=1 (scalar value) and rotations around the axes at 0
    movements.yaw_rotation(env, [1, 0, 0, 0], "Final rotation") 
    drop_position = np.array([0, 0, obj_pos[2]]) # Keep the same z as the original (on table surface)
    # For now we activate the stop on contact only for the descent
    obj_body_id = env.sim.model.body_name2id(obj.root_body)
    movements.complex_traslation(env, drop_position, "Final descent", 
                                    stop_on_contact_geom_ids=monitored_geom_ids, 
                                    allowed_body_ids={table_body_id, obj_body_id} | gripper_body_ids)
    movements.toggle_grab(env)
"""