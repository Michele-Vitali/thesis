import custom_env  # noqa: F401
import custom_robot  # noqa: F401

# Import utilities files
import movements
import numpy as np
import robosuite as suite

# Import the settings file
import settings
from robosuite import load_controller_config, macros
from robosuite.models import objects
from robosuite.wrappers import DomainRandomizationWrapper


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
                    pick_and_place_task(config)

            choice = input("Want to repeat the task? (Y/N): ")
            task_loop = choice.upper() == "Y"

        choice = input("Want to terminate? (Otherwise choose a task later...) (Y/N): ")
        menu_loop = choice.upper() != "Y"

def pick_and_place_task(config):
    # PickAndPlace situation

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
        control_freq=20, # Limits the robot to 20 actions/second
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
    env.render()

    for obj in env.objects:
        obj_name = obj.root_body
        obj_id = env.sim.model.body_name2id(obj_name)
        # Grab the position
        pos = env.sim.data.body_xpos[obj_id].copy()
        # Grab the rotation (quaternion)                # Copy! Otherwise we get a reference to the original array and 
        quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!
        obj_pos_rot = (pos, quat)
        pick_and_place_action(env, obj, obj_pos_rot)

        success = env.env._check_success()
        print(f"{obj_name}: {'Success' if success else 'Failed'}!")

    env.close()

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

def pick_and_place_action(env, obj, obj_pos_rot):
    """
    Define the position over the cube, which is object_pos + 10cm on the z-axis
    target_pos:
        - [0], positions (x,y,z)
        - [1], rotations (qx, qy, qz, qw)
    """
    obj_pos, obj_rot = obj_pos_rot[0], obj_pos_rot[1]

    over_obj_pos = obj_pos + np.array([0, 0, 0.10])

    # Initialize the movements controller
    movement_ctrl = movements.MovementController(env)

    n_faces = movement_ctrl.determine_n_faces(obj)

    total_steps = 0
    total_reward = 0

    # 1. Move over the target
    steps, reward = movement_ctrl.complex_traslation(over_obj_pos, "Move over")
    total_steps += steps
    total_reward += reward

    # 2. Orientate the gripper as the cube
    steps, reward = movement_ctrl.yaw_rotation(obj_rot, "Rotate", n_faces)
    total_steps += steps
    total_reward += reward

    # 2. Start the descent, we just reuse the same function...
    #    but we ensure the gripper is initially open!
    steps, reward = movement_ctrl.complex_traslation(obj_pos, "Descent")
    total_steps += steps
    total_reward += reward

    # 3. Grab the cube and elevate it!
    movement_ctrl.toggle_grab()
    steps, reward = movement_ctrl.complex_traslation(over_obj_pos, "Elevate")
    total_steps += steps
    total_reward += reward

    # 4. Go in the middle, rotate, go down and drop!
    over_final_pos = np.array([0, 0, over_obj_pos[2]])
    steps, reward = movement_ctrl.complex_traslation(over_final_pos, "Move center")
    total_steps += steps
    total_reward += reward

    # We align the cube with the system axes by putting the target as 
    # a quaternion with w=1 (scalar value) and rotations around the axes at 0
    steps, reward = movement_ctrl.yaw_rotation([1, 0, 0, 0], "Final rotation", n_faces) 
    total_steps += steps
    total_reward += reward
    drop_position = np.array([0, 0, obj_pos[2] + 0.01]) # Keep the same z as the original (on table surface) plus a margin
    steps, reward = movement_ctrl.complex_traslation(drop_position, "Final descent")
    total_steps += steps
    total_reward += reward
    movement_ctrl.toggle_grab()

    # 5. Go back to neutral position
    neutral_pos = np.array([0, 0, obj_pos[2] + 0.30])
    steps, reward = movement_ctrl.complex_traslation(neutral_pos, "Neutral position")
    total_steps += steps
    total_reward += reward

    print(f"Final total steps: {total_steps}. Final total reward: {total_reward}")

if __name__ == "__main__":
    main()