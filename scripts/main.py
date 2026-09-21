import robosuite as suite
from robosuite import load_controller_config
import robosuite.macros as macros
from robosuite.wrappers import DomainRandomizationWrapper
import robosuite.models.objects as objects
import numpy as np

# Import the custom robot and environment so they are initialized
import custom_robot
import custom_env

# Import the settings file
import settings

# Import utilities files
import movements
import time

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
            except:
                print("Invalid choice... Retry.")

        task_loop = True
        while task_loop:
            match num_choice:
                case 1:
                    pick_and_place_task(config)

            choice = input("Want to repeat the task? (Y/N): ")
            task_loop = True if choice.upper() == "Y" else False

        choice = input("Want to terminate? (Otherwise choose a task later...) (Y/N): ")
        menu_loop = False if choice.upper() =="Y" else True

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

    # For now we have just 1 cube!
    for obj in env.objects:
        obj_name = obj.root_body
        obj_id = env.sim.model.body_name2id(obj_name)
        # Grab the position
        pos = env.sim.data.body_xpos[obj_id].copy()
        # Grab the rotation (quaternion)                # Copy! Otherwise we get a reference to the original array and 
        quat = env.sim.data.body_xquat[obj_id].copy()   # it will be modified by the env.step() function!
        obj_pos_rot = (pos, quat)
        n_faces = 4 if isinstance(obj, objects.BoxObject) else 100
        pick_and_place_action(env, obj_pos_rot, n_faces)

    env.close()

def pick_and_place_action(env, obj_pos_rot, n_faces=4):
    """
    Define the position over the cube, which is object_pos + 10cm on the z-axis
    target_pos:
        - [0], positions (x,y,z)
        - [1], rotations (qx, qy, qz, qw)
    """
    obj_pos = obj_pos_rot[0]
    obj_rot = obj_pos_rot[1]

    over_obj_pos = obj_pos + np.array([0, 0, 0.10])

    # 1. Move over the target
    movements.move_to_target(env, over_obj_pos, "Move over")

    # 2. Orientate the gripper as the cube
    movements.yaw_rotation(env, obj_rot, "Rotate", n_faces)

    # 2. Start the descent, we just reuse the same function...
    #    but we ensure the gripper is initially open!
    movements.move_to_target(env, obj_pos, "Descent")

    # 3. Grab the cube and elevate it!
    movements.toggle_grab(env)
    movements.move_to_target(env, over_obj_pos, "Elevate")

    # 4. Go in the middle, rotate, go down and drop!
    over_final_pos = [0,0, over_obj_pos[2]]
    movements.move_to_target(env, over_final_pos, "Move center")

    # We align the cube with the system axes by putting the target as 
    # a quaternion with w=1 (scalar value) and rotations around the axes at 0
    movements.yaw_rotation(env, [1, 0, 0, 0], "Final rotation") 
    drop_position = [0, 0, obj_pos[2]] # Keep the same z as the original (on table surface)
    movements.move_to_target(env, drop_position, "Final descent")
    movements.toggle_grab(env)

if __name__ == "__main__":
    main()