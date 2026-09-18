import robosuite as suite
from robosuite import load_composite_controller_config
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
    config = load_composite_controller_config(controller="BASIC")

    # Create the base environment
    env = suite.make(
        env_name=settings.env_name,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20, # Limits the robot to 20 actions/second
        controller_configs=config
    )

    env.reset()

    final_pos_rot = ()

    # For now we have just 1 cube!
    for obj in env.objects:
        obj_name = obj.root_body
        obj_id = env.sim.model.body_name2id(obj_name)
        # Grab the position
        pos = env.sim.data.body_xpos[obj_id]
        # Grab the rotation (quaternion)
        quat = env.sim.data.body_xquat[obj_id]
        final_pos_rot = (pos, quat)

    """
    Remember! 
    Our robot has a maximum movement per step of 0.05 meter (or 5cm).
    So to move for example 20cm we must move 0.05 for 4 steps!
    """
        

    print("Simulation started!")

    env.render()
        
    # PickAndPlace situation
    # 1. Move over the cube
    """
    Define the position over the cube, which is target_pos + 10cm on the z-axis
    target_pos:
        - [0], positions (x,y,z)
        - [1], rotations (qx, qy, qz, qw)
    """
    target_pos = final_pos_rot[0] + np.array([0, 0, 0.10])
    target_rot = final_pos_rot[1]

    # 1. Move over the target
    movements.move_to_target(env, target_pos)

    # 2. Orientate the gripper as the cube
    movements.rotate(env, target_rot)

    # 2. Start the descent, we just reuse the same function...
    #    but we ensure the gripper is initially open!
    grab = False

    movements.move_to_target(env, final_pos_rot[0])
    movements.toggle_grab(env, grab)

    # 3. Grab the cube and elevate it!
    grab = True

    movements.toggle_grab(env, grab)
    movements.move_to_target(env, target_pos)

    # 4. Go in the middle and drop!
    target_pos = [0,0, final_pos_rot[0][2]] # Keep the same z as the original (on table surface)
    grab = False
    movements.move_to_target(env, target_pos)
    movements.toggle_grab(env, grab)

    time.sleep(10)

if __name__ == "__main__":
    main()