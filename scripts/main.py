import custom_env  # noqa: F401
import custom_robot  # noqa: F401
import ik

# Import utilities files
import movements
import mujoco
import numpy as np
import robosuite as suite

# Import the settings file
import settings
from robosuite import load_controller_config, macros
from robosuite.utils import transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.wrappers import DomainRandomizationWrapper


def main():
    # Create the CompositeController JOINT_POSITION modality
    config = load_controller_config(default_controller="JOINT_POSITION")

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

    # Initialize movements and IK controllers
    movement_ctrl = movements.MovementController(env)
    ik_ctrl = ik.IKController(env)

    # Convert the quaternion to robosuite convention (x, y, z, w)
    quat_robosuite = T.convert_quat(quat, to="xyzw")
    quat_matrix = T.quat2mat(quat_robosuite)

    # Define the transformation needed for mink
    target_position, r_matrix = ik_ctrl.define_target(pos, quat_matrix)

    # Create the mink target in a suitable format
    mink_transformation = ik.make_mink_target(target_position, r_matrix)

    movement_ctrl.get_robot_joints_pos()

    """
    n_faces = movement_ctrl.determine_n_faces(obj)

    Let's print the rewards to see if the heuristic is correct (reward should be growing in a monothonic way.)

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
    """

    return (0,0,0)
    
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