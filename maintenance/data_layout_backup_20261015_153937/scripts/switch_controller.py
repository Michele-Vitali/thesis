import copy

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

def switch_arm_controller(env: "SingleArmEnv", controller_config: dict):
    """Switches the robosuite's controller without modifying any joint position.

    Args:
        env (SingleArmEnv): The simulation environment.
        controller_config (dict): The optional controller configurations.

    Raises:
        TypeError: If the passed controller configurations are not a dict type.
        ValueError: If the passed controller_config is missing the 'type' field.
        RuntimeError: If when switching controllers the action dimension changes.
        RuntimeError: If when switching controllers the joint positions change. 
        RuntimeError: If when switching controller the joint velocities change.

    Returns:
        Controller: The new active controller.
    """
    # Grab the robot instance
    robot = env.robots[0]

    # Check for dict format
    if not isinstance(controller_config, dict):
        raise TypeError("controller_config must be a dict.")

    if "type" not in controller_config:
        raise ValueError("'type' field is missing.")

    # Save the pre-switch state.
    qpos_before = np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes], copy=True)
    qvel_before = np.array(env.sim.data.qvel[robot._ref_joint_vel_indexes], copy=True)
    old_action_dim = robot.action_dim
    old_controller = robot.controller
    old_config = robot.controller_config

    try:
        # Copy a configuration without any reference to the runtime instance
        robot.controller_config = copy.deepcopy(controller_config)

        # Recreate the controller while keeping the same Mujoco simulation.
        robot._load_controller()

        # Update the base pose reference.
        robot.controller.update_base_pose(robot.base_pos, robot.base_ori)

        # Align the new controller with the old configuration.
        # For the OSC_POSE controller also align the cartesian state
        robot.controller.update_initial_joints(qpos_before)

        env.sim.forward()
    except Exception:
        robot.controller = old_controller
        robot.controller_config = old_config

    # Grab the post-switch situation.
    qpos_after = np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes], copy=True)
    qvel_after = np.array(env.sim.data.qvel[robot._ref_joint_vel_indexes], copy=True)

    # Check if switching has changed the situation...
    if not np.allclose(qpos_before, qpos_after, atol=1e-10):
        raise RuntimeError("The controller switch has changed the joint positions!")

    if not np.allclose(qvel_before, qvel_after, atol=1e-10):
        raise RuntimeError("The controller switch has changed the joint velocities!")

    print(f"Active controller: {robot.controller.name} | action_dim = {robot.action_dim}")

    return robot.controller