import copy

import numpy as np
import settings
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from switch_controller import switch_arm_controller


def move_to_home(env: "SingleArmEnv", home_qpos: np.ndarray = settings.starting_pose, tolerance: float = 0.02, max_steps: float = 500):
    """Move the P-Rob3 to HOME using JOINT_POSITION."""

    # Grab the controller instance.
    robot = env.robots[0]
    controller = robot.controller

    # Check its type
    if controller.name != "JOINT_POSITION":
        raise RuntimeError("JOINT_POSITION controller required!")

    # Ensure the pose array format.
    home_qpos = np.asarray(home_qpos, dtype=float)

    for _ in range(max_steps):
        # Current joint positions
        qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()

        # Difference between target and current position
        error = home_qpos - qpos

        # Check if HOME has been reached and robot is stable
        qvel = env.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        # Check if withing position tolerance and slow enough.
        if np.max(np.abs(error)) < tolerance and np.max(np.abs(qvel)) < 0.05:
            return

        # Convert joint displacement into controller action
        output_min = np.broadcast_to(controller.output_min, error.shape)
        output_max = np.broadcast_to(controller.output_max, error.shape)
        input_min = np.broadcast_to(controller.input_min, error.shape)
        input_max = np.broadcast_to(controller.input_max, error.shape)

        action_joints = input_min + ((np.clip(error, output_min, output_max) - output_min) / (output_max - output_min)) * (input_max - input_min)

        # Last dimension controls the gripper: 1.0 = Fully open
        action = np.concatenate([action_joints, [1.0]])

        _, _, done, _ = env.step(action)
        if env.has_renderer:
            env.render()

        if done:
            raise RuntimeError("Episode ended during homing!")

    raise RuntimeError("Failed to reach HOME!")

def prepare_episode(env: "SingleArmEnv", joint_config: dict | None = None, osc_config : dict | None = None):
    """Prepares the simulation environment for a new episode.

    Args:
        env (SingleArmEnv): The simulation environment.
        joint_config (dict, optional): The JOINT_POSITION controller configuration dict. Defaults to None.
        osc_config (dict, optional): The OSC_POSE controller configuration dict. Defaults to None.
    """

    # Grab the robot's instance.
    robot = env.robots[0]

    # Ensure the next reset uses JOINT_POSITION.
    robot.controller_config = copy.deepcopy(joint_config)

    env.reset()

    if robot.controller.name != "JOINT_POSITION":
        raise RuntimeError("JOINT_POSITION is not active after reset.")

    move_to_home(env, home_qpos=settings.starting_pose)

    # Switch to the same custom OSC used for data collection.
    switch_arm_controller(env, osc_config)

    if robot.controller.name != "OSC_POSE":
        raise RuntimeError("OSC_POSE was not activated after homing.")

    env.sim.forward()

    try:
        observation = env._get_observations(force_update=True)
    except TypeError:
        observation = env._get_observations()

    print("[HOMING] P-Rob3 is ready for the task.")

    return observation
