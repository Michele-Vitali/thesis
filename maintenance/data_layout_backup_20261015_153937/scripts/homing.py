import copy
import math

import numpy as np

import settings
from switch_controller import switch_arm_controller
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv


def _read_joint_state(env: "SingleArmEnv"):
    """Read current arm joint positions and velocities.
    
    Args:
        env (SingleArmEnv): The simulation environment.
    
    Returns:
        (np.ndarray, np.ndarray): A tuple where the first array are the joint's positions and the second array are the joint's velocities. 
    """

    # Grab the robot instance.
    robot = env.robots[0]

    # Grab the joint's positions and velocities.
    qpos = np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=float).copy()
    qvel = np.asarray(env.sim.data.qvel[robot._ref_joint_vel_indexes], dtype=float).copy()

    return (qpos, qvel)


def _joint_delta_to_action(controller, delta: np.ndarray):
    """
    Convert a desired joint displacement (radians) into a normalized action for robosuite JOINT_POSITION.

    Args:
        controller: The current controller.
        delta (np.ndarray): The delta from which we should compute the corresponding action.
    
    Returns:
        np.ndarray: The action array.
    """

    # Ensure the delta format is correct.
    delta = np.asarray(delta, dtype=float)
    n = delta.size

    # Define the controller's input/output minimum/maximum actions.
    input_min = np.broadcast_to(np.asarray(controller.input_min, dtype=float), (n,))
    input_max = np.broadcast_to(np.asarray(controller.input_max, dtype=float), (n,))
    output_min = np.broadcast_to(np.asarray(controller.output_min, dtype=float), (n,))
    output_max = np.broadcast_to(np.asarray(controller.output_max, dtype=float), (n,))

    # Compute the range of output.
    output_span = output_max - output_min

    # Check if the range is valid.
    if np.any(output_span <= 0):
        raise ValueError("Invalid JOINT_POSITION output limits.")

    # Invert the action scaling performed by robosuite.
    action = (delta - output_min) / output_span * (input_max - input_min) + input_min

    return np.clip(action, input_min, input_max)


import numpy as np


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

    for step in range(max_steps):
        # Current joint positions
        qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()

        # Difference between target and current position
        error = home_qpos - qpos

        # Check if HOME has been reached and robot is stable
        qvel = env.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

        # Check if withing position tolerance and slow enough.
        if np.max(np.abs(error)) < tolerance and np.max(np.abs(qvel)) < 0.05:
            print(f"HOME reached in {step} steps!")
            return

        # Convert joint displacement into controller action
        output_min = np.broadcast_to(controller.output_min, error.shape)
        output_max = np.broadcast_to(controller.output_max, error.shape)
        input_min = np.broadcast_to(controller.input_min, error.shape)
        input_max = np.broadcast_to(controller.input_max, error.shape)

        action_joints = input_min + ((np.clip(error, output_min, output_max) - output_min) / (output_max - output_min)) * (input_max - input_min)

        # Last dimension controls the gripper: 0 = HOLD
        action = np.concatenate([action_joints, [0.0]])

        _, _, done, _ = env.step(action)
        env.render()

        if done:
            raise RuntimeError("Episode ended during homing!")

    raise RuntimeError("Failed to reach HOME!")

def prepare_episode(env: "SingleArmEnv", joint_config: dict | None = None, osc_config : dict | None = None):
    """
    Prepare a new simulation episode.

    1. Reset using JOINT_POSITION.
    2. Move from REST to HOME.
    3. Switch to the original custom OSC_POSE.
    4. Return the initial observation for the task.
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
