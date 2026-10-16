import time

import custom_env
import custom_robot
import mujoco
import numpy as np
import robosuite as suite
import settings
from custom_osc import install_prob3_custom_osc
from homing import prepare_episode
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from vla_client import OpenVLAClient


def main():

    # First we must re-create the same environment we used in dataset gathering
    # 1. Initialize our custom OSC_POSE controller.
    install_prob3_custom_osc()

    # 2. Grab the controllers' configurations
    joint_config = load_controller_config(default_controller="JOINT_POSITION")
    osc_config = load_controller_config(default_controller="OSC_POSE")
    osc_config.update(settings.osc_config)

    # 3. Create the actual env.
    env = suite.make(
            env_name=settings.env_name,
            robots=settings.robot,
            gripper_types=settings.gripper_types,
            has_renderer=settings.render_on_screen, # On-screen renderer, usefule for debug, but slow for dataset gathering...
            has_offscreen_renderer=True, # Necessary for the dataset!
            use_camera_obs=True, # Also for dataset...
            camera_names=settings.dataset_camera_name,
            camera_heights=settings.dataset_image_height,
            camera_widths=settings.dataset_image_width,
            camera_depths=False,
            control_freq=settings.dataset_fps,
            controller_configs=joint_config,
            hard_reset=False, # Avoids segfault on macos or glfw error on Linux (per docs...)
            horizon=1000, # So we are sure that all the pick and places terminate
            reward_shaping=True, # Not useful in our case but had implemented it anyway...
        )

    # 3.1 Remove rangefinders visualization...
    context = getattr(env.sim, "_render_context_offscreen", None)

    if context is not None:
        context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    # 4. Instantiate the OpenVLAClient.
    client = OpenVLAClient(
        server_url = settings.server_url,
        unnorm_key=settings.unnorm_key,
        timeout=settings.server_timeout,
    )

    results = []

    n_tasks = 0
    is_good = False
    while not is_good:
        try:
            n_tasks = int(input("How many tasks you want to evaluate? "))
            is_good = True
        except Exception:
            print("You must insert a valid number!")

    print("="*60)
    print("Starting OpenVLA model evaluation...")
    print("="*60)

    n_successes = 0
    for task_id in range(n_tasks):
        success = False
        print(f"[Task-{task_id}]")
        observation = prepare_episode(env, joint_config, osc_config)

        if observation is None:
            observation = read_observation(env)

        if env.robots[0].controller.name != "OSC_POSE":
            raise RuntimeError("OSC_POSE controller not active after homing!")

        if env.robots[0].action_dim != 7:
            raise RuntimeError(f"Expected robot action_dim = 7, got {env.robots[0].action_dim}")

        t0 = time.monotonic()

        error, total_steps = execute_task(env, observation, client)
        elapsed = time.monotonic() - t0

        if error < settings.final_tolerance:
            success = True
            n_successes += 1

        print(f"The task was a {'SUCCESS' if success else 'FAIL'}, it used {total_steps} steps and lasted a total of {elapsed: .1f} seconds.")

        results.append((error, total_steps))

    print_final_report(n_successes, n_tasks, results)

def print_final_report(n_successes: int, n_total_tasks: int, results: list[tuple[float, int]]) -> None:
    """This function print the final report after executing N tasks.

    Args:
        n_successes (int): The number of successes we achieved.
        n_total_tasks (int): The number of total tasks we simulated.
        results (list[tuple[float, int]]): A list of 'n_total_tasks' tuples, where each tuple contains the final error norm of the executed task and the number of total steps it has lasted.
    """

    print("=" * 60)
    print(f"OSC final report: {n_successes}/{n_total_tasks} successful tasks.")
    print("=" * 60)
    np_results = np.array(results)
    averages = np.mean(np_results, axis=0)
    print(f"Average final error from the drop point: {averages[0]: 0.5f}")
    print(f"Average number of steps per episode: {averages[1]: 0.5f}")

def execute_task(env: "SingleArmEnv", obs: dict, client: OpenVLAClient, max_steps: int = 1000) -> tuple[bool, int]:
    """Executes the pick and place task.

    Args:
        env (SingleArmEnv): The simulation environment.
        obs (dict): The observation dictionary.
        client (OpenVLAClient): The OpenVLAClient to interrogate for the next action.
        max_steps (int, optional): The maximum number of steps our simulation should last. Defaults to 400.

    Returns:
        tuple[bool, int]: A tuple where the first element is a boolean that represent whether the task was a success or a fail,
                          the second element indicates the number of steps the simulation lasted. 
    """

    # Counter and flag variables
    total_steps = 0

    for _ in range(max_steps):
        # Prepare the image in the correct expected format.
        image = prepare_rgb(obs)

        # Make the model predict the next action.
        prediction = client.predict(image, settings.dataset_task)

        # We must adapt the output (e.g. the gripper action as we use different conventions...)
        action = adapt_action(prediction)

        # Execute the action
        obs, _, _, _ = env.step(action)
        env.render()
        total_steps += 1

        # Check for success...
        if env.objects:
            obj = env.objects[0]
            body_id = env.sim.model.body_name2id(obj.root_body)
            obj_xy = env.sim.data.body_xpos[body_id][:2]
            target_xy = np.asarray(settings.drop_point, dtype=float)[:2]
            error = float(np.linalg.norm(obj_xy - target_xy))

            if error < settings.final_tolerance:
                # If we are within the tolerance it means we succeded!
                break

    return (error, total_steps)

def prepare_rgb(obs: dict):

    # Get the obs dict key corresponding to the camera.
    camera_obs_key = settings.dataset_camera_observation_key

    # Check if the key is valid.
    if camera_obs_key not in obs:
        raise KeyError(f"Camera observation key: {camera_obs_key} is missing from the observation dict! Available keys: {list(obs)}")

    # Ensure the value is an np.ndarray
    image = np.asarray(obs[camera_obs_key])

    # Check for image format/shape.
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        # If the image was passed with the pixels' range of [0, 1], we rescale the range.
        if np.max(image) <= 1.0:
            image *= 255.0

        # Clip for safety...
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Grab the expected image format (H, W, 3)
    expected = (settings.dataset_image_height, settings.dataset_image_width, 3)

    # Last check on image format...
    if image.shape != expected:
        raise ValueError(f"Expected image shap: {expected}, got {image.shape}")

    return np.ascontiguousarray(image)


def adapt_action(prediction: np.ndarray):

    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    if prediction.shape != (7,) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"The OpenVLA model returned a bad action: {prediction}")

    # The predicted action gets clipped into the [-1.0, 1.0] range.
    prediction[:6] = np.clip(prediction[:6], -1.0, 1.0)

    # Use a threshold for deciding the action on the gripper.
    prediction[-1] = 1.0 if prediction[-1] >= 0.5 else 0.0

    return prediction

def read_observation(env: "SingleArmEnv") -> dict:

    try:
        return env._get_observations(force_update=True)
    except TypeError:
        return env._get_observations()

if __name__ == "__main__":
    main()