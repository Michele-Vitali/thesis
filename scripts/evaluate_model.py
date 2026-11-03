import argparse
import time
from pathlib import Path

import custom_env
import custom_robot
import h5py
import mujoco
import numpy as np
import robosuite as suite
import settings
from action_utils import process_osc_action
from custom_osc import install_prob3_custom_osc
from homing import prepare_episode
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from switch_controller import switch_arm_controller
from vla_client import OpenVLAClient
import json
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from plot_metrics import plot_stacked_success_rate


def get_init_conditions(hdf5_path: Path, demo_name) -> dict:
    initial_conditions = {
        "init_obj_pos": None,
        "init_obj_quat": None,
        "init_obj_size": None
    }
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_name]

        initial_conditions["init_obj_pos"] = demo.attrs['object_initial_position_m']
        init_obj_quat = np.asarray(demo.attrs['object_initial_quat_wxyz'])
        initial_conditions["init_obj_quat"] = init_obj_quat #[[1,2,3,0]]
        initial_conditions["init_obj_size"] = demo.attrs['object_size_parameters_m']

    return initial_conditions

def create_env(joint_config: dict | None = None, init_cond: dict | None = None) -> SingleArmEnv:
    # 3. Create the actual env.
    env = suite.make(
        env_name=settings.env_name,
        init_conditions=init_cond,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,  # On-screen renderer, useful for debug, but slow for dataset gathering.
        has_offscreen_renderer=True,  # Necessary for the dataset.
        use_camera_obs=True,  # Also for dataset.
        camera_names=settings.dataset_camera_names,
        render_camera="sideview",
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=joint_config,
        hard_reset=False,  # Avoids segfault on macos or glfw error on Linux (per docs).
        horizon=1000,  # So we are sure that all the pick and places terminate.
        reward_shaping=True,  # Not useful in our case but had implemented it anyway.
    )

    # Sposta la telecamera sideview a livello del tavolo.
    table_z = float(env.mujoco_arena.table_top_abs[2])
    cam_id = env.sim.model.camera_name2id("sideview")
    
    # Posizionala sul lato (y=-0.6) a +5cm dal piano del tavolo.
    env.sim.model.cam_pos[cam_id] = [0.0, -0.6, table_z]
    
    # Ruotala per guardare dritto verso il centro del tavolo (asse +Y).
    env.sim.model.cam_quat[cam_id] = [0.7071068, 0.7071068, 0.0, 0.0]
    '''
    # Enable shadows (like in training!).
    model = env.sim.model
    if model.nlight > 0:
        model.light_castshadow[:] = int(settings.camera_shadows)
        active_lights = model.light_active.astype(bool)

        model.light_ambient[active_lights] = settings.camera_shadow_ambient
        model.light_diffuse[active_lights] = settings.camera_shadow_diffuse
    '''
    # 3.1 Remove rangefinders visualization.
    context = getattr(env.sim, "_render_context_offscreen", None)

    if context is not None:
        context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    env.reset()
    env.sim.forward()

    return env

def main():
    # Parse command line arguments for overfit evaluation.
    parser = argparse.ArgumentParser(description="Valutazione online in simulazione Robosuite.")
    parser.add_argument("--hdf5", type=str, default=None, help="Path al file HDF5 per riprodurre le posizioni esatte di overfit")
    parser.add_argument("--max_demos", type=int, default=5, help="Numero di episodi da testare in overfit")
    args = parser.parse_args()

    # First we must re-create the same environment we used in dataset gathering.
    # 1. Initialize our custom OSC_POSE controller.
    install_prob3_custom_osc()

    # 2. Grab the controllers' configurations.
    joint_config = load_controller_config(default_controller="JOINT_POSITION")
    osc_config = load_controller_config(default_controller="OSC_POSE")
    osc_config.update(settings.osc_config)

    # Load stored demo names if an HDF5 path for overfitting is provided.
    overfit_demos = []
    if args.hdf5:
        hdf5_path = Path(args.hdf5).resolve()
        with h5py.File(hdf5_path, "r") as f:
            overfit_demos = sorted(f["data"].keys())[: args.max_demos]
        n_tasks = len(overfit_demos)
        #print(f"[OVERFIT MODE] Loaded {n_tasks} initial positions from: {hdf5_path.name}")
    else:
        n_tasks = 0
        is_good = False
        while not is_good:
            try:
                n_tasks = int(input("How many tasks you want to evaluate? "))
                is_good = True
            except Exception:
                print("You must insert a valid number!")

    print("=" * 60)
    print("Starting OpenVLA model evaluation...")
    print("=" * 60)

    results = []
    base_results = []
    n_successes = 0

    # 4. Instantiate the OpenVLAClient.
    client = OpenVLAClient(
        server_url=settings.server_url,
        unnorm_key=settings.unnorm_key,
        timeout=settings.server_timeout,
    )

    for task_id in range(n_tasks):
        if args.hdf5 and overfit_demos:
            demo_name = overfit_demos[task_id]
            print(f"[OVERFIT] Executing stored episode: {demo_name}")

        with h5py.File(hdf5_path, "r") as f:
            demo = f['data'][f'demo_{task_id}']
            expert_steps = len(demo['obs']['agentview_rgb'][0])

        initial_conditions = get_init_conditions(Path(args.hdf5), demo_name)
        #print("="*60)
        #print(initial_conditions)
        #print("="*60)

        env = create_env(joint_config, initial_conditions)
        env.render()
        prepare_episode(env, joint_config, osc_config)
        observation = read_observation(env)

        success = False
        print(f"\n[Task-{task_id}]")

        if env.robots[0].controller.name != "OSC_POSE":
            raise RuntimeError("OSC_POSE controller not active after homing!")

        if env.robots[0].action_dim != 7:
            raise RuntimeError(f"Expected robot action_dim = 7, got {env.robots[0].action_dim}")

        t0 = time.monotonic()
        error, total_steps = execute_task(env, task_id, observation, client)
        elapsed = time.monotonic() - t0

        if error < settings.online_eval_tolerance:
            success = True
            n_successes += 1

        print(f"The task was a {'SUCCESS' if success else 'FAIL'}, it used {total_steps} steps and lasted a total of {elapsed: .1f} seconds.")
        print(f"The error was: {error: .5f} and the steps were: {total_steps}")

        results.append({
            "task_id": task_id,
            "expert_steps": expert_steps,
            "error": error,
            "success": success,
            "model_steps": total_steps,
            "step_ratio": round(expert_steps/total_steps, 3), # 3 decimal spots
            "model": client.model,
            "duration": elapsed,
            })

        base_results.append((error, total_steps))

    save_results(results)
    print_final_report(n_successes, n_tasks, base_results)
    plot_results(results)

def print_final_report(n_successes: int, n_total_tasks: int, results: list[tuple[float, int]]) -> None:
    """This function prints the final report after executing N tasks.

    Args:
        n_successes (int): The number of successes we achieved.
        n_total_tasks (int): The number of total tasks we simulated.
        results (list[tuple[float, int]]): A list of 'n_total_tasks' tuples, where each tuple contains the final error norm of the executed task and the number of total steps it has lasted.
    """

    print("=" * 60)
    print(f"OSC final report: {n_successes}/{n_total_tasks} successful tasks. -> Success rate: {n_successes/n_total_tasks * 100: .2f}")
    print("=" * 60)
    np_results = np.array(results)
    averages = np.mean(np_results, axis=0)
    print(f"Average final error from the drop point: {averages[0]: 0.5f}")
    print(f"Average number of steps per episode: {averages[1]: 0.5f}")

def execute_task(env: "SingleArmEnv", task_id: int,obs: dict, client: OpenVLAClient, max_steps: int = 400) -> tuple[float, int]:
    """Executes the pick and place task.

    Args:
        env (SingleArmEnv): The simulation environment.
        obs (dict): The observation dictionary.
        client (OpenVLAClient): The OpenVLAClient to interrogate for the next action.
        max_steps (int, optional): The maximum number of steps our simulation should last. Defaults to 400.

    Returns:
        tuple[float, int]: A tuple where the first element is a float representing the final error norm,
                           the second element indicates the number of steps the simulation lasted.
    """

    # Counter and flag variables.
    total_steps = 0
    error = 1.0

    # Inizializziamo l'azione precedente a zero per la rampa di smoothing
    previous_action = np.zeros(env.action_dim, dtype=float)

    stable_steps = 0

    for _ in range(max_steps):
        try:
            # Prepare the image in the correct expected format.
            image = prepare_rgb(obs)           

            try:
                # Make the model predict the next action.
                prediction = client.predict(image, settings.dataset_task)

                # Usiamo la STESSA IDENTICA FUNZIONE del data collector per filtrare e tagliare l'azione!
                action = process_osc_action(prediction, previous_action)
                
                # Aggiorniamo il buffer interno per il frame successivo
                previous_action = action.copy()

                next_obs, _, _, _ = env.step(action)
                env.render()

                obs = next_obs
                total_steps += 1

                # Check for success.
                if env.objects:
                    obj = env.objects[0]
                    body_id = env.sim.model.body_name2id(obj.root_body)
                    obj_xy = env.sim.data.body_xpos[body_id][:2]
                    obj_z = env.sim.data.body_xpos[body_id][2]
                    target_xy = np.asarray(settings.drop_point, dtype=float)[:2]
                    table_z = float(env.mujoco_arena.table_top_abs[2])
                    target_z = table_z + obj.size[2]
                    error = float(np.linalg.norm(obj_xy - target_xy))
                    z_error = abs(obj_z - target_z)

                    if error < settings.online_eval_tolerance and z_error < 0.0005:
                        stable_steps += 1

                        if stable_steps >= 10:
                            break

            except Exception as e:
                print(f"Communication error with OpenVLA server: {e}\nTerminating...")
                break
        except Exception as e:
            print("Trajectory terminated!")

    env.close()

    return (error, total_steps)

def prepare_rgb(obs: dict):
    # Get the obs dict key corresponding to the camera.
    camera_obs_key = settings.dataset_external_camera_observation_key

    # Check if the key is valid.
    if camera_obs_key not in obs:
        raise KeyError(f"Camera observation key: {camera_obs_key} is missing from the observation dict! Available keys: {list(obs)}")

    # Ensure the value is an np.ndarray.
    image = np.asarray(obs[camera_obs_key])

    # Check for image format/shape.
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        # If the image was passed with the pixels' range of [0, 1], we rescale the range.
        if np.max(image) <= 1.0:
            image *= 255.0

        # Clip for safety.
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Grab the expected image format (H, W, 3).
    expected = (settings.dataset_image_height, settings.dataset_image_width, 3)

    # Last check on image format.
    if image.shape != expected:
        raise ValueError(f"Expected image shape: {expected}, got {image.shape}")

    # IMPORTANT: Flip the image! Robosutie return upside-down images!!!
    image = np.flipud(image).copy()

    return np.ascontiguousarray(image)

# NOTA: La funzione adapt_action è stata rimossa completamente, delegata ad action_utils.py

def read_observation(env: "SingleArmEnv") -> dict:
    try:
        return env._get_observations(force_update=True)
    except TypeError:
        return env._get_observations()

def get_save_dir(model) -> Path:
    models_dir = Path(__file__).resolve().parent / "../runs/openvla"
    model_family = model.split("--")[1]

    parent_model_dir = None
    for item in models_dir.iterdir():
        if item.is_dir() and model_family in item.name:
            after_family = item.name.split("--", 1)[1]

            if "--" not in after_family:
                parent_model_dir = item
                break

    return parent_model_dir

def save_results(results: list)-> None:

    model = results[0]['model']
    parent_model_dir = get_save_dir(model)

    if parent_model_dir is not None:
        with open(parent_model_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results file saved to location: {parent_model_dir / 'evaluation_results.json'}")

def plot_results(results: list) -> None:

    model = results[0]['model']
    parent_model_dir = get_save_dir(model)

    successes = sum(1 for x in results if x['success'] is True)
    fails = sum(1 for x in results if x['success'] is False)
    total = successes + fails
    assert total == len(results), "More successes and fails than number of results!"

    fig = plot_stacked_success_rate(successes, fails)
    save_path = parent_model_dir / "success_rate.svg"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()