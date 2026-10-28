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
from custom_osc import install_prob3_custom_osc
from homing import prepare_episode
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from switch_controller import switch_arm_controller
from vla_client import OpenVLAClient

def get_init_conditions(hdf5_path: Path, demo_name, osc_config: dict | None = None) -> dict:
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
        #render_camera="sideview",
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

    return env

def main():
    # Parse command line arguments for overfit evaluation.
    parser = argparse.ArgumentParser(description="Valutazione online in simulazione Robosuite.")
    parser.add_argument("--overfit_hdf5", type=str, default=None, help="Path al file HDF5 per riprodurre le posizioni esatte di overfit")
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
    if args.overfit_hdf5:
        hdf5_path = Path(args.overfit_hdf5).resolve()
        with h5py.File(hdf5_path, "r") as f:
            overfit_demos = sorted(f["data"].keys())[: args.max_demos]
        n_tasks = len(overfit_demos)
        print(f"[OVERFIT MODE] Loaded {n_tasks} initial positions from: {hdf5_path.name}")
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
    n_successes = 0

    # 4. Instantiate the OpenVLAClient.
    client = OpenVLAClient(
        server_url=settings.server_url,
        unnorm_key=settings.unnorm_key,
        timeout=settings.server_timeout,
    )

    for task_id in range(n_tasks):
        if args.overfit_hdf5 and overfit_demos:
            demo_name = overfit_demos[task_id]
            print(f"[OVERFIT] Executing stored episode: {demo_name}")

        initial_conditions = get_init_conditions(Path(args.overfit_hdf5), demo_name)
        print("="*60)
        print(initial_conditions)
        print("="*60)

        env = create_env(joint_config, initial_conditions)
        observation = read_observation(env)
        env.render()
        prepare_episode(env, joint_config, osc_config)
        success = False
        print(f"\n[Task-{task_id}]")

        if env.robots[0].controller.name != "OSC_POSE":
            raise RuntimeError("OSC_POSE controller not active after homing!")

        if env.robots[0].action_dim != 7:
            raise RuntimeError(f"Expected robot action_dim = 7, got {env.robots[0].action_dim}")

        t0 = time.monotonic()

        input("Continue...")
        error, total_steps = execute_task(env, observation, client)
        elapsed = time.monotonic() - t0

        if error < settings.final_tolerance:
            success = True
            n_successes += 1

        print(f"The task was a {'SUCCESS' if success else 'FAIL'}, it used {total_steps} steps and lasted a total of {elapsed: .1f} seconds.")

        results.append((error, total_steps))

    print_final_report(n_successes, n_tasks, results)


def print_final_report(n_successes: int, n_total_tasks: int, results: list[tuple[float, int]]) -> None:
    """This function prints the final report after executing N tasks.

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


def execute_task(env: "SingleArmEnv", obs: dict, client: OpenVLAClient, max_steps: int = 1000) -> tuple[float, int]:
    """Executes the pick and place task.

    Args:
        env (SingleArmEnv): The simulation environment.
        obs (dict): The observation dictionary.
        client (OpenVLAClient): The OpenVLAClient to interrogate for the next action.
        max_steps (int, optional): The maximum number of steps our simulation should last. Defaults to 1000.

    Returns:
        tuple[float, int]: A tuple where the first element is a float representing the final error norm,
                           the second element indicates the number of steps the simulation lasted.
    """

    # Counter and flag variables.
    total_steps = 0
    error = 1.0
    last_pos = np.zeros(3)
    iterations = 0
    loop_times = []

    for _ in range(max_steps):
        start = time.time()
        # Prepare the image in the correct expected format.
        image = prepare_rgb(obs)

        # Make the model predict the next action.
        prediction = client.predict(image, settings.dataset_task)

        # Adapt predicted action to robot controller bounds.
        action = adapt_action(prediction)

        # Measure what happens when the action is executed.
        robot = env.robots[0]
        eef_before = env.sim.data.site_xpos[robot.eef_site_id].copy()

        #print(f"Actual z sent to robot: {action[2]}")
        #print(f"Actual eef z: {robot._hand_pos[2]}")
        next_obs, _, _, _ = env.step(action)
        env.render()

        eef_after = env.sim.data.site_xpos[robot.eef_site_id].copy()

        # Actual end-effector displacement in meters.
        eef_movement = float(np.linalg.norm(eef_after - eef_before))

        # How much has the image changed.
        next_image = prepare_rgb(next_obs)
        image_delta = float(np.mean(np.abs(next_image.astype(np.float32) - image.astype(np.float32))))

        obs = next_obs
        total_steps += 1

        # Check for success.
        if env.objects:
            obj = env.objects[0]
            body_id = env.sim.model.body_name2id(obj.root_body)
            obj_xy = env.sim.data.body_xpos[body_id][:2]
            obj_pos = env.sim.data.body_xpos[body_id][:3]
            if iterations == 0:
                last_pos = obj_pos
            target_xy = np.asarray(settings.drop_point, dtype=float)[:2]
            error = float(np.linalg.norm(obj_xy - target_xy))

            if error < settings.final_tolerance:
                # If we are within the tolerance it means we succeeded.
                break

        #print(f"End-effector position: {env.robots[0]._hand_pos}")

        if total_steps % 50 == 0:
            finger_qpos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy()
            has_moved = False
            if np.linalg.norm(np.abs(last_pos - obj_pos)) > 1e-3:
                has_moved = True

            last_pos = np.array([obj_pos])
            '''
            print(
                f"\n[STEP {total_steps}]"
                f"\n  Action: {np.round(action, 4)}"
                f"\n  EEF movement: {eef_movement:.6f} m"
                f"\n  EEF Z: {eef_after[2]:.4f} m"
                f"\n  Finger qpos: {np.round(finger_qpos, 4)}"
                f"\n  Image delta: {image_delta:.4f}"
                f"\n  Center error: {error: .5f}"
                f"\n  Has the object moved: {has_moved}"
            )'''

        # Check for table contacts...
        table_geoms = env.mujoco_arena.table_collision
        gripper_geoms = env.robots[0].gripper.important_geoms

        if env.check_contact(geoms_1=gripper_geoms, geoms_2=table_geoms):
            print("CONTACT BETWEEN TABLE AND GRIPPER GEOMS!")

        iterations += 1
        elapsed = time.time() - start
        loop_times.append(elapsed)

        if len(loop_times) % 10 == 0:
            avg_time = sum(loop_times[-10:]) / 10
            actual_hz = 1.0 /avg_time
            print(f"Loop time: {avg_time: .3f} | Actual Hz: {actual_hz: .2f}")

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

    return np.ascontiguousarray(image)


def adapt_action(prediction: np.ndarray):
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    if prediction.shape != (7,) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"The OpenVLA model returned a bad action: {prediction}")

    #print(f"Model output z: {prediction[2]}")

    # The predicted action gets clipped into the [-1.0, 1.0] range.
    prediction[:6] = np.clip(prediction[:6], -1.0, 1.0)
    #print(f"De-normalized z: {prediction[2]}")

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