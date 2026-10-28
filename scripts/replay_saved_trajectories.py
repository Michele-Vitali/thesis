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
        initial_conditions["init_obj_quat"] = init_obj_quat
        initial_conditions["init_obj_size"] = demo.attrs['object_size_parameters_m']

    return initial_conditions

def get_saved_actions(hdf5_path: Path, demo_name: str) -> np.ndarray:
    """Extracts the recorded actions from the HDF5 file for a specific episode."""
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_name]
        return np.asarray(demo["actions_openvla"]).copy()

def create_env(joint_config: dict | None = None, init_cond: dict | None = None) -> SingleArmEnv:
    # 3. Create the actual env.
    env = suite.make(
        env_name=settings.env_name,
        init_conditions=init_cond,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,  
        has_offscreen_renderer=True,  
        use_camera_obs=True,  
        camera_names=settings.dataset_camera_names,
        #render_camera="sideview",
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=joint_config,
        hard_reset=False,  
        horizon=1500,  
        reward_shaping=True,  
    )

    # 3.1 Remove rangefinders visualization.
    context = getattr(env.sim, "_render_context_offscreen", None)

    if context is not None:
        context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    return env

def main():
    parser = argparse.ArgumentParser(description="Replay traiettorie salvate da dataset HDF5.")
    parser.add_argument("--hdf5", type=str, required=True, help="Path al file HDF5 per riprodurre le traiettorie esatte")
    parser.add_argument("--max_demos", type=int, default=5, help="Numero di episodi da testare in overfit")
    args = parser.parse_args()

    # 1. Initialize our custom OSC_POSE controller.
    install_prob3_custom_osc()

    # 2. Grab the controllers' configurations.
    joint_config = load_controller_config(default_controller="JOINT_POSITION")
    osc_config = load_controller_config(default_controller="OSC_POSE")
    osc_config.update(settings.osc_config)

    # Load stored demo names if an HDF5 path is provided.
    hdf5_path = Path(args.hdf5).resolve()
    with h5py.File(hdf5_path, "r") as f:
        overfit_demos = sorted(f["data"].keys())[: args.max_demos]
    n_tasks = len(overfit_demos)
    print(f"Loaded {n_tasks} initial positions from: {hdf5_path.name}")

    print("=" * 60)
    print("Starting HDF5 trajectories replay...")
    print("=" * 60)

    results = []
    n_successes = 0

    for task_id in range(n_tasks):
        demo_name = overfit_demos[task_id]
        print(f"Executing stored episode: {demo_name}")

        initial_conditions = get_init_conditions(hdf5_path, demo_name)
        print("="*60)
        print(initial_conditions)
        print("="*60)
        
        # Load the action sequence for the episode
        action_sequence = get_saved_actions(hdf5_path, demo_name)

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

        # Pass the pre-recorded action sequence instead of a model client
        error, total_steps = execute_task(env, action_sequence)
        env.close()
        elapsed = time.monotonic() - t0

        if error < settings.final_tolerance:
            success = True
            n_successes += 1

        print(f"The task was a {'SUCCESS' if success else 'FAIL'}, it used {total_steps} steps and lasted a total of {elapsed: .1f} seconds.")

        results.append((error, total_steps))

    print_final_report(n_successes, n_tasks, results)


def print_final_report(n_successes: int, n_total_tasks: int, results: list[tuple[float, int]]) -> None:
    print("=" * 60)
    print(f"OSC final report: {n_successes}/{n_total_tasks} successful tasks.")
    print("=" * 60)
    np_results = np.array(results)
    averages = np.mean(np_results, axis=0)
    print(f"Average final error from the drop point: {averages[0]: 0.5f}")
    print(f"Average number of steps per episode: {averages[1]: 0.5f}")


def execute_task(env: "SingleArmEnv", action_sequence: np.ndarray) -> tuple[float, int]:
    """Executes the pick and place task using a pre-recorded sequence of actions."""
    
    total_steps = 0
    error = 1.0
    last_pos = np.zeros(3)
    iterations = 0

    max_steps = action_sequence.shape[0]

    for step_idx in range(max_steps):
        # Extract the exact action logged during dataset creation
        action = action_sequence[step_idx]

        next_obs, _, _, _ = env.step(action)
        env.render()

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

            # Optional: early stopping inside replay if tolerance is met
            # if error < settings.final_tolerance:
            #     break

    return (error, total_steps)

def read_observation(env: "SingleArmEnv") -> dict:
    try:
        return env._get_observations(force_update=True)
    except TypeError:
        return env._get_observations()


if __name__ == "__main__":
    main()