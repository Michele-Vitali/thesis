import argparse
import io
import json
import multiprocessing
import random
import time
import traceback
from pathlib import Path

import h5py
import mujoco
import numpy as np
import requests
import robosuite as suite
from matplotlib import pyplot as plt
from PIL import Image
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from scripts.evaluations.vla_client import OpenVLAClient
from scripts.evaluations.vla_server import DeployConfig, run_server_with_cfg
from scripts.expert import settings
from scripts.expert.action_utils import process_osc_action
from scripts.expert.custom_osc import install_prob3_custom_osc
from scripts.expert.homing import prepare_episode
from scripts.metrics.plot_metrics import plot_stacked_success_rate


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_init_conditions(hdf5_path: Path, demo_name) -> dict:
    initial_conditions = {"init_obj_pos": None, "init_obj_quat": None, "init_obj_size": None}
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_name]
        initial_conditions["init_obj_pos"] = demo.attrs['object_initial_position_m']
        initial_conditions["init_obj_quat"] = np.asarray(demo.attrs['object_initial_quat_wxyz'])
        initial_conditions["init_obj_size"] = demo.attrs['object_size_parameters_m']
    return initial_conditions

def create_env(joint_config: dict | None = None, init_cond: dict | None = None) -> SingleArmEnv:
    env = suite.make(
        env_name=settings.env_name,
        init_conditions=init_cond,
        robots=settings.robot,
        gripper_types=settings.gripper_types,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=settings.dataset_camera_names,
        camera_heights=settings.dataset_image_height,
        camera_widths=settings.dataset_image_width,
        camera_depths=False,
        control_freq=settings.dataset_fps,
        controller_configs=joint_config,
        hard_reset=False,
        horizon=1000,
    )

    table_z = float(env.mujoco_arena.table_top_abs[2])
    cam_id = env.sim.model.camera_name2id("sideview")
    env.sim.model.cam_pos[cam_id] = [0.0, -0.6, table_z]
    env.sim.model.cam_quat[cam_id] = [0.7071068, 0.7071068, 0.0, 0.0]

    context = getattr(env.sim, "_render_context_offscreen", None)
    if context is not None:
        context.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    env.reset()
    env.sim.forward()
    return env

def wait_for_server(process, timeout=600):
    url = f"{settings.server_url}/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if not process.is_alive():
            raise RuntimeError(f"OpenVLA Server crashed: {process.exitcode}")
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200 and response.json().get("status") == "OK":
                print("[SERVER] Server ready to receive requests!")
                return
        except (requests.RequestException, ValueError, RuntimeError):
            pass
        time.sleep(1)
    raise TimeoutError("Timeout avvio server OpenVLA!")

def resolve_deploy_config(path: Path) -> DeployConfig:
    if (path / "adapter_config.json").is_file():
        return DeployConfig(openvla_path=settings.openvla_path, adapter_path=path.resolve())

    if (path /"config.json").is_file():
        return DeployConfig(openvla_path=path.resolve(), adapter_path=None)

    raise FileNotFoundError(f"Neither adapter nor merge model were found at: {path}")

def main():
    target_seed = settings.base_env_rand_test_seed
    
    parser = argparse.ArgumentParser(description="Valutazione online automatizzata su N modelli.")
    parser.add_argument("--checkpoint_dir", type=str, default=settings.adapters_root, help="Specifies the checkpoints dir where we have our adapters.")
    parser.add_argument("--hdf5", type=str, default=None, help="Path HDF5 per overfit")
    parser.add_argument("--max_demos", type=int, default=5, help="Numero episodi")
    parser.add_argument("--n_models", type=int, default=1, help="Numero di ultimi modelli da valutare")
    parser.add_argument("--queue", type=str, default=None,
                        help="JSON generato da analyze_checkpoints.py: valuta i modelli della coda (ignora --n_models)")
    parser.add_argument("--categories", nargs="+", default=["SICURO", "POTREI"],
                        choices=["SICURO", "POTREI", "NON DOVREI", "INCOMPLETO"],
                        help="Categorie della coda da valutare (default: SICURO POTREI)")
    parser.add_argument("--skip_done", action=argparse.BooleanOptionalAction, default=True,
                        help="Con --queue, salta i modelli che hanno già evaluation_results.json (default: sì)")
    args = parser.parse_args()

    install_prob3_custom_osc()
    joint_config = load_controller_config(default_controller="JOINT_POSITION")
    osc_config = load_controller_config(default_controller="OSC_POSE")
    osc_config.update(settings.osc_config)

    overfit_demos = []
    if args.hdf5:
        hdf5_path = Path(args.hdf5).resolve()
        with h5py.File(hdf5_path, "r") as f:
            overfit_demos = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[-1]))[: args.max_demos]
        n_tasks = len(overfit_demos)
    else:
        n_tasks = int(input("How many tasks you want to evaluate? "))

    # Find the models
    script_dir = Path(__file__).resolve().parent
    checkpoints_base_dir = script_dir / "../../runs/openvla"
    
    if args.queue:
        models = load_queue(Path(args.queue), args.categories, checkpoints_base_dir, args.skip_done)
    else:
        models = sorted(
            [p for p in checkpoints_base_dir.iterdir() if p.is_dir() and "chkpt" in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:args.n_models]
    
    if not models:
        print("No model to evaluate was found" + (f" in the queue {args.queue}" if args.queue else f" at: {checkpoints_base_dir}"))
        return

    print("=" * 60)
    print(f"Starting evaluation on {len(models)} models")
    print("=" * 60)

    # Models loop
    for model_path in models:
        print(f"\n>>> Starting evalaution of: {model_path.name} <<<")
        
        # 1. Start the server in an independent process
        current_config = DeployConfig(adapter_path=model_path)
        server_process = multiprocessing.Process(target=run_server_with_cfg, args=(current_config,))
        server_process.start()

        try:
            wait_for_server(server_process)
            
            client = OpenVLAClient(
                server_url=settings.server_url,
                unnorm_key=settings.unnorm_key,
                timeout=settings.server_timeout,
            )
            
            results = []
            n_successes = 0

            # Task loop
            for task_id in range(n_tasks):
                current_seed = target_seed + task_id
                set_global_seed(current_seed)
                
                if args.hdf5 and overfit_demos:
                    demo_name = overfit_demos[task_id]
                    with h5py.File(hdf5_path, "r") as f:
                        expert_steps = len(f['data'][demo_name]['obs']['agentview_rgb'])
                    initial_conditions = get_init_conditions(hdf5_path, demo_name)
                    env = create_env(joint_config, initial_conditions)
                else:
                    env = create_env(joint_config)

                prepare_episode(env, joint_config, osc_config)
                observation = read_observation(env)

                t0 = time.monotonic()
                error, total_steps, placed = execute_task(env, observation, client)
                elapsed = time.monotonic() - t0

                success = bool(placed)
                if success: 
                    n_successes += 1

                print(f"[Task-{task_id}] {'SUCCESS' if success else 'FAIL'} | Err: {error:.4f} | Steps: {total_steps} | Time: {elapsed:.1f}s")

                res = {
                    "task_id": task_id,
                    "error": error,
                    "success": success,
                    "model_steps": total_steps,
                    "model": model_path.name,
                    "duration": elapsed,
                }
                if args.hdf5 and overfit_demos:
                    res["expert_steps"] = expert_steps
                    res["step_ratio"] = round(expert_steps/total_steps, 3) if total_steps > 0 else 0
                
                results.append(res)

            # Save the results for this model
            save_results(results, model_path)
            print_final_report(n_successes, n_tasks)
            plot_results(results, model_path)
            client.close()

        finally:
            # 2. Destroy the model to flush the VRAM
            print(f"Chiudo il server per liberare VRAM ({model_path.name})...")
            server_process.kill()
            server_process.join()
            time.sleep(3)  # Pause for safety

def load_queue(queue_path: Path, categories: list, checkpoints_base_dir: Path, skip_done: bool = True) -> list:
    """Legge la coda creata da analyze_checkpoints.py e ritorna i Path degli adapter da valutare,
    già in ordine di priorità (SICURO -> POTREI -> ...)."""
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)

    wanted = set(categories)
    models = []
    for entry in queue["models"]:
        if entry["verdict"] not in wanted:
            continue

        path = Path(entry["adapter_path"]) if entry.get("adapter_path") else None
        if (path is None or not path.is_dir()) and entry.get("name"):
            path = checkpoints_base_dir / entry["name"]      # fallback se la coda arriva da un'altra macchina
        if path is None or not path.is_dir():
            print(f"[Queue] Checkpoint non trovato su disco, salto: {entry['run']} step {entry['step']}")
            continue
        if skip_done and (path / "evaluation_results.json").is_file():
            print(f"[Queue] Già valutato, salto: {path.name}")
            continue

        models.append(path)
        print(f"[Queue] {entry['verdict']:<7} {path.name}")

    print(f"[Queue] {len(models)} modelli da valutare ({', '.join(categories)})")
    return models

def execute_task(env: "SingleArmEnv", obs: dict, client: OpenVLAClient, max_steps: int = 400) -> tuple[float, int, bool]:
    total_steps = 0
    error = 1.0
    #previous_action = np.zeros(env.action_dim, dtype=float)
    placed = False
    stable_steps = 0

    for _ in range(max_steps):
        try:
            image1 = prepare_rgb(obs, jpeg=True)
            image2 = prepare_rgb(obs, jpeg=False)
            prediction1 = client.predict(image1, settings.dataset_task)
            prediction2 = client.predict(image2, settings.dataset_task)

            print(np.abs(prediction1 - prediction2).max())

            action = process_osc_action(prediction1) # It was (prediction, previous_action)
            #previous_action = action.copy()

            next_obs, _, _, _ = env.step(action)
            if env.has_renderer:
                env.render()
            obs = next_obs
            total_steps += 1

            if env.objects:
                obj = env.objects[0]
                body_id = env.sim.model.body_name2id(obj.root_body)
                obj_xy = env.sim.data.body_xpos[body_id][:2]
                obj_z = env.sim.data.body_xpos[body_id][2]
                target_xy = np.asarray(settings.drop_point, dtype=float)[:2]
                table_z = float(env.mujoco_arena.table_top_abs[2])
                
                error = float(np.linalg.norm(obj_xy - target_xy))
                z_error = abs(obj_z - (table_z + obj.size[2]))

                if error < settings.online_eval_tolerance and z_error < 0.002:
                    stable_steps += 1
                    if stable_steps >= 10:
                        placed = True
                        break
                else:
                    stable_steps = 0

        except requests.RequestException as e:
            print(f"[Task] Network error at step {total_steps}: {e}")
            break
        except Exception:
            print(f"[Task] Unexpected error at step {total_steps}:")
            traceback.print_exc()
            break
            
    env.close()
    return (error, total_steps, placed)

def read_observation(env: "SingleArmEnv") -> dict:
    try: return env._get_observations(force_update=True)
    except TypeError: return env._get_observations()

def prepare_rgb(obs: dict, jpeg: bool = True):
    key = settings.dataset_external_camera_observation_key
    image = np.asarray(obs[key])
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if np.max(image) <= 1.0: 
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    image = np.flipud(image).copy()

    if jpeg:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=settings.eval_jpeg_quality)
        image = np.asarray(Image.open(buf).convert("RGB"))
    
    # Important! Add the center crop!
    out_size = 224
    area_scale = 0.9
    side = np.sqrt(area_scale)
    off = (1.0 - side) / 2.0
    pil = Image.fromarray(image).resize((out_size, out_size), Image.LANCZOS)
    box = (off * out_size, off * out_size, (off + side) * out_size, (off + side) * out_size)
    pil = pil.resize((out_size, out_size), Image.BILINEAR, box=box)
    
    return np.ascontiguousarray(np.asarray(pil))

def save_results(results: list, model_path: Path)-> None:
    with open(model_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def plot_results(results: list, model_path: Path) -> None:
    successes = sum(1 for x in results if x['success'] is True)
    fails = sum(1 for x in results if x['success'] is False)
    fig = plot_stacked_success_rate(successes, fails)
    fig.savefig(model_path / "success_rate.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

def print_final_report(n_successes, n_total):
    print("=" * 60)
    print(f"Success rate: {n_successes/n_total * 100: .2f}% ({n_successes}/{n_total})")
    print("=" * 60)

if __name__ == "__main__":
    main()