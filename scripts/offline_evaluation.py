import os

os.environ["USE_TF"] = "0"  # Disables TensorFlow (TF) warnings, used since we do not utilize TF here but it gets imported via HF dependencies...

import multiprocessing
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import settings
from offline_metrics import compute_metrics, save_metrics
from vla_client import OpenVLAClient
from vla_server import DeployConfig, run_server_with_cfg


def wait_for_server(process, timeout=600):
    """Wait until the OpenVLA server responds with OK.

    Args:
        process (_type_): _description_
        timeout (int, optional): The maximum time that we should wait before calling a failure. Defaults to 600.
    """

    url = f"{settings.server_url}/health"

    start = time.monotonic()

    while time.monotonic() - start < timeout:
        if not process.is_alive():
            raise RuntimeError(f"OpenVLA server crashed: {process.exitcode}")

        try:
            response = requests.get(url, timeout=2)

            if response.status_code == 200 and response.json().get("status") == "OK":
                print("[SERVER] Ready!")
                return
        except (requests.RequestException, ValueError, RuntimeError):
            pass

        time.sleep(1)

    raise TimeoutError("OpenVLA server startup timeout!")

def main():

    try:
        script_dir = Path(__file__).parent.resolve()
        thesis_root = script_dir.parent
        hdf5_root = thesis_root / "hdf5_data"

        # Grab the available datasets and order them from the most recent to the oldest.
        available_datasets = sorted([path for path in hdf5_root.iterdir() if path.is_dir() and (path / "splits" / "val.hdf5").is_file()],
                                    key=lambda p: p.stat().st_mtime,
                                    reverse=True)

        if not available_datasets:
            raise RuntimeError("No dataset with val.hdf5 was found!")

        # Print the available datasets in the directory
        print("="*60)
        print("Available datasets:")
        for i, path in enumerate(available_datasets):
            print(f"[{i}] {path.name}")

        # Make the user choos the dataset to use...
        choice = input("\nChosen dataset index (Enter => Choose the latest): ").strip()
        index = int(choice) if choice else 0

        if not 0 <= index < len(available_datasets):
            raise ValueError("Invalid dataset index chosen!")

        # Grab the dataset directory and the validation file path
        dataset_dir = available_datasets[index]
        val_file = dataset_dir / "splits" / "val.hdf5"

        print(f"Selected dataset: {dataset_dir.name}")
        print(f"Validation file: {val_file}")
        print("="*60)

        # Compute the models' directory
        checkpoints_base_dir = script_dir / Path("../runs/openvla")

        n_models_to_test = int(input("How many models you want to test (the latest N will be chosen...)? "))
        if n_models_to_test <= 0:
            raise ValueError("At least 1 model must be testes!")

        if checkpoints_base_dir.exists():
            models = sorted([p for p in checkpoints_base_dir.iterdir() if p.is_dir()],
                            key=lambda p: p.stat().st_mtime,
                            reverse=True)[:n_models_to_test]

            models_name = [p.name for p in models]
            models = [str(p) for p in models]
        else:
            raise ValueError("The checkpoints' folder does not exist!")        

        print(f"Models found to test: {len(models)}")
        if not models:
                raise ValueError("No models to be tested were found!")

        print("="*60)
        print("These models will be tested:")
        for i, model in enumerate(models_name):
            print(f"[Model-{i}] {model}")
        print("="*60)

        all_models_results = {}
        # Loop to test every model.
        for i, model_path in enumerate(models):
            
            # Grab the last part of the path (the unique model name...)
            print("="*60)
            print(f"[Model-{i}] {models_name[i]}")
            print("="*60)

            # Create a new config for the current model
            current_config = DeployConfig(openvla_path=model_path)

            # Start a new process for the model, so when it finishes all the VRAM gets flushed.
            server_process = multiprocessing.Process(target=run_server_with_cfg, args=(current_config,))
            server_process.start()

            try:
                wait_for_server(server_process)

                # Create a client to allow the model to predict actions...
                # Note: Server must already be active!
                client = OpenVLAClient()

                # We use some metrics for the arm and some for the gripper.
                # Arm: Axis-MAE, MAE translation, MAE rotation, RMSE, Action direction cosine, Zero-action baseline.
                # Gripper: Accuracy, Precision closed, Recall closed, F1 closed, F1 open, Macro F1, Confusion Matrix.
                # Note: For the gripper we use a threshold of >= 0.5 to be considered 1.0!
                all_gt_actions = []
                all_pred_actions = []

                try:
                    val_file = dataset_dir / "splits" / "val.hdf5"

                    if not val_file.is_file():
                        raise FileNotFoundError(val_file)

                    with h5py.File(val_file, "r") as f:
                        # Grab and sort the contained episodes
                        demos = sorted(f["data"].keys())
                        print(f"Total number of validation episodes: {len(demos)}")

                        # Loop through all episodes
                        for demo_name in demos:
                            # Grab the correct data group in the episodes file
                            group = f["data"][demo_name]

                            # Grab images, expert actions and language instruction
                            images = group["obs"]["agentview_rgb"]
                            gt_actions = group["actions_openvla"]

                            language_instruction = group.attrs["language_instruction"]

                            if isinstance(language_instruction, bytes):
                                language_instruction = language_instruction.decode("utf-8")

                            num_frames = len(gt_actions)

                            if len(images) != num_frames:
                                raise ValueError("We have fewer/more images than the number of frames!")

                            print(f"[{demo_name}] Evaluating: {num_frames} frames.")

                            # Now loop through every episode (1 episode = N steps or frames)
                            for t in range(num_frames):
                                # 1. Read the observation at timestep t.
                                image = np.asarray(images[t], dtype=np.uint8)

                                # 2. Read the expert action at the SAME timestep t.
                                gt_action = np.asarray(gt_actions[t], dtype=np.float32)

                                # 3. Ask the model to predict an action...
                                pred_action = client.predict(image=image, instruction=language_instruction)

                                # 4. Check the prediction.
                                pred_action = np.asarray(pred_action, dtype=np.float32).reshape(-1) # Flatten it.

                                if gt_action.shape != (7,):
                                    raise ValueError(f"{demo_name}, frame {t}, invalid GT shape: {gt_action.shape}")

                                if pred_action.shape != (7,):
                                    raise ValueError(f"{demo_name}, frame {t}, invalid pred shape: {pred_action.shape}")

                                # 5. Store both actions in the previously declared lists.
                                all_gt_actions.append(gt_action)
                                all_pred_actions.append(pred_action)

                            print(f"[OK] {demo_name} completed.")

                        # After all episodes have been analyzed we compute the metrics over them...
                        all_gt_actions = np.asarray(all_gt_actions, dtype=np.float32)
                        all_pred_actions = np.asarray(all_pred_actions, dtype=np.float32)

                        metrics = compute_metrics(all_gt_actions, all_pred_actions)

                        results_dir = thesis_root / "evaluations" / models_name[i]
                        split_name = "val"
                        save_metrics(models_name[i], dataset_dir.name, split_name, metrics, results_dir)
                        all_models_results[models_name[i]] = metrics

                        # Print the final report!
                        print("\n" + "=" * 60)
                        print(f"Evaluation results for model: {models_name[i]}")
                        print("="*60)

                        print(f"Evaluated frames: {metrics['num_frames']}")
                        print(f"Arm MAE: {metrics['arm']['mae']}")
                        print(f"Translation MAE: {metrics['arm']['translation_mae']}")
                        print(f"Rotation MAE: {metrics['arm']['rotation_mae']}")
                        print(f"Gripper accuracy: {metrics['gripper']['accuracy']}")
                        print(f"Gripper Macro F1: {metrics['gripper']['macro_f1']}")

                except (FileNotFoundError, ValueError) as e:
                    print(f"An exception occured while evaluating a model! Error: {e}")
                finally:
                    client.close()

            except KeyboardInterrupt:
                print("[Child] Process terminated by the user...")
                raise KeyboardInterrupt
            finally:
                server_process.kill()
                server_process.join()   # Waits for the child process to be killed ...

        # Now we plot the graphs for easier model comparison...
        discrete_metrics = {
            "Model": []
        }


    except KeyboardInterrupt:
        print("Process terminated by the user...")


if __name__ == "__main__":
    main()