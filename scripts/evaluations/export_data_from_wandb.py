import json
import re
import sys
from pathlib import Path

import wandb

ADAPTER_ROOT = Path(__file__).resolve().parents[2] / "adapters/openvla"

def main():
    # Initialize the wandb API
    api = wandb.Api()

    try:
        entity = api.default_entity
    except Exception:
        print("[Error] You must login to W&B first!")
        entity = api.default_entity

    print("Retrieving projects...")
    projects = api.projects(entity=entity)
    project_list = [p.name for p in projects]

    if not project_list:
        print("No project was found for this account!")
        sys.exit()

    print("Select the project to export:")
    for i, proj_name in enumerate(project_list, 1):
        print(f"[{i}] {proj_name}")

    proj_choice = menu_choice({"input": "Choose a project: ", "error": "You chose an invalid project!"}, len(project_list))
    selected_proj = project_list[proj_choice]

    print(f"Retrieving run for project: {selected_proj}")
    runs = api.runs(f"{entity}/{selected_proj}")
    run_list = [(r.id, r.name) for r in runs]

    print("Select a run (the names are partly truncated for easier reading): ")
    for i, (run_id, run_name) in enumerate(run_list, 1):
        print(f"[{i}] {run_name} (ID: {run_id})")

    run_choice = menu_choice({"input": "Choose a run (0 to download everything): ", "error": "You chose an invalid run!"}, len(run_list))

    n_downloads = 0
    if run_choice == -1:
        n_downloads = len(run_list)
        for i in range(n_downloads):
            data, selected_run_name = download_data(i, run_list, api, entity, selected_proj)
            save_data(selected_run_name, data)
    else: 
        data, selected_run_name = download_data(run_choice, run_list, api, entity, selected_proj)
        save_data(selected_run_name, data)

def download_data(run_choice, run_list, api, entity, selected_proj):

    selected_run_id, selected_run_name = run_list[run_choice]
            
    print(f"Downloading data for run: {selected_run_name}...")
    run = api.run(f"{entity}/{selected_proj}/{selected_run_id}")

    history_scan = run.scan_history()
    history_data = list(history_scan)

    return (history_data, selected_run_name)

def save_data(folder: str, data) -> None:

    save_path = Path(ADAPTER_ROOT) / f"{folder}" / "log_data.json"   

    folder_path = save_path.parent

    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Path created for saving data: {save_path}")

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print("[Success] Data exported successfully!")
    print(f"File saved at: {save_path}")

def menu_choice(strings: dict, max_id: int) -> int:
    input_bad = True
    while input_bad:
        try:
            choice_proj = int(input(f"\n{strings['input']}"))
            if choice_proj < 0 or choice_proj > max_id:
                raise ValueError
            
            input_bad = False
        except Exception:
            print(f"{strings['error']}")

    return choice_proj - 1

if __name__ == "__main__":
    main()