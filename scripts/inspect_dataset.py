import h5py
from pathlib import Path
import numpy as np

phases = ['approach', 'descend', 'grasp', 'lift', 'transport', 'lower', 'release']
phase_stats = {
    phase: {'num_frames': 0,'opened_grip_command': 0, 'closed_grip_command': 0} 
    for phase in phases
}

def main():
    file_path = Path(input("Insert the absolute file path of the dataset file: "))

    if not file_path.is_file():
        raise ValueError("The inserted path is not  valid *file* path!")

    if file_path.suffix not in ['.h5', '.hdf5']:
        raise ValueError("The inserted path does not point to a valid hdf5 file!")

    with h5py.File(file_path, "r") as f:
        print(f"Inspecting file: {f.filename}")
        
        # Access the data root
        data_group = f["data"]
        
        # Read global metadata
        print("=== GLOBAL METADATA ===")
        for k, v in data_group.attrs.items():
            print(f"{k}: {v}")
            
        demos = list(data_group.keys())
        total_episodes = len(demos)
        total_frames = 0
        lengths = []
        obj_positions = []
        obj_rotations = []
        
        print(f"\nTotal found episodes: {total_episodes}")
        
        # Analisi campionaria del primo episodio per capire la struttura
        if total_episodes > 0:
            first_demo = data_group[demos[0]]
            print(f"\n=== SAMPLE STRUCTURE({demos[0]}) ===")
            print(f"Language instruction: {first_demo.attrs.get('language_instruction')}")
            print(f"Manipulated object: {first_demo.attrs.get('object_name')} ({first_demo.attrs.get('object_type')})")
            print(f"Total episode frames: {first_demo.attrs.get('num_samples')}")
            
        # Scorriamo tutte le traiettorie per estrarre statistiche        
        for index, demo_name in enumerate(demos):
            demo = data_group[demo_name]
            obs = demo['obs']          

            analyze_demo(demo, obs)
            
            num_samples = demo.attrs["num_samples"]
            obj_positions.append(demo.attrs["object_initial_position_m"])
            obj_rotations.append(demo.attrs["object_initial_quat_wxyz"])
            lengths.append(num_samples)
            total_frames += num_samples
            
        print("\n=== TRAJECTORIES STATISTICS ===")
        print(f"Total saved frames: {total_frames}")
        print(f"Average episode duration: {np.mean(lengths):.2f} frame")
        print(f"Shortes episode: {np.min(lengths)} frames | Longest episode: {np.max(lengths)} frames")

        print("\n=== SIMULATION STATISTICS ===")
        print(f"Average object position: {np.mean(np.asarray(obj_positions), axis=0)}")
        print(f"Max object distance from center: {np.max(np.linalg.norm(np.asarray(obj_positions), axis=1))}")
        print("Phase stasts:")
        for i in range(len(phases)):
            num_frames = phase_stats[phases[i]]['num_frames']
            open_grip = phase_stats[phases[i]]['opened_grip_command']
            close_grip = phase_stats[phases[i]]['closed_grip_command']
            print(f"[{phases[i].capitalize()}] Total frames: {num_frames}, {num_frames/total_frames*100: .2f}% | Open grip: {open_grip}, {open_grip/num_frames*100: .2f}% | Close grip: {close_grip}, {close_grip/num_frames*100: .2f}%")
            
        analyzed_frames = sum(stats['num_frames'] for stats in phase_stats.values())
        total_open_grip = sum(stats['opened_grip_command'] for stats in phase_stats.values())
        total_close_grip = sum(stats['closed_grip_command'] for stats in phase_stats.values())
        print(f"Totale frames: {total_frames} | Open grip: {total_open_grip}, {total_open_grip/analyzed_frames*100: .2f}% | Close grip: {total_close_grip}, {total_close_grip/analyzed_frames*100: .2f}%")
        print(f"Analyzed all? {'YES' if total_frames == analyzed_frames else 'NO'}")

def analyze_demo(demo, obs):
    num_frames = len(demo['states'])

    for i in range(num_frames):
        #if i == 0 or i == num_frames - 1:
            #print(f"[Frame-{i}] Action saved: {demo['actions'][i]}")
            #print(f"[Frame-{i}] End-Effector z: {demo['actions'][i][2]}")

        phase = phases[demo['phase_id'][i]]
        grip_command = demo['gripper_command'][i]

        phase_stats[phase]['num_frames'] += 1

        if grip_command == 0:
            phase_stats[phase]['closed_grip_command'] += 1
        else:
            phase_stats[phase]['opened_grip_command'] += 1

def print_hdf5_tree(group, prefix=""):
    items = list(group.items())
    for index, (key, obj) in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "

        if isinstance(obj, h5py.Dataset):
            print(
                f"{prefix}{connector}{key} (Dataset: shape={obj.shape}, dtype={obj.dtype})"
            )
        elif isinstance(obj, h5py.Group):
            print(f"{prefix}{connector}{key}/ (Group)")
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_hdf5_tree(obj, new_prefix)        


if __name__ == "__main__":
    main()