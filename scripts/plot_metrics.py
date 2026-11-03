import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Configuration of possible simulation phases (dynamic).
PHASE_NAMES = [
    "approach", "descend", "grasp", "lift",
    "transport", "align", "lower", "release",
]

def print_report(df: pd.DataFrame):
    """Calculates and prints the best values for each metric saved in the DataFrame."""
    print("=" * 70)
    print(f"{'METRIC':<25} | {'BEST VALUE':<15} | {'STEP':<10}")
    print("-" * 70)
    
    for col in df.columns:
        if col == "step":
            continue
            
        # Action Accuracy should be maximized, Losses should be minimized.
        if col == "action_accuracy":
            best_idx = df[col].idxmax()
            best_val = df.loc[best_idx, col]
            print(f"{col:<25} |  {best_val:<14.5f} | {df.loc[best_idx, 'step']:<10} (Max)")
        else:
            best_idx = df[col].idxmin()
            best_val = df.loc[best_idx, col]
            print(f"{col:<25} |  {best_val:<14.5f} | {df.loc[best_idx, 'step']:<10} (Min)")
    print("=" * 70)

def plot_single_run(jsonl_path: Path, output_file: Path = None):
    """Reads a JSONL file, prints its report, and exports aggregated plots."""
    if not jsonl_path.is_file():
        print(f"[SKIP] File not found: {jsonl_path}")
        return
        
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    if not data:
        print(f"[SKIP] No data present in {jsonl_path}")
        return
        
    # Load via Pandas DataFrame (much more robust).
    df = pd.DataFrame(data)
    
    print(f"\n[RUN ANALYSIS]: {jsonl_path.parent.name}")
    print_report(df)

    # Initialize the figure in Seaborn 2x2 mode.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Training Metrics - {jsonl_path.parent.name}", fontsize=18, fontweight="bold")

    # --- QUADRANT 1: Train Loss & Action Accuracy (Shared axes) ---
    ax1 = axes[0, 0]
    if "train_loss" in df.columns:
        sns.lineplot(data=df, x="step", y="train_loss", ax=ax1, color="blue", label="Train Loss")
        ax1.set_title("Global Loss & Action Accuracy")
        ax1.set_ylabel("Train Loss (Cross-Entropy)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        
        if "action_accuracy" in df.columns:
            ax1_twin = ax1.twinx()
            sns.lineplot(data=df, x="step", y="action_accuracy", ax=ax1_twin, color="magenta", label="Accuracy")
            ax1_twin.set_ylabel("Action Accuracy", color="magenta")
            ax1_twin.tick_params(axis='y', labelcolor="magenta")
            # Merge legends from both axes.
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
            ax1_twin.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
            ax1.get_legend().remove()

    # --- QUADRANT 2: L1 Action Loss (Continuous Action) ---
    ax2 = axes[0, 1]
    if "l1_loss" in df.columns:
        sns.lineplot(data=df, x="step", y="l1_loss", ax=ax2, color="teal")
        ax2.set_title("L1 Action Loss (Continuous Space)")
        ax2.set_ylabel("L1 Loss")

    # --- QUADRANT 3: XYZ, RPY and Gripper Loss ---
    ax3 = axes[1, 0]
    if all(c in df.columns for c in ["loss_xyz", "loss_rpy", "loss_gripper"]):
        sns.lineplot(data=df, x="step", y="loss_xyz", ax=ax3, label="Delta XYZ", color="green")
        sns.lineplot(data=df, x="step", y="loss_rpy", ax=ax3, label="Delta RPY", color="orange")
        sns.lineplot(data=df, x="step", y="loss_gripper", ax=ax3, label="Gripper", color="purple")
        ax3.set_title("Loss by Action Component")
        ax3.set_ylabel("Cross-Entropy Loss")

    # --- QUADRANT 4: Loss by Simulation Phase ---
    ax4 = axes[1, 1]
    phase_cols = [f"loss_phase_{p}" for p in PHASE_NAMES if f"loss_phase_{p}" in df.columns]
    
    if phase_cols:
        for col in phase_cols:
            phase_label = col.replace("loss_phase_", "").capitalize()
            sns.lineplot(data=df, x="step", y=col, ax=ax4, label=phase_label)
        ax4.set_title("Loss by Simulation Phase (Detailed)")
    elif "loss_phase_open" in df.columns and "loss_phase_close" in df.columns:
        sns.lineplot(data=df, x="step", y="loss_phase_open", ax=ax4, label="Open Phase", color="blue")
        sns.lineplot(data=df, x="step", y="loss_phase_close", ax=ax4, label="Close Phase", color="red")
        ax4.set_title("Loss by Gripper Phase (Fallback)")
    
    ax4.set_ylabel("Cross-Entropy Loss")
    if ax4.get_legend() is not None:
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Saving plot.
    fig.tight_layout()
    if output_file is None:
        output_file = jsonl_path.parent / "comprehensive_training_metrics.png"
        
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Plot saved to: {output_file}\n")

def plot_all_runs(runs_dir: Path, name_contains: str = None):
    """Iterates and plots for all run folders within the base directory."""
    if not runs_dir.is_dir():
        print(f"[ERROR] Base directory does not exist: {runs_dir}")
        return
        
    run_folders = [p for p in runs_dir.iterdir() if p.is_dir()]
    for folder in run_folders:
        if name_contains and name_contains not in folder.name:
            continue
            
        jsonl_file = folder / "fine_grained_metrics.jsonl"
        plot_single_run(jsonl_file)

def plot_stacked_success_rate(successes, fails) -> Figure:
    total = successes + fails
    
    if total > 0:
        pct_successes = (successes/total) * 100
        pct_fails = (fails/total) * 100
    else:
        pct_successes = pct_fails = 0

    fig, ax = plt.subplots(figsize=(5,6))
    
    # Stacked bars!
    ax.bar('Model', successes, color='#2ec4b6', label=f'Successes ({successes})', width=0.4)
    ax.bar('Model', fails, color='#e71d36', label=f'Fails ({fails})', width=0.4)

    ax.title('Rapporto Successi / Fallimenti', fontsize=14, fontweight='bold', pad=15)
    ax.ylabel('Numero di Task (Unità)', fontsize=12)
    ax.ylim(0, total * 1.1)  # Spazio in alto per il testo del totale
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    # Insert percentages if they are > 0.
    if successes > 0:
        ax.text(0, successes / 2, f'{pct_successes:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    if fails > 0:
        ax.text(0, successes + (fails / 2), f'{pct_fails:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # Total tag over the column.
    ax.text(0, total + (total * 0.01), f'Totale Task: {total}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    fig.tight_layout()

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots OpenVLA metrics in a 2x2 dashboard and prints the report.")
    
    # Exclusive optional arguments.
    parser.add_argument("--log_file", type=str, default=None, help="Direct path to the .jsonl file (Single run).")
    parser.add_argument("--runs_dir", type=str, default=None, help="Directory containing the various run folders.")
    parser.add_argument("--contains", type=str, default=None, help="Filters folders in runs_dir containing this string.")
    
    args = parser.parse_args()
    
    if args.log_file:
        plot_single_run(Path(args.log_file))
    elif args.runs_dir:
        plot_all_runs(Path(args.runs_dir), args.contains)
    else:
        # If nothing is passed, it automatically searches in runs/openvla/
        default_runs = Path(__file__).resolve().parents[1] / "runs" / "openvla"
        print(f"No argument provided. Starting scan on default directory: {default_runs}")
        plot_all_runs(default_runs, args.contains)