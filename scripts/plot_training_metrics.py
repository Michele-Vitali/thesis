import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

THESIS_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = THESIS_ROOT / "runs" / "openvla"

PHASE_NAMES = [
    "approach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "align",
    "lower",
    "release",
]


def plot_training_losses(runs_dir=None, name_contains=None):
    """Plots fine-grained training losses (by simulation phase and action component)."""
    base_dir = Path(runs_dir) if runs_dir else RUNS_ROOT

    if not base_dir.is_dir():
        print(f"[SKIP] Runs directory does not exist: {base_dir}")
        return

    run_folders = [p for p in base_dir.iterdir() if p.is_dir()]

    for folder in run_folders:
        if name_contains and name_contains not in folder.name:
            continue

        jsonl_file = folder / "fine_grained_metrics.jsonl"
        if not jsonl_file.is_file():
            continue

        data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        if not data:
            continue

        df = pd.DataFrame(data)
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Check if individual 8-phase metrics exist in dataframe.
        phase_cols = [f"loss_phase_{p}" for p in PHASE_NAMES if f"loss_phase_{p}" in df.columns]

        if phase_cols:
            for col in phase_cols:
                phase_label = col.replace("loss_phase_", "").capitalize()
                sns.lineplot(data=df, x="step", y=col, ax=axes[0], label=phase_label)
            axes[0].set_title("Training Loss by Simulation Phase (8 Phases)")
        elif "loss_phase_open" in df.columns and "loss_phase_close" in df.columns:
            sns.lineplot(data=df, x="step", y="loss_phase_open", ax=axes[0], label="Open Phase", color="blue")
            sns.lineplot(data=df, x="step", y="loss_phase_close", ax=axes[0], label="Close Phase", color="red")
            axes[0].set_title("Training Loss by Gripper Phase (Fallback)")

        axes[0].set_xlabel("Gradient Steps")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot action kinematic components.
        if "loss_xyz" in df.columns and "loss_rpy" in df.columns and "loss_gripper" in df.columns:
            sns.lineplot(data=df, x="step", y="loss_xyz", ax=axes[1], label="Delta XYZ (Translation)", color="green")
            sns.lineplot(data=df, x="step", y="loss_rpy", ax=axes[1], label="Delta RPY (Rotation)", color="orange")
            sns.lineplot(data=df, x="step", y="loss_gripper", ax=axes[1], label="Gripper State", color="purple")
            axes[1].set_title("Training Loss by Action Component")
            axes[1].set_xlabel("Gradient Steps")
            axes[1].set_ylabel("Cross-Entropy Loss")
            axes[1].legend()

        fig.tight_layout()
        output_plot = folder / "fine_grained_training_loss.png"
        fig.savefig(output_plot, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[PLOTS] Saved training loss plot to: {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contains", default=None, help="Substring to match run directory name.")
    args = parser.parse_args()

    plot_training_losses(name_contains=args.contains)