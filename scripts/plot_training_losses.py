import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(jsonl_path: Path):
    steps = []
    train_loss = []
    action_accuracy = []
    l1_loss = []
    loss_xyz = []
    loss_rpy = []
    loss_gripper = []
    loss_phase_open = []
    loss_phase_close = []
    
    # Legge il file JSONL riga per riga
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            steps.append(data["step"])
            train_loss.append(data["train_loss"])
            action_accuracy.append(data['action_accuracy'])
            l1_loss.append(data['l1_loss'])
            loss_xyz.append(data["loss_xyz"])
            loss_rpy.append(data["loss_rpy"])
            loss_gripper.append(data['loss_gripper'])
            loss_phase_open.append(data['loss_phase_open'])
            loss_phase_close.append(data['loss_phase_close'])

    best_values = {
        "train_loss": [-1, +np.inf],
        "action_accuracy": [-1, +np.inf],
        "l1_loss": [-1, +np.inf],
        "loss_xyz": [-1, +np.inf],
        "loss_rpy": [-1, +np.inf],
        "loss_gripper": [-1, +np.inf],
        "loss_phase_open": [-1, +np.inf],
        "loss_phase_close": [-1, +np.inf]
    }

    for step in range(len(steps)):
        best_values['train_loss'] = [step, min(best_values['train_loss'][1], train_loss[step])]
        best_values['action_accuracy'] = [step, min(best_values['action_accuracy'][1], action_accuracy[step])]
        best_values['l1_loss'] = [step, min(best_values['l1_loss'][1], l1_loss[step])]
        best_values['loss_xyz'] = [step, min(best_values['loss_xyz'][1], loss_xyz[step])]
        best_values['loss_rpy'] = [step, min(best_values['loss_rpy'][1], loss_rpy[step])]
        best_values['loss_gripper'] = [step, min(best_values['loss_gripper'][1], loss_gripper[step])]
        best_values['loss_phase_open'] = [step, min(best_values['loss_phase_open'][1], loss_phase_open[step])]
        best_values['loss_phase_close'] = [step, min(best_values['loss_phase_close'][1], loss_phase_close[step])]

    print_report(best_values)

    # Creazione dei grafici
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Andamento Loss di Addestramento (OpenVLA)", fontsize=16, fontweight="bold")

    axs[0].plot(steps, train_loss, label="Train Loss Globale", color="blue", linewidth=2)
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)

    axs[1].plot(steps, loss_xyz, label="Loss XYZ (Traslazione)", color="green", linewidth=2)
    axs[1].set_ylabel("Loss XYZ")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)

    axs[2].plot(steps, loss_rpy, label="Loss RPY (Rotazione)", color="orange", linewidth=2)
    axs[2].set_ylabel("Loss RPY")
    axs[2].set_xlabel("Gradient Steps")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    
    out_file = jsonl_path.parent / "training_loss_plot.png"
    plt.savefig(out_file, dpi=200)
    print(f"Grafico salvato in: {out_file}")

def print_report(best_val: dict):
    print("="*60)
    for key, val in best_val.items():
        print(f"Best [{key}] -> {val[1]: .30f}, Step: {val[0]}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotta le metriche salvate in fine_grained_metrics.jsonl")
    parser.add_argument("--log_file", type=str, required=True, help="Path al file fine_grained_metrics.jsonl")
    args = parser.parse_args()
    
    plot_metrics(Path(args.log_file))