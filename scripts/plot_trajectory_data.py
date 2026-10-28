import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_temporal_series(hdf5_path: Path, output_dir: Path, max_demos: int = 5):
    """
    Legge un dataset HDF5 generato per P-Rob3 e produce grafici dell'andamento 
    temporale (step-by-step) di azioni (comandi) e posizioni (giunti).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"Gruppo 'data' mancante nel file {hdf5_path}")
        
        # Prende solo le prime N dimostrazioni per evitare di generare centinaia di grafici
        demo_names = sorted(f["data"].keys())[:max_demos]
        print(f"Generazione grafici temporali per {len(demo_names)} episodi in: {output_dir.resolve()}")

        for demo_name in demo_names:
            demo = f["data"][demo_name]
            
            # Estrazione dei dati
            actions = demo["actions_openvla"][:]
            joints = demo["obs"]["joint_states"][:]
            
            steps = np.arange(actions.shape[0])
            
            # Creazione di una figura con 4 subplot impilati verticalmente
            fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
            fig.suptitle(f"Andamento Temporale - Traiettoria: {demo_name}", fontsize=16, fontweight="bold")
            
            # 1. Comandi di Traslazione (dx, dy, dz)
            axs[0].plot(steps, actions[:, 0], label="dx", alpha=0.85, linewidth=1.5)
            axs[0].plot(steps, actions[:, 1], label="dy", alpha=0.85, linewidth=1.5)
            axs[0].plot(steps, actions[:, 2], label="dz", alpha=0.85, linewidth=1.5)
            axs[0].set_ylabel("Traslazione (m)")
            axs[0].set_title("Comandi di Traslazione (OSC Delta Pos)")
            axs[0].legend(loc="upper right")
            axs[0].grid(True, linestyle="--", alpha=0.6)
            
            # 2. Comandi di Rotazione (dRx, dRy, dRz)
            axs[1].plot(steps, actions[:, 3], label="dRx", alpha=0.85, linewidth=1.5)
            axs[1].plot(steps, actions[:, 4], label="dRy", alpha=0.85, linewidth=1.5)
            axs[1].plot(steps, actions[:, 5], label="dRz", alpha=0.85, linewidth=1.5)
            axs[1].set_ylabel("Rotazione (rad)")
            axs[1].set_title("Comandi di Rotazione (OSC Delta Quat/Euler)")
            axs[1].legend(loc="upper right")
            axs[1].grid(True, linestyle="--", alpha=0.6)
            
            # 3. Comando Gripper
            axs[2].plot(steps, actions[:, 6], label="Gripper (0=Chiuso, 1=Aperto)", color="purple", drawstyle="steps-post", linewidth=2)
            axs[2].set_ylabel("Stato")
            axs[2].set_title("Stato del Gripper")
            axs[2].legend(loc="upper right")
            axs[2].grid(True, linestyle="--", alpha=0.6)
            
            # 4. Posizioni dei Giunti (q1-q6)
            for i in range(6):
                axs[3].plot(steps, joints[:, i], label=f"q{i+1}", alpha=0.85, linewidth=1.5)
            axs[3].set_ylabel("Angolo (rad)")
            axs[3].set_title("Stato Effettivo: Posizione dei 6 Giunti (qpos)")
            axs[3].set_xlabel("Sim Step")
            axs[3].legend(loc="center right", bbox_to_anchor=(1.12, 0.5))
            axs[3].grid(True, linestyle="--", alpha=0.6)
            
            plt.tight_layout()
            
            # Salvataggio
            out_file = output_dir / f"{demo_name}_temporal.png"
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f" -> Salvato: {out_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi Temporale Traiettorie HDF5")
    parser.add_argument("--hdf5", type=str, required=True, help="Path al file dataset.hdf5")
    parser.add_argument("--output", type=str, default="temporal_analysis", help="Cartella di output per i grafici")
    parser.add_argument("--max_demos", type=int, default=5, help="Numero massimo di episodi da plottare")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    output_path = Path(args.output)
    
    if not hdf5_path.is_file():
        print(f"Errore: Il file {hdf5_path} non esiste.")
    else:
        plot_temporal_series(hdf5_path, output_path, args.max_demos)