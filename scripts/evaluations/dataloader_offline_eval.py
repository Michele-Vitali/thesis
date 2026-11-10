import os
os.environ["USE_TF"] = "0"  # Disabilita i warning di TensorFlow

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# Importiamo OpenVLA AutoClasses
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from scripts.metrics.offline_metrics import compute_metrics, save_metrics

# FONDAMENTALE: Registriamo il dataset locale per le statistiche di un-normalization
try:
    from integrations.openvla.register import register_prob3
    register_prob3()
except ImportError:
    print("Attenzione: Impossibile importare register_prob3(). Le azioni non verranno denormalizzate correttamente.")


class HDF5OfflineDataset(Dataset):
    """
    Dataset PyTorch per caricare efficientemente l'HDF5 di validazione.
    Invece di aprire il file e bloccarlo, mappiamo in __init__ i puntatori (demo, timestep, text)
    per rendere la complessità di __getitem__ O(1).
    """
    def __init__(self, hdf5_path, max_demos=None):
        self.hdf5_path = hdf5_path
        self.samples = []

        with h5py.File(self.hdf5_path, "r") as f:
            demos = sorted(f["data"].keys())
            if max_demos is not None:
                demos = demos[:max_demos]

            for demo_name in demos:
                num_frames = len(f["data"][demo_name]["actions_openvla"])
                instruction = f["data"][demo_name].attrs["language_instruction"]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

                # Salviamo un riferimento "piatto" per ogni singolo frame
                for t in range(num_frames):
                    self.samples.append((demo_name, t, instruction))
                    
        self.file = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Il file HDF5 viene aperto al volo dal worker del dataloader per evitare conflitti di I/O
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

        demo_name, t, instruction = self.samples[idx]

        # Estraiamo immagine e convertiamola subito in PIL (formato amato dal processor di OpenVLA)
        img_array = self.file["data"][demo_name]["obs"]["agentview_rgb"][t]
        image = Image.fromarray(np.asarray(img_array, dtype=np.uint8))
        
        # Ground Truth Action
        gt_action = np.asarray(self.file["data"][demo_name]["actions_openvla"][t], dtype=np.float32)

        return image, instruction, gt_action, demo_name, t


def custom_collate_fn(batch):
    """
    Impedisce al DataLoader di tentare di fondere in tensori oggetti non supportati (come stringhe o immagini PIL).
    Converte il batch di tuple in liste indipendenti.
    """
    images = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    gt_actions = np.stack([item[2] for item in batch])
    demo_names = [item[3] for item in batch]
    timesteps = [item[4] for item in batch]
    
    return images, instructions, gt_actions, demo_names, timesteps


def main():
    parser = argparse.ArgumentParser(description="Valutazione offline modelli OpenVLA in Batch Mode.")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train", "test"], help="Split da valutare")
    parser.add_argument("--max_demos", type=int, default=None, help="Numero massimo di episodi")
    parser.add_argument("--n_models", type=int, default=None, help="Numero di ultimi modelli da valutare")
    parser.add_argument("--batch_size", type=int, default=16, help="Dimensione del batch su GPU (es. 16 o 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="Numero di thread per caricare HDF5")
    args = parser.parse_args()

    # Setup dei percorsi come nel tuo script
    script_dir = Path(__file__).parent.resolve()
    thesis_root = script_dir.parent
    hdf5_root = thesis_root / "hdf5_data"

    split_filename = f"{args.split}.hdf5" if args.split != "dataset" else "dataset.hdf5"

    available_datasets = sorted(
        [path for path in hdf5_root.iterdir() if path.is_dir() and ((path / "splits" / split_filename).is_file() or (path / split_filename).is_file())],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not available_datasets:
        raise RuntimeError(f"No dataset with {split_filename} was found!")

    print("=" * 60)
    print("Available datasets:")
    for i, path in enumerate(available_datasets):
        print(f"[{i}] {path.name}")

    choice = input("\nChosen dataset index (Enter => Choose the latest): ").strip()
    index = int(choice) if choice else 0

    dataset_dir = available_datasets[index]
    eval_file = dataset_dir / "splits" / split_filename if (dataset_dir / "splits" / split_filename).is_file() else dataset_dir / split_filename

    checkpoints_base_dir = script_dir / Path("../runs/openvla")
    n_models_to_test = args.n_models if args.n_models is not None else int(input("How many models you want to test? "))

    models = sorted(
        [p for p in checkpoints_base_dir.iterdir() if p.is_dir() and "chkpt" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n_models_to_test]
    
    models_name = [p.name for p in models]
    models = [str(p) for p in models]

    # --- Registrazione classi OpenVLA su HuggingFace ---
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Inizializziamo Datset e Dataloader una sola volta per tutti i modelli
    print("\nInizializzazione Dataset HDF5...")
    dataset = HDF5OfflineDataset(eval_file, args.max_demos)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn, 
        pin_memory=True
    )
    print(f"Totale frame da valutare: {len(dataset)} divisi in {len(dataloader)} batch.\n")

    all_models_results = {}

    for i, model_path in enumerate(models):
        print("=" * 60)
        print(f"[Model-{i}] {models_name[i]}")
        print("=" * 60)

        # 1. Caricamento del modello locale diretto sulla GPU (salta client e server)
        print("Caricamento in memoria della GPU in corso...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cuda").eval()

        all_gt_actions = []
        all_pred_actions = []

        start_time = time.time()
        
        # 2. Loop inferenza super veloce a batch
        with torch.no_grad():
            for images, instructions, gt_actions, demo_names, timesteps in tqdm(dataloader, desc="Inferenza Batch"):
                
                # predict_action fa tutto: tokenizzazione, generazione e UN-NORMALIZATION usando la tua chiave
                pred_actions = vla.predict_action(
                    images,                  # Accetta direttamente la lista
                    instructions,            # Accetta direttamente la lista
                    unnorm_key="prob3_pick_place", # FONDAMENTALE: Richiama le metriche fisiche
                    do_sample=False
                )
                
                # Salviamo nel buffer globale
                all_pred_actions.extend(pred_actions)
                all_gt_actions.extend(gt_actions)
                
        eval_time = time.time() - start_time
        fps = len(dataset) / eval_time
        print(f"[Inferenza Completata] Tasso di analisi: {fps:.2f} FPS")

        # 3. Metriche come da tuo codice originale
        all_gt_actions = np.asarray(all_gt_actions, dtype=np.float32)
        all_pred_actions = np.asarray(all_pred_actions, dtype=np.float32)
        
        metrics = compute_metrics(all_gt_actions, all_pred_actions)
        results_dir = thesis_root / "evaluations" / models_name[i]
        save_metrics(models_name[i], dataset_dir.name, args.split, metrics, results_dir)
        all_models_results[models_name[i]] = metrics

        print("\n" + "=" * 60)
        print(f"Evaluation results for model: {models_name[i]}")
        print(f"Evaluated frames: {metrics['num_frames']}")
        print(f"Arm MAE: {metrics['arm']['mae']}")
        print(f"Translation MAE: {metrics['arm']['translation_mae']}")
        print(f"Rotation MAE: {metrics['arm']['rotation_mae']}")
        print(f"Gripper accuracy: {metrics['gripper']['accuracy']}")
        print(f"Gripper Macro F1: {metrics['gripper']['macro_f1']}")
        
        # Flush VRAM
        del vla
        del processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()