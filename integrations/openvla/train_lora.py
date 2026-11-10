#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

# --- 1. CONFIGURAZIONE AMBIENTE (Variabili export) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "online"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Gestione PYTHONPATH (aggiunge la cartella corrente)
pwd = os.getcwd()
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{pwd}"

# --- 2. PERCORSI (Root e Directory) ---
THESIS_ROOT = Path.home() / "Scrivania" / "thesis"
VENV = THESIS_ROOT / "venv_openvla"

# Definizione dei Dataset (Predisposta per i futuri mix)
DATA_DIRS = [
    str(THESIS_ROOT / "rlds_data" / "rlds_base"),
    # str(THESIS_ROOT / "rlds_data" / "rlds_lights"), # Decommenta in futuro
]
DATA_WEIGHTS = [1.0]  # Aggiungi pesi futuri (es. , 0.3) qui

RUN_ROOT = str(THESIS_ROOT / "runs" / "openvla")
ADAPTER_ROOT = str(THESIS_ROOT / "adapters" / "openvla")
os.environ["HF_HOME"] = str(THESIS_ROOT / "cache" / "huggingface")

VAL_FILE = str(THESIS_ROOT / "hdf5_data" / "v3.0_no_randomization_20261109_142456_559122" / "splits" / "validation.hdf5")
TRAIN_WRAPPER = str(THESIS_ROOT / "integrations" / "openvla" / "finetune.py")

# --- 3. IPERPARAMETRI FISSI ---
MODEL = "openvla/openvla-7b"
DATASET = "prob3_pick_place"
NUM_GPUS = 2

USE_LORA = "True"
LORA_DROPOUT = "0.05" 
USE_QUANTIZATION = "False"
BATCH_SIZE = "8"
IMAGE_AUG = "True"
SHUFFLE_BUFFER_SIZE = "100000"
SAVE_LATEST_CHECKPOINT_ONLY = "False"
TARGET_MODULES = "all-linear"

MAX_STEPS = "30000"
WARMUP_STEPS = "500"
SAVE_STEPS = "1000"

# --- 4. GRIGLIA DI CONFIGURAZIONI (Variabili nel loop) ---
# Ogni tupla contiene: (learning_rate, grad_accumulation, lora_rank, note)
CONFIGURATIONS = [
    ("1e-4", "2", 32, "v3.0.1_LR1e-4_Acc2_R32_A16_alllinear"),
    ("4e-4", "2", 32, "v3.0.1_LR4e-4_Acc4_R32_A16-alllinear")
]

# --- 5. LOOP DI TRAINING ---
torchrun_path = str(VENV / "bin" / "torchrun")

for lr, accum, rank, note in CONFIGURATIONS:
    str_rank = str(rank)
    #alpha = str_rank

    print("=" * 60)
    print(f"INIZIO TRAINING: {note} (LR={lr} | Accum={accum} | Rank={str_rank})")
    print("=" * 60)

    # Costruzione del comando lineare pulito
    cmd = [
        torchrun_path,
        "--standalone",
        "--nnodes", "1",
        "--nproc-per-node", str(NUM_GPUS),
        
        TRAIN_WRAPPER,
        
        "--vla_path", MODEL,
        "--dataset_name", DATASET,
        "--data_root_dirs", str(DATA_DIRS),
        "--dataset_weights", str(DATA_WEIGHTS),
        "--val_file", VAL_FILE,
        "--run_root_dir", RUN_ROOT,
        "--adapter_tmp_dir", ADAPTER_ROOT,
        "--use_lora", USE_LORA,
        "--lora_rank", str_rank,
        #"--lora_alpha", alpha,
        "--lora_dropout", LORA_DROPOUT,
        "--use_quantization", USE_QUANTIZATION,
        "--batch_size", BATCH_SIZE,
        "--grad_accumulation_steps", accum,
        "--learning_rate", lr,
        "--max_steps", MAX_STEPS,
        "--warmup_steps", WARMUP_STEPS,
        "--save_steps", SAVE_STEPS,
        "--image_aug", IMAGE_AUG,
        "--shuffle_buffer_size", SHUFFLE_BUFFER_SIZE,
        "--run_id_note", note,
        "--save_latest_checkpoint_only", SAVE_LATEST_CHECKPOINT_ONLY,
        "--target_modules", TARGET_MODULES,
    ]

    try:
        # Esegue il processo inoltrando l'output in tempo reale sul terminale
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print(f"TRAINING {note} COMPLETATO CON SUCCESSO!")
        print("=" * 60 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"\nERRORE durante il training della configurazione {note}!")
        print(f"Il processo è terminato con codice di errore: {e.returncode}")
        # Decidi se interrompere lo script totale o continuare con la configurazione successiva
        sys.exit(e.returncode)
