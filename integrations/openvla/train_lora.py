#!/usr/bin/env python3
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# --- 1. CONFIGURAZIONE AMBIENTE ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 2. PERCORSI ---
THESIS_ROOT = Path.home() / "Scrivania" / "thesis"
VENV = THESIS_ROOT / "venv_openvla"

# Garantisce che `integrations.openvla.*` sia importabile da qualsiasi directory di lancio
pwd = os.getcwd()
existing_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{existing_pp}:{THESIS_ROOT}:{pwd}".strip(":")

RLDS_BASE = str(THESIS_ROOT / "rlds_data" / "rlds_base")

RUN_ROOT = str(THESIS_ROOT / "runs" / "openvla")
ADAPTER_ROOT = str(THESIS_ROOT / "adapters" / "openvla")
os.environ["HF_HOME"] = str(THESIS_ROOT / "cache" / "huggingface")

# Flag di attivazione validation in finetune.py (il path non viene letto, ma deve esistere)
VAL_FILE = str(
    THESIS_ROOT
    / "hdf5_data"
    / "v5.0_no_randomization_better_camera_perturbed_20261113_133035_010799"
    / "splits"
    / "val.hdf5"
)
TRAIN_WRAPPER = str(THESIS_ROOT / "integrations" / "openvla" / "finetune.py")

# --- 3. IPERPARAMETRI FISSI ---
MODEL = "openvla/openvla-7b"
DATASET = "prob3_pick_place"
NUM_GPUS = 2

USE_LORA = "True"
LORA_DROPOUT = "0.05"
USE_QUANTIZATION = "False"
IMAGE_AUG = "True"
SHUFFLE_BUFFER_SIZE = "100000"
SAVE_LATEST_CHECKPOINT_ONLY = "False"
TARGET_MODULES = "all-linear"
VAL_MAX_BATCHES = "1000"

# Se True, in caso di errore su una run prosegue comunque con le successive nel weekend
CONTINUE_ON_ERROR = True
MIN_FREE_DISK_GB = 135.0


@dataclass(frozen=True)
class TrainExperiment:
    note: str
    data_dirs: List[str]
    data_weights: List[float]
    max_steps: int
    warmup_steps: int
    save_steps: int
    lr: str
    batch_size: int
    grad_accum: int
    lora_rank: int
    lora_alpha: int


# --- 4. GRIGLIA DI CONFIGURAZIONI WEEKEND SU v5.0 (~84.4 ore -> fine ~martedì 04:20) ---
CONFIGURATIONS: List[TrainExperiment] = [
    # 1) Estensione diretta della best run v5.0 (LR=1e-4, BES=32, R=64, A=16, D=0.05) -> 45k step (~18.9h)
    TrainExperiment(
        note="v5.0_LR1e_4_ACC2_BES32_R64_A16_D005_45k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=45000,
        warmup_steps=400,
        save_steps=1500,
        lr="1e-4",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 2) BES=32 con LR moderato (-30%): 7e-5 su 45k step (~18.9h)
    TrainExperiment(
        note="v5.0_LR7e_5_ACC2_BES32_R64_A16_D005_45k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=45000,
        warmup_steps=500,
        save_steps=1500,
        lr="7e-5",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 3) Ricetta forum su BES=32: LR=2e-5, R=32, A=64 (scaling 2.0x), warmup=100 su 48k step (~19.8h)
    TrainExperiment(
        note="v5.0_LR2e_5_ACC2_BES32_R32_A64_D005_48k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=48000,
        warmup_steps=100,
        save_steps=2000,
        lr="2e-5",
        batch_size=8,
        grad_accum=2,
        lora_rank=32,
        lora_alpha=64,
    ),
    # 4) ACC=1 (BES=16) con LR=7e-5, R=64, A=16 su 40k step (~8.7h)
    TrainExperiment(
        note="v5.0_LR7e_5_ACC1_BES16_R64_A16_D005_40k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=40000,
        warmup_steps=500,
        save_steps=2000,
        lr="7e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 5) ACC=1 (BES=16) con LR basso 5e-5, R=64, A=16 su 44k step (~9.5h)
    TrainExperiment(
        note="v5.0_LR5e_5_ACC1_BES16_R64_A16_D005_44k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=44000,
        warmup_steps=600,
        save_steps=2200,
        lr="5e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 6) Ricetta forum su BES=16: LR=2e-5, R=32, A=64 (scaling 2.0x), warmup=100 su 40k step (~8.7h)
    TrainExperiment(
        note="v5.0_LR2e_5_ACC1_BES16_R32_A64_D005_40k",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        max_steps=40000,
        warmup_steps=100,
        save_steps=2000,
        lr="2e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=32,
        lora_alpha=64,
    ),
]


def validate_configurations(torchrun_bin: Path, exps: List[TrainExperiment]) -> None:
    """Controllo preventivo su file, spazio disco, dataset RLDS e parametri prima di avviare le GPU."""
    if not torchrun_bin.is_file():
        raise FileNotFoundError(f"torchrun non trovato: {torchrun_bin}")
    if not Path(TRAIN_WRAPPER).is_file():
        raise FileNotFoundError(f"finetune.py non trovato: {TRAIN_WRAPPER}")
    if not Path(VAL_FILE).is_file():
        raise FileNotFoundError(f"VAL_FILE non trovato: {VAL_FILE}")

    Path(RUN_ROOT).mkdir(parents=True, exist_ok=True)
    Path(ADAPTER_ROOT).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ADAPTER_ROOT).free / (1024**3)
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Spazio disco insufficiente in {ADAPTER_ROOT}: {free_gb:.1f} GB liberi "
            f"(richiesti almeno {MIN_FREE_DISK_GB:.0f} GB)."
        )

    visible_gpus = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if len(visible_gpus) != NUM_GPUS:
        raise ValueError(f"NUM_GPUS={NUM_GPUS}, ma CUDA_VISIBLE_DEVICES={visible_gpus}")

    seen_notes = set()
    for exp in exps:
        if exp.note in seen_notes:
            raise ValueError(f"run_id_note duplicato: {exp.note}")
        seen_notes.add(exp.note)

        if exp.max_steps % exp.save_steps != 0:
            raise ValueError(
                f"[{exp.note}] max_steps ({exp.max_steps}) non divisibile per save_steps ({exp.save_steps})!"
            )
        if exp.warmup_steps >= exp.max_steps:
            raise ValueError(f"[{exp.note}] warmup_steps ({exp.warmup_steps}) >= max_steps ({exp.max_steps})")
        if len(exp.data_dirs) != len(exp.data_weights) or not exp.data_dirs:
            raise ValueError(f"[{exp.note}] Mismatch tra data_dirs e data_weights")
        if any(w <= 0 for w in exp.data_weights):
            raise ValueError(f"[{exp.note}] Tutti i data_weights devono essere > 0: {exp.data_weights}")

        for d in exp.data_dirs:
            ds_path = Path(d) / DATASET
            if not ds_path.is_dir():
                raise FileNotFoundError(
                    f"[{exp.note}] Dataset RLDS mancante: {ds_path}. "
                    "Esegui prima prepare_dataset.py e build_dataset.py --publish!"
                )
            versions = sorted(p for p in ds_path.iterdir() if p.is_dir())
            if not versions:
                raise FileNotFoundError(f"[{exp.note}] Nessuna cartella versione dentro {ds_path}")
            latest_ver = versions[-1]
            train_shards = list(latest_ver.glob("*train.tfrecord*"))
            val_shards = list(latest_ver.glob("*val.tfrecord*"))
            if not train_shards or not val_shards:
                raise FileNotFoundError(
                    f"[{exp.note}] Shard train/val mancanti in {latest_ver} "
                    f"(train={len(train_shards)}, val={len(val_shards)})"
                )

    print(
        f"[Pre-check OK] Verificate {len(exps)} configurazioni | "
        f"Spazio libero: {free_gb:.1f} GB | Shard RLDS train/val presenti."
    )


def main() -> None:
    torchrun_path = VENV / "bin" / "torchrun"
    validate_configurations(torchrun_path, CONFIGURATIONS)

    results = []

    for idx, exp in enumerate(CONFIGURATIONS, start=1):
        eff_batch = exp.batch_size * exp.grad_accum * NUM_GPUS
        num_ckpts = exp.max_steps // exp.save_steps

        print("\n" + "=" * 72)
        print(
            f"[{idx}/{len(CONFIGURATIONS)}] INIZIO TRAINING: {exp.note}\n"
            f"  Datasets : {[Path(d).name for d in exp.data_dirs]} | Weights: {exp.data_weights}\n"
            f"  Params   : LR={exp.lr} | Batch/GPU={exp.batch_size} | Accum={exp.grad_accum} (BES={eff_batch})\n"
            f"  LoRA     : Rank={exp.lora_rank} | Alpha={exp.lora_alpha} (scale={exp.lora_alpha / exp.lora_rank:.2f}) | Dropout={LORA_DROPOUT}\n"
            f"  Schedule : Steps={exp.max_steps} | Warmup={exp.warmup_steps} | Save every {exp.save_steps} ({num_ckpts} ckpts)"
        )
        print("=" * 72)

        cmd = [
            str(torchrun_path),
            "--standalone",
            "--nnodes", "1",
            "--nproc-per-node", str(NUM_GPUS),
            TRAIN_WRAPPER,
            "--vla_path", MODEL,
            "--dataset_name", DATASET,
            "--data_root_dirs", str(exp.data_dirs),
            "--dataset_weights", str(exp.data_weights),
            "--val_file", VAL_FILE,
            "--run_root_dir", RUN_ROOT,
            "--adapter_tmp_dir", ADAPTER_ROOT,
            "--use_lora", USE_LORA,
            "--lora_rank", str(exp.lora_rank),
            "--lora_alpha", str(exp.lora_alpha),
            "--lora_dropout", LORA_DROPOUT,
            "--use_quantization", USE_QUANTIZATION,
            "--batch_size", str(exp.batch_size),
            "--grad_accumulation_steps", str(exp.grad_accum),
            "--learning_rate", exp.lr,
            "--max_steps", str(exp.max_steps),
            "--warmup_steps", str(exp.warmup_steps),
            "--save_steps", str(exp.save_steps),
            "--image_aug", IMAGE_AUG,
            "--shuffle_buffer_size", SHUFFLE_BUFFER_SIZE,
            "--val_max_batches", VAL_MAX_BATCHES,
            "--run_id_note", exp.note,
            "--save_latest_checkpoint_only", SAVE_LATEST_CHECKPOINT_ONLY,
            "--target_modules", TARGET_MODULES,
            "--early_stopping_enabled", "True",
            "--early_stopping_target_acc", "0.95",
            "--early_stopping_patience", "5",
            "--early_stopping_min_rel_delta", "0.005",
            "--early_stopping_ema_alpha", "0.5",
            "--early_stopping_l1_source", "val",
        ]

        try:
            subprocess.run(cmd, check=True)
            print("\n" + "=" * 72)
            print(f"TRAINING {exp.note} COMPLETATO CON SUCCESSO!")
            print("=" * 72 + "\n")
            results.append((exp.note, "OK"))
        except subprocess.CalledProcessError as e:
            print(f"\nERRORE durante il training della configurazione {exp.note} (exit code {e.returncode})!")
            results.append((exp.note, f"FAILED (code {e.returncode})"))
            if not CONTINUE_ON_ERROR:
                sys.exit(e.returncode)

    print("\n" + "=" * 72)
    print("RIEPILOGO FINALE CODA WEEKEND:")
    for note, status in results:
        print(f"  - {note}: {status}")
    print("=" * 72)


if __name__ == "__main__":
    main()