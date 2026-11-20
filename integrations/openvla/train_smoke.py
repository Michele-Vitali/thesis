#!/usr/bin/env python3
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

THESIS_ROOT = Path.home() / "Scrivania" / "thesis"
VENV = THESIS_ROOT / "venv_openvla"
TORCHRUN = VENV / "bin" / "torchrun"
TRAIN_WRAPPER = THESIS_ROOT / "integrations" / "openvla" / "finetune.py"

DATASET_NAME = "prob3_pick_place"
EXPECTED_V5_TAG = "v5.0_no_randomization_better_camera_perturbed_20261113_133035_010799"

RLDS_BASE = str(THESIS_ROOT / "rlds_data" / "rlds_base")
VAL_FILE = str(THESIS_ROOT / "hdf5_data" / EXPECTED_V5_TAG / "splits" / "val.hdf5")
ADAPTER_ROOT = THESIS_ROOT / "adapters" / "openvla"

SMOKE_RUN_ROOT = Path("/tmp/openvla_smoke_runs")
SMOKE_ADAPTER_ROOT = Path("/tmp/openvla_smoke_adapters")
MIN_FREE_DISK_GB = 135.0


@dataclass(frozen=True)
class SmokeExperiment:
    note: str
    data_dirs: List[str]
    data_weights: List[float]
    lr: str
    batch_size: int
    grad_accum: int
    lora_rank: int
    lora_alpha: int


# Rispecchia 1:1 le 6 configurazioni della coda weekend su v5.0
SMOKE_CONFIGURATIONS: List[SmokeExperiment] = [
    # 1) Best run v5.0 estesa (BES=32, LR=1e-4, R=64, A=16)
    SmokeExperiment(
        note="SMOKE_1_v5_LR1e_4_ACC2_BES32_R64_A16",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="1e-4",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 2) BES=32 con LR moderato (7e-5, R=64, A=16)
    SmokeExperiment(
        note="SMOKE_2_v5_LR7e_5_ACC2_BES32_R64_A16",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="7e-5",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 3) Ricetta forum su BES=32 (LR=2e-5, R=32, A=64)
    SmokeExperiment(
        note="SMOKE_3_v5_LR2e_5_ACC2_BES32_R32_A64",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="2e-5",
        batch_size=8,
        grad_accum=2,
        lora_rank=32,
        lora_alpha=64,
    ),
    # 4) ACC=1 / BES=16 con LR=7e-5 (R=64, A=16)
    SmokeExperiment(
        note="SMOKE_4_v5_LR7e_5_ACC1_BES16_R64_A16",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="7e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 5) ACC=1 / BES=16 con LR=5e-5 (R=64, A=16)
    SmokeExperiment(
        note="SMOKE_5_v5_LR5e_5_ACC1_BES16_R64_A16",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="5e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=64,
        lora_alpha=16,
    ),
    # 6) Ricetta forum su BES=16 (LR=2e-5, R=32, A=64)
    SmokeExperiment(
        note="SMOKE_6_v5_LR2e_5_ACC1_BES16_R32_A64",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="2e-5",
        batch_size=8,
        grad_accum=1,
        lora_rank=32,
        lora_alpha=64,
    ),
]


def cleanup() -> None:
    for p in (SMOKE_RUN_ROOT, SMOKE_ADAPTER_ROOT):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def preflight_checks() -> None:
    if not TORCHRUN.is_file():
        raise FileNotFoundError(f"torchrun non trovato: {TORCHRUN}")
    if not TRAIN_WRAPPER.is_file():
        raise FileNotFoundError(f"finetune.py non trovato: {TRAIN_WRAPPER}")
    if not Path(VAL_FILE).is_file():
        raise FileNotFoundError(f"VAL_FILE non trovato: {VAL_FILE}")

    ds_path = Path(RLDS_BASE) / DATASET_NAME
    if not ds_path.is_dir():
        raise FileNotFoundError(f"Dataset RLDS mancante: {ds_path}")
    versions = sorted(p for p in ds_path.iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"Nessuna cartella versione dentro {ds_path}")
    latest_ver = versions[-1]
    train_shards = list(latest_ver.glob("*train.tfrecord*"))
    val_shards = list(latest_ver.glob("*val.tfrecord*"))
    if not train_shards or not val_shards:
        raise FileNotFoundError(
            f"Shard train/val mancanti in {latest_ver} (train={len(train_shards)}, val={len(val_shards)})"
        )

    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ADAPTER_ROOT).free / (1024**3)
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Spazio disco insufficiente in {ADAPTER_ROOT}: {free_gb:.1f} GB liberi "
            f"(servono almeno {MIN_FREE_DISK_GB:.0f} GB per il weekend)."
        )

    netrc = Path.home() / ".netrc"
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY")) or (
        netrc.is_file() and "api.wandb.ai" in netrc.read_text(errors="ignore")
    )
    if not has_wandb_key:
        raise RuntimeError("Credenziali W&B non trovate! Esegui `wandb login` prima di avviare il weekend.")

    print(
        f"[Pre-check OK] Spazio libero: {free_gb:.1f} GB | W&B login OK | "
        f"Shard RLDS OK in {latest_ver}"
    )


def main() -> None:
    cleanup()
    preflight_checks()

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HOME"] = str(THESIS_ROOT / "cache" / "huggingface")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{existing_pp}:{THESIS_ROOT}:{os.getcwd()}".strip(":")

    print("=" * 72)
    print(f"AVVIO SMOKE TEST COMPLETO WEEKEND ({len(SMOKE_CONFIGURATIONS)} CONFIGURAZIONI, 2 GPU DDP)")
    print("=" * 72)

    t0_total = time.time()

    try:
        for idx, exp in enumerate(SMOKE_CONFIGURATIONS, start=1):
            t0 = time.time()
            print(
                f"\n--- [{idx}/{len(SMOKE_CONFIGURATIONS)}] {exp.note} "
                f"(LR={exp.lr}, Accum={exp.grad_accum}, R={exp.lora_rank}, A={exp.lora_alpha}) ---"
            )

            cmd = [
                str(TORCHRUN),
                "--standalone",
                "--nnodes", "1",
                "--nproc-per-node", "2",
                str(TRAIN_WRAPPER),
                "--vla_path", "openvla/openvla-7b",
                "--dataset_name", DATASET_NAME,
                "--data_root_dirs", str(exp.data_dirs),
                "--dataset_weights", str(exp.data_weights),
                "--val_file", VAL_FILE,
                "--run_root_dir", str(SMOKE_RUN_ROOT),
                "--adapter_tmp_dir", str(SMOKE_ADAPTER_ROOT),
                "--use_lora", "True",
                "--lora_rank", str(exp.lora_rank),
                "--lora_alpha", str(exp.lora_alpha),
                "--lora_dropout", "0.05",
                "--use_quantization", "False",
                "--batch_size", str(exp.batch_size),
                "--grad_accumulation_steps", str(exp.grad_accum),
                "--learning_rate", exp.lr,
                "--max_steps", "4",
                "--warmup_steps", "1",
                "--save_steps", "2",
                "--image_aug", "True",
                "--shuffle_buffer_size", "256",
                "--val_max_batches", "4",
                "--run_id_note", exp.note,
                "--save_latest_checkpoint_only", "False",
                "--target_modules", "all-linear",
                "--early_stopping_enabled", "True",
                "--early_stopping_target_acc", "0.0",
                "--early_stopping_acc_window", "2",
                "--early_stopping_patience", "2",
                "--early_stopping_min_rel_delta", "0.005",
                "--early_stopping_ema_alpha", "0.5",
                "--early_stopping_l1_source", "val",
            ]

            found_v5_provenance = False
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if EXPECTED_V5_TAG in line:
                    found_v5_provenance = True

            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)

            if idx == 1 and not found_v5_provenance:
                raise RuntimeError(
                    f"rlds_base NON contiene {EXPECTED_V5_TAG}! "
                    "Esegui prima: python integrations/openvla/build_dataset.py "
                    f"--source hdf5_data/{EXPECTED_V5_TAG}/splits --variant base --publish"
                )

            best_dirs = list(SMOKE_ADAPTER_ROOT.glob(f"*{exp.note}*--best_chkpt"))
            es_states = list(SMOKE_RUN_ROOT.glob(f"*{exp.note}*/early_stopping_state.json"))
            if len(best_dirs) != 1 or len(es_states) != 1:
                raise RuntimeError(
                    f"[{exp.note}] Artefatti mancanti: best_chkpt={len(best_dirs)}, es_state={len(es_states)}"
                )

            best_dir = best_dirs[0]
            if not (best_dir / "dataset_statistics.json").is_file() or not (best_dir / "adapter_config.json").is_file():
                raise RuntimeError(f"[{exp.note}] File incompleti dentro {best_dir}")

            print(f"[{idx}/{len(SMOKE_CONFIGURATIONS)}] {exp.note} OK in {time.time() - t0:.1f}s")

        print("\n" + "=" * 72)
        print(
            f"TUTTE E {len(SMOKE_CONFIGURATIONS)} LE CONFIGURAZIONI HANNO SUPERATO LO SMOKE TEST "
            f"IN {time.time() - t0_total:.1f}s! (Dataset attivo verificato: v5.0)"
        )
        print("Puoi lanciare il batch del weekend dentro tmux in totale sicurezza.")
        print("=" * 72)
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[SMOKE TEST FALLITO]: {exc}", file=sys.stderr)
        sys.exit(1)