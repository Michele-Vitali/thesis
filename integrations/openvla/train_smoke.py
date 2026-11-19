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

RLDS_BASE = str(THESIS_ROOT / "rlds_data" / "rlds_base")
RLDS_LIGHTS = str(THESIS_ROOT / "rlds_data" / "rlds_lights")
VAL_FILE = str(
    THESIS_ROOT
    / "hdf5_data"
    / "v5.0_no_randomization_better_camera_perturbed_20261113_133035_010799"
    / "splits"
    / "val.hdf5"
)

SMOKE_RUN_ROOT = Path("/tmp/openvla_smoke_runs")
SMOKE_ADAPTER_ROOT = Path("/tmp/openvla_smoke_adapters")


@dataclass(frozen=True)
class SmokeExperiment:
    note: str
    data_dirs: List[str]
    data_weights: List[float]
    lr: str
    batch_size: int
    grad_accum: int
    lora_rank: int


SMOKE_CONFIGURATIONS: List[SmokeExperiment] = [
    SmokeExperiment(
        note="SMOKE_1_base_LR1e_4_ACC1_BES16_R32",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="1e-4",
        batch_size=8,
        grad_accum=1,
        lora_rank=32,
    ),
    SmokeExperiment(
        note="SMOKE_2_base_LR3e_4_ACC2_BES32_R64",
        data_dirs=[RLDS_BASE],
        data_weights=[1.0],
        lr="3e-4",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
    ),
    SmokeExperiment(
        note="SMOKE_3_base_lights_LR1e_4_ACC1_BES16_R32",
        data_dirs=[RLDS_BASE, RLDS_LIGHTS],
        data_weights=[0.6, 0.4],
        lr="1e-4",
        batch_size=8,
        grad_accum=1,
        lora_rank=32,
    ),
    SmokeExperiment(
        note="SMOKE_4_base_lights_LR3e_4_ACC2_BES32_R64",
        data_dirs=[RLDS_BASE, RLDS_LIGHTS],
        data_weights=[0.6, 0.4],
        lr="3e-4",
        batch_size=8,
        grad_accum=2,
        lora_rank=64,
    ),
]


def cleanup() -> None:
    for p in (SMOKE_RUN_ROOT, SMOKE_ADAPTER_ROOT):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    cleanup()

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
    print(f"AVVIO SMOKE TEST COMPLETO SU {len(SMOKE_CONFIGURATIONS)} CONFIGURAZIONI (2 GPU DDP)")
    print("=" * 72)

    t0_total = time.time()

    try:
        for idx, exp in enumerate(SMOKE_CONFIGURATIONS, start=1):
            t0 = time.time()
            print(
                f"\n--- [{idx}/{len(SMOKE_CONFIGURATIONS)}] {exp.note} "
                f"(LR={exp.lr}, Accum={exp.grad_accum}, Rank={exp.lora_rank}, "
                f"Datasets={[Path(d).name for d in exp.data_dirs]}) ---"
            )

            cmd = [
                str(TORCHRUN),
                "--standalone",
                "--nnodes", "1",
                "--nproc-per-node", "2",
                str(TRAIN_WRAPPER),
                "--vla_path", "openvla/openvla-7b",
                "--dataset_name", "prob3_pick_place",
                "--data_root_dirs", str(exp.data_dirs),
                "--dataset_weights", str(exp.data_weights),
                "--val_file", VAL_FILE,
                "--run_root_dir", str(SMOKE_RUN_ROOT),
                "--adapter_tmp_dir", str(SMOKE_ADAPTER_ROOT),
                "--use_lora", "True",
                "--lora_rank", str(exp.lora_rank),
                "--lora_dropout", "0.0",
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

            subprocess.run(cmd, env=env, check=True)

            best_dirs = list(SMOKE_ADAPTER_ROOT.glob(f"*{exp.note}*--best_chkpt"))
            es_states = list(SMOKE_RUN_ROOT.glob(f"*{exp.note}*/early_stopping_state.json"))
            if len(best_dirs) != 1 or len(es_states) != 1:
                raise RuntimeError(
                    f"[{exp.note}] Artefatti mancanti: best_chkpt={len(best_dirs)}, es_state={len(es_states)}"
                )

            print(f"[{idx}/{len(SMOKE_CONFIGURATIONS)}] {exp.note} OK in {time.time() - t0:.1f}s")

        print("\n" + "=" * 72)
        print(
            f"TUTTE E {len(SMOKE_CONFIGURATIONS)} LE CONFIGURAZIONI HANNO SUPERATO LO SMOKE TEST "
            f"IN {time.time() - t0_total:.1f}s!"
        )
        print("Puoi lanciare train_lora.py per la notte in sicurezza.")
        print("=" * 72)
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[SMOKE TEST FALLITO]: {exc}", file=sys.stderr)
        sys.exit(1)