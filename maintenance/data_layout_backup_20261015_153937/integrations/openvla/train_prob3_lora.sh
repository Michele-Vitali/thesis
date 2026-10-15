#!/usr/bin/env bash

set -euo pipefail


# ================================================================
# PROJECT
# ================================================================

THESIS_ROOT="$HOME/Scrivania/thesis"

OPENVLA_ENV="$THESIS_ROOT/openvla_env"

DATA_ROOT="$THESIS_ROOT/data/rlds"

RUN_ROOT="$THESIS_ROOT/runs/openvla"

ADAPTER_ROOT="$THESIS_ROOT/adapters/openvla"

HF_HOME_DIR="$THESIS_ROOT/cache/huggingface"

TRAIN_WRAPPER="$THESIS_ROOT/integrations/openvla/finetune_prob3.py"


# ================================================================
# MODEL / DATASET
# ================================================================

MODEL="openvla/openvla-7b"

DATASET="prob3_pick_place"


# ================================================================
# DISTRIBUTED TRAINING
# ================================================================

# You have 2 x RTX PRO 5000 Blackwell.
#
# We can eventually use both.
#
# Set to 1 initially if you want a conservative smoke-training run.
# Set to 2 for the actual multi-GPU LoRA run.

NUM_GPUS=2


# ================================================================
# LoRA
# ================================================================

USE_LORA=True

LORA_RANK=32

LORA_DROPOUT=0.0

USE_QUANTIZATION=False


# ================================================================
# TRAINING HYPERPARAMETERS
#
# THESE ARE INTENTIONALLY EASY TO EDIT BEFORE TRAINING.
# ================================================================

BATCH_SIZE=8

GRAD_ACCUMULATION_STEPS=1

LEARNING_RATE=5e-4

MAX_STEPS=10000

SAVE_STEPS=1000

IMAGE_AUG=True

SHUFFLE_BUFFER_SIZE=10000


# ================================================================
# RUN IDENTIFICATION
# ================================================================

RUN_NOTE="prob3_lora"


# ================================================================
# WANDB
#
# Keep offline for now.
#
# Change this only when / if you want cloud logging.
# ================================================================

export WANDB_MODE=offline


# ================================================================
# HUGGING FACE CACHE
# ================================================================

export HF_HOME="$HF_HOME_DIR"


# ================================================================
# CUDA
# ================================================================

export CUDA_VISIBLE_DEVICES=0,1


# ================================================================
# BUILD COMMAND
# ================================================================

CMD=(
    torchrun
    --standalone
    --nnodes
    1
    --nproc-per-node
    "$NUM_GPUS"

    "$TRAIN_WRAPPER"

    --vla_path
    "$MODEL"

    --data_root_dir
    "$DATA_ROOT"

    --dataset_name
    "$DATASET"

    --run_root_dir
    "$RUN_ROOT"

    --adapter_tmp_dir
    "$ADAPTER_ROOT"

    --use_lora
    "$USE_LORA"

    --lora_rank
    "$LORA_RANK"

    --lora_dropout
    "$LORA_DROPOUT"

    --use_quantization
    "$USE_QUANTIZATION"

    --batch_size
    "$BATCH_SIZE"

    --grad_accumulation_steps
    "$GRAD_ACCUMULATION_STEPS"

    --learning_rate
    "$LEARNING_RATE"

    --max_steps
    "$MAX_STEPS"

    --save_steps
    "$SAVE_STEPS"

    --image_aug
    "$IMAGE_AUG"

    --shuffle_buffer_size
    "$SHUFFLE_BUFFER_SIZE"

    --run_id_note
    "$RUN_NOTE"
)


# ================================================================
# SAFETY
#
# IMPORTANT:
#
# DRY_RUN=1 means:
#
#   PRINT COMMAND ONLY
#
# No training is started.
# ================================================================

DRY_RUN=1


if [[ "$DRY_RUN" == "1" ]]; then

    echo
    echo "============================================================"
    echo "OpenVLA-7B P-Rob3 LoRA"
    echo "============================================================"
    echo
    echo "DRY RUN ONLY."
    echo
    echo "The following command WOULD be executed:"
    echo

    printf '%q ' "${CMD[@]}"

    echo
    echo
    echo "NO TRAINING STARTED."
    echo

    exit 0
fi


# ================================================================
# ACTUAL TRAINING
#
# This line is unreachable while DRY_RUN=1.
# ================================================================

exec "${CMD[@]}"
