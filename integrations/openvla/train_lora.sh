#!/usr/bin/env bash

set -o pipefail

# Remove secondary warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1

# Project directories
THESIS_ROOT="$HOME/Scrivania/thesis"
OPENVLA_ENV="$THESIS_ROOT/openvla_env"
DATA_ROOT="$THESIS_ROOT/rlds_data/rlds"
RUN_ROOT="$THESIS_ROOT/runs/openvla"
ADAPTER_ROOT="$THESIS_ROOT/adapters/openvla"
HF_HOME_DIR="$THESIS_ROOT/cache/huggingface"

# Train wrapper script (switch to integrations/openvla/finetune.py for full training)
TRAIN_WRAPPER="$THESIS_ROOT/integrations/openvla/finetune_overfit_test.py"

# Current base model and dataset
MODEL="openvla/openvla-7b"
DATASET="prob3_pick_place"

# Training parameters
NUM_GPUS=2

# LoRA parameters
USE_LORA=True
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.0    # 0.05 for generalizing!
USE_QUANTIZATION=False

# Training hyperparameters
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-4  # In VLA paper it is described as the optimal value!

N_EPOCHS=27 # In VLA paper it is described as the optimal value!

MAX_STEPS=2000 # For a 500 trajectories dataset of 180 frames per trajectory, set to 40k (~30 epochs)
SAVE_STEPS=250  # Set to 10% of the max_steps so we get 10 checkpoints
IMAGE_AUG=False # True for generalizing!
SHUFFLE_BUFFER_SIZE=270 # Set to 100k on big dataset!
SAVE_LATEST_CHECKPOINT_ONLY=False

# Run id
clearRUN_NOTE="v7_overfit_smooth_second_try"

# Wandb logging, online mode so we can track the loss!
export WANDB_MODE=offline

# HuggingFace cache directory
export HF_HOME="$HF_HOME_DIR"

# Cuda visible devices
export CUDA_VISIBLE_DEVICES=0,1

# Build torchrun command array
CMD=(
    "$OPENVLA_ENV/bin/torchrun"
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

    --save_latest_checkpoint_only
    "$SAVE_LATEST_CHECKPOINT_ONLY"
)

DRY_RUN=0

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

exec "${CMD[@]}" "$@"