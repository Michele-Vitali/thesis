#!/usr/bin/env bash

set -o pipefail
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH}:${pwd}"

THESIS_ROOT="$HOME/Scrivania/thesis"
OPENVLA_ENV="$THESIS_ROOT/openvla_env"

DATA_ROOT_BASE="$THESIS_ROOT/rlds_data/rlds_base"
# DATA_ROOT_LIGHTS="$THESIS_ROOT/rlds_data/rlds_lights" # (Decommenta quando avrai il secondo dataset)

DATA_DIRS=("$DATA_ROOT_BASE")   # When we gathe rthe lights dataset, we add: , "$DATA_ROOT_LIGHTS")
DATA_WEIGHTS=(1.0)  # When other dataset are ready we add: , 0.3, ...)

DATA_DIRS_STR="["
for i in "${!DATA_DIRS[@]}"; do
    if [ $i -gt 0 ]; then DATA_DIRS_STR+=", "; fi
    DATA_DIRS_STR+="'${DATA_DIRS[$i]}'"
done
DATA_DIRS_STR+="]"

DATA_WEIGHTS_STR="["
for i in "${!DATA_WEIGHTS[@]}"; do
    if [ $i -gt 0 ]; then DATA_WEIGHTS_STR+=", "; fi
    DATA_WEIGHTS_STR+="${DATA_WEIGHTS[$i]}"
done
DATA_WEIGHTS_STR+="]"

RUN_ROOT="$THESIS_ROOT/runs/openvla"
ADAPTER_ROOT="$THESIS_ROOT/adapters/openvla"
HF_HOME_DIR="$THESIS_ROOT/cache/huggingface"
VAL_FILE="$THESIS_ROOT/hdf5_data/v2.1_no_randomization_20261103_125141_869177/splits/validation.hdf5"

TRAIN_WRAPPER="$THESIS_ROOT/integrations/openvla/finetune.py"
MODEL="openvla/openvla-7b"
DATASET="prob3_pick_place"

NUM_GPUS=2
USE_LORA=True
LORA_DROPOUT=0.05
USE_QUANTIZATION=False
BATCH_SIZE=4
IMAGE_AUG=True
SHUFFLE_BUFFER_SIZE=100000
SAVE_LATEST_CHECKPOINT_ONLY=False

MAX_STEPS=23000
WARMUP_STEPS=1500
SAVE_STEPS=500

export WANDB_MODE=online
export HF_HOME="$HF_HOME_DIR"
export CUDA_VISIBLE_DEVICES=0,1

CONFIGURATIONS=(
    "1e-4 6 32 v2.1.2_baseline_LR1e-4_Acc6_R32"
    "5e-5 6 32 v2.1.2_low_lr_LR5e-5_Acc6_R32"
    "1e-4 2 32 v2.1.2_low_accum_LR1e-4_Acc2_R32"
    "1e-4 6 64 v2.1.2_high_rank_LR1e-4_Acc6_R64"
    "5e-5 4 32 v2.1.2_mid_accum_cautious_LR5e-5_Acc4_R32"
)

for CONFIG in "${CONFIGURATIONS[@]}"; do
    read -r CURRENT_LR CURRENT_ACCUM CURRENT_RANK CURRENT_NOTE <<< "$CONFIG"

    echo "============================================================"
    echo "INIZIO TRAINING: $CURRENT_NOTE (LR=$CURRENT_LR | Accum=$CURRENT_ACCUM | Rank=$CURRENT_RANK)"
    echo "============================================================"

    CMD=(
        "$OPENVLA_ENV/bin/torchrun"
        --standalone
        --nnodes 1
        --nproc-per-node "$NUM_GPUS"

        "$TRAIN_WRAPPER"
        
        --vla_path "$MODEL"
        --dataset_name "$DATASET"
        
        --data_root_dirs "$DATA_DIRS_STR"
        --dataset_weights "$DATA_WEIGHTS_STR"

        --val_file "$VAL_FILE"

        --run_root_dir "$RUN_ROOT"
        --adapter_tmp_dir "$ADAPTER_ROOT"
        --use_lora "$USE_LORA"
        --lora_rank "$CURRENT_RANK"
        --lora_dropout "$LORA_DROPOUT"
        --use_quantization "$USE_QUANTIZATION"
        --batch_size "$BATCH_SIZE"
        --grad_accumulation_steps "$CURRENT_ACCUM"
        --learning_rate "$CURRENT_LR"
        --max_steps "$MAX_STEPS"
        --warmup_steps "$WARMUP_STEPS"
        --save_steps "$SAVE_STEPS"
        --image_aug "$IMAGE_AUG"
        --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
        --run_id_note "$CURRENT_NOTE"
        --save_latest_checkpoint_only "$SAVE_LATEST_CHECKPOINT_ONLY"
    )

    "${CMD[@]}"

    echo -e "============================================================\nTRAINING $CURRENT_NOTE COMPLETATO!\n============================================================\n"
done