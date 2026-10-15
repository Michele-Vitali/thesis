
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="$ROOT/tools/openvla:$ROOT/integrations/openvla${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/huggingface"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "$ROOT/runs/openvla" "$ROOT/adapters/openvla"

exec "$ROOT/openvla_env/bin/torchrun" \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 2 \
    "$ROOT/integrations/openvla/finetune_prob3.py" \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$ROOT/data/rlds" \
    --dataset_name prob3_pick_place \
    --run_root_dir "$ROOT/runs/openvla" \
    --adapter_tmp_dir "$ROOT/adapters/openvla" \
    --use_lora True \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --use_quantization False \
    --batch_size 8 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --max_steps 8000 \
    --save_steps 2000 \
    --image_aug True \
    --shuffle_buffer_size 10000 \
    --run_id_note prob3_first_8k
