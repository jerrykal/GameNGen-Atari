#!/usr/bin/env bash

dataset_path=${DATASET_PATH:-"data/gamengen/pong"}
dataset_name=${DATASET_NAME:-$(basename "$dataset_path")}

seed=${SEED:-42}
max_train_steps=${MAX_TRAIN_STEPS:-10000}
learning_rate=${LEARNING_RATE:-2e-5}

total_batch_size=${TOTAL_BATCH_SIZE:-16}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-16}
gradient_accumulation_steps=$((total_batch_size / batch_size_per_device))
checkpoints_total_limit=${CHECKPOINTS_TOTAL_LIMIT:-2}
checkpointing_steps=${CHECKPOINTING_STEPS:-1000}

name_suffix="bsz${total_batch_size}_st${max_train_steps}_$(date +%Y%m%d_%H%M)"
output_dir=${OUTPUT_DIR:-"saves/latent_decoder/${dataset_name}_${name_suffix}"}

args=(
    --output_dir="$output_dir"
    --dataset_name="$dataset_path"
    --seed="$seed"
    --max_train_steps="$max_train_steps"
    --learning_rate="$learning_rate"
    --train_batch_size="$batch_size_per_device"
    --gradient_accumulation_steps="$gradient_accumulation_steps"
    --checkpoints_total_limit="$checkpoints_total_limit"
    --checkpointing_steps="$checkpointing_steps"
    --resume_from_checkpoint="latest"
    "$@"
)

# NOTE: Temporary set num_noise_buckets to 0 to avoid noise augmentation
# python -m gamengen.train "${args[@]}"
accelerate launch --mixed_precision="bf16" -m gamengen.finetune_latent_decoder "${args[@]}"