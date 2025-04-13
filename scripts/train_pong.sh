#!/usr/bin/env bash

output_dir=${OUTPUT_DIR:-"saves/gamengen_pong_$(date +%Y%m%d_%H%M)"}
dataset_name=${DATASET_NAME:-"data/gamengen/pong"}
seed=${SEED:-42}
max_train_steps=${MAX_TRAIN_STEPS:-700000}
learning_rate=${LEARNING_RATE:-2e-5}
context_length=${CONTEXT_LENGTH:-64}

total_batch_size=${TOTAL_BATCH_SIZE:-16}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-2}
gradient_accumulation_steps=$((total_batch_size / batch_size_per_device))

args=(
    --output_dir="$output_dir"
    --dataset_name="$dataset_name"
    --seed="$seed"
    --num_noise_buckets=0
    --context_length="$context_length"
    --max_train_steps="$max_train_steps"
    --learning_rate="$learning_rate"
    --train_batch_size="$batch_size_per_device"
    --gradient_accumulation_steps="$gradient_accumulation_steps"
    --checkpoints_total_limit=2
    "$@"
)

# NOTE: Temporary set num_noise_buckets to 0 to avoid noise augmentation
# python -m gamengen.train "${args[@]}"
accelerate launch --mixed_precision="bf16" -m gamengen.train "${args[@]}"