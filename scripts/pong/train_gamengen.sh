#!/usr/bin/env bash

dataset_path=${DATASET_PATH:-"data/gamengen/pong"}
output_name=$(basename "$dataset_path")

seed=${SEED:-42}
max_train_steps=${MAX_TRAIN_STEPS:-700000}
learning_rate=${LEARNING_RATE:-2e-5}
context_length=${CONTEXT_LENGTH:-4}
num_noise_buckets=${NUM_NOISE_BUCKETS:-10}

total_batch_size=${TOTAL_BATCH_SIZE:-16}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-16}
gradient_accumulation_steps=$((total_batch_size / batch_size_per_device))
validation_steps=${VALIDATION_STEPS:-1000}
checkpoints_total_limit=${CHECKPOINTS_TOTAL_LIMIT:-2}
checkpointing_steps=${CHECKPOINTING_STEPS:-1000}

render_width=${WIDTH:-256}
render_height=${HEIGHT:-256}

output_suffix="w${render_width}_h${render_height}_cl${context_length}_nb${num_noise_buckets}_bsz${total_batch_size}_st${max_train_steps}_$(date +%Y%m%d%H%M)"
output_dir=${OUTPUT_DIR:-"saves/gamengen/${output_name}_${output_suffix}"}

args=(
    --output_dir="$output_dir"
    --dataset_name="$dataset_path"
    --seed="$seed"
    --context_length="$context_length"
    --num_noise_buckets="$num_noise_buckets"
    --max_train_steps="$max_train_steps"
    --learning_rate="$learning_rate"
    --train_batch_size="$batch_size_per_device"
    --gradient_accumulation_steps="$gradient_accumulation_steps"
    --validation_steps="$validation_steps"
    --checkpoints_total_limit="$checkpoints_total_limit"
    --checkpointing_steps="$checkpointing_steps"
    --resume_from_checkpoint="latest"
    --render_width="$render_width"
    --render_height="$render_height"
    "$@"
)

# python -m gamengen.train "${args[@]}"
accelerate launch --mixed_precision="bf16" -m gamengen.train "${args[@]}"