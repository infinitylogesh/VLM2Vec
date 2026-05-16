#!/bin/bash

set -e

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec-debug
export WANDB_API_KEY=
export HUGGING_FACE_HUB_TOKEN=
export WANDB_RUN_GROUP=Qwen3.5-2b-videos
export EXP_NAME=Qwen3.5-2b-videos

export WANDB_NAME=$EXP_NAME
export EXP_DIR=.../$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /workspace/VLM2Vec
cmd="python3 train.py \
  --lora \
  --lora_r 16 \
  --model_name Qwen/Qwen3.5-2b \
  --bf16 \
  --pooling eos \
  --normalize True \
  --temperature 0.02 \
  --dataloader_num_workers 8 \
  --dataset_config /workspace/VLM2Vec/experiments/public/train/train_videos.yaml \
  --run_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --grad_cache True \
  --per_device_train_batch_size 128 \
  --gc_q_chunk_size 8 \
  --gc_p_chunk_size 8 \
  --interleave_batch_size 64 \
  --lr_scheduler_type linear \
  --learning_rate 5e-5 \
  --max_steps 10 \
  --warmup_steps 0 \
  --save_steps 100 \
  --logging_steps 1 \
  --remove_unused_columns False \
  --report_to none \
  2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
