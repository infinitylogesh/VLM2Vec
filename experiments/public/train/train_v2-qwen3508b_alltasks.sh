#!/bin/bash

set -e

# NOTE: replace ... with actual paths
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec-debug
export WANDB_API_KEY=
export HUGGING_FACE_HUB_TOKEN=
export WANDB_PROJECT=vlm2vec-debug
export WANDB_RUN_GROUP=Qwen3.5-08b-continued-mixed-alltasks
export EXP_NAME=Qwen3.5-08b-continued-mixed-alltasks

export WORKDIR=/workspace/VLM2Vec
export WANDB_NAME=$EXP_NAME
export EXP_DIR="${WORKDIR}/outputs/${EXP_NAME}"
export WANDB_DIR=$EXP_DIR
MODEL="infinitylogesh/Qwen3-0.8b-image-embeddings-merged-4750"
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /workspace/VLM2Vec
cmd="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=2207 --max_restarts=0 train.py \
  --lora \
  --lora_r 16 \
  --model_name $MODEL \
  --bf16 \
  --pooling eos \
  --normalize True \
  --temperature 0.02 \
  --dataset_config /workspace/VLM2Vec/experiments/public/train/train_alltasks.yaml \
  --run_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --grad_cache True \
  --per_device_train_batch_size 192 \
  --gc_q_chunk_size 8 \
  --gc_p_chunk_size 8 \
  --interleave_batch_size 64 \
  --lr_scheduler_type linear \
  --learning_rate 2e-5 \
  --max_steps 5000 \
  --warmup_steps 100 \
  --save_steps 250 \
  --logging_steps 1 \
  --remove_unused_columns False \
  --gradient_accumulation_steps 1 \
  --resume_from auto \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 2 \
  --dataloader_num_workers 4 \
  --ddp_find_unused_parameters False \
  --report_to wandb \
  2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
