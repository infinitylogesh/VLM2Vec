#!/bin/bash
set -e

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

cd /workspace/VLM2Vec

BASE_MODEL="infinitylogesh/Qwen3-0.8b-image-embeddings-merged-4750"

CHECKPOINTS=(
  "/workspace/VLM2Vec/outputs/checkpoint-250"
  "/workspace/VLM2Vec/outputs/checkpoint-750"
  "/workspace/VLM2Vec/outputs/checkpoint-1250"
  "/workspace/VLM2Vec/outputs/checkpoint-1750"
  "/workspace/VLM2Vec/outputs/checkpoint-2250"
)

DATA_BASEDIR="data/vlm2vec_eval"
OUTPUT_BASEDIR="outputs/evaluation"
BATCH_SIZE=16
DATA_CONFIG="experiments/public/eval/visdoc.yaml"

for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
  CKPT_NAME=$(basename "$CHECKPOINT_PATH")
  OUTPUT_PATH="${OUTPUT_BASEDIR}/${CKPT_NAME}/"
  mkdir -p "$OUTPUT_PATH"

  echo ""
  echo "############################################################"
  echo "# Checkpoint: $CKPT_NAME  —  visdoc"
  echo "############################################################"

  uv run python eval.py \
    --model_name "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --lora True \
    --pooling eos \
    --normalize True \
    --per_device_eval_batch_size $BATCH_SIZE \
    --dataset_config "$DATA_CONFIG" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee "${OUTPUT_PATH}/eval_visdoc.log"

  echo "Done: $CKPT_NAME / visdoc"
done

echo ""
echo "All visdoc checkpoints complete. Results in $OUTPUT_BASEDIR/"
