#!/bin/bash

set -e

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

cd /workspace/VLM2Vec

# ==============================================================================
# Configuration
# ==============================================================================
BASE_MODEL="Qwen/Qwen3.5-0.8b"

# List of checkpoint paths to evaluate one by one
CHECKPOINTS=(
  "/workspace/VLM2Vec/outputs/checkpoint-4000"
  "/workspace/VLM2Vec/outputs/checkpoint-3000"
  "/workspace/VLM2Vec/outputs/checkpoint-2000"
  "/workspace/VLM2Vec/outputs/checkpoint-1000"
  "/workspace/VLM2Vec/outputs/checkpoint-500"
  # Add more checkpoints here, e.g.:
  # "/workspace/VLM2Vec/outputs/checkpoint-3750"
)

DATA_BASEDIR="data/data/vlm2vec_eval"
OUTPUT_BASEDIR="outputs/evaluation"
BATCH_SIZE=16

MODALITIES=("image" "video" "visdoc") # add "video" "visdoc" as needed

# ==============================================================================
# Checkpoint loop
# ==============================================================================
for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
  CKPT_NAME=$(basename "$CHECKPOINT_PATH")

  echo ""
  echo "############################################################"
  echo "# Checkpoint: $CKPT_NAME"
  echo "# Path:       $CHECKPOINT_PATH"
  echo "############################################################"

  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG="experiments/public/eval/${MODALITY}.yaml"
    OUTPUT_PATH="${OUTPUT_BASEDIR}/${CKPT_NAME}/"

    echo "================================================="
    echo "Modality: $MODALITY"
    echo "Output:   $OUTPUT_PATH"
    echo "================================================="

    mkdir -p "$OUTPUT_PATH"

    python eval.py \
      --model_name "$BASE_MODEL" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --pooling eos \
      --normalize True \
      --per_device_eval_batch_size $BATCH_SIZE \
      --dataset_config "$DATA_CONFIG" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR" \
      2>&1 | tee "${OUTPUT_PATH}/eval.log"

    echo "Done: $CKPT_NAME / $MODALITY"
  done
done

echo ""
echo "All checkpoints complete. Results in $OUTPUT_BASEDIR/"
