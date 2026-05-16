#!/bin/bash

set -e

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

cd /workspace/VLM2Vec

# ==============================================================================
# Configuration — update these two paths before running
# ==============================================================================
BASE_MODEL="Qwen/Qwen3.5-2b"
CHECKPOINT_PATH=".../Qwen3.5-2b-hatefulmemes/checkpoint-50"   # e.g. .../Qwen3.5-2b-hatefulmemes/checkpoint-50

DATA_BASEDIR="data/vlm2vec_eval"
OUTPUT_BASEDIR="exps/qwen3.5-eval/$(basename $CHECKPOINT_PATH)"
BATCH_SIZE=16

MODALITIES=("image") #"video" "visdoc")

# ==============================================================================
# Eval loop
# ==============================================================================
for MODALITY in "${MODALITIES[@]}"; do
  DATA_CONFIG="experiments/public/eval/${MODALITY}.yaml"
  OUTPUT_PATH="${OUTPUT_BASEDIR}/${MODALITY}/"

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

  echo "Done: $MODALITY"
done

echo "All modalities complete. Results in $OUTPUT_BASEDIR"
