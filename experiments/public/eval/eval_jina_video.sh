#!/bin/bash
# Evaluate Jina v5-omni models on MMEB video benchmark
# Note: Jina encodes single frames (first frame per clip) as image proxy for video.
set -e

cd /workspace/VLM2Vec

export HF_DATASETS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

DATA_BASEDIR="data/vlm2vec_eval"
DATASET_CONFIG="experiments/public/eval/video.yaml"
BATCH_SIZE=8

MODELS=(
  "jinaai/jina-embeddings-v5-omni-nano"
  "jinaai/jina-embeddings-v5-omni-small"
)

OUTPUT_NAMES=(
  "jina-v5-omni-nano"
  "jina-v5-omni-small"
)

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  NAME="${OUTPUT_NAMES[$i]}"
  OUTPUT_DIR="outputs/evaluation/${NAME}"

  echo ""
  echo "############################################################"
  echo "# Model: $MODEL  (video — first-frame proxy)"
  echo "# Output: $OUTPUT_DIR"
  echo "############################################################"

  mkdir -p "$OUTPUT_DIR"

  uv run python experiments/eval_jina_mmeb.py \
    --model_name "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_config "$DATASET_CONFIG" \
    --data_basedir "$DATA_BASEDIR" \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "${OUTPUT_DIR}/eval_video.log"

  echo "Done: $NAME / video"
done

echo ""
echo "All Jina video evaluations complete."
