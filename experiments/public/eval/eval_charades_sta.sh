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
BATCH_SIZE=4   # reduced batch size as the eval code does for Charades-STA

for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
  CKPT_NAME=$(basename "$CHECKPOINT_PATH")
  OUTPUT_PATH="${OUTPUT_BASEDIR}/${CKPT_NAME}/"
  mkdir -p "$OUTPUT_PATH"

  echo "============================================================"
  echo "Checkpoint: $CKPT_NAME  →  Charades-STA"
  echo "============================================================"

  # Write a single-task yaml on the fly
  CHARADES_YAML=$(mktemp /tmp/charades_sta_XXXX.yaml)
  cat > "$CHARADES_YAML" << 'YAML'
Charades-STA:
    dataset_parser: moment_retrieval
    dataset_name: Charades-STA
    video_root: ""
    clip_root: ""
    frame_root: video-tasks/frames/video_mret/Charades-STA
    num_negative_clips: 9
    max_video_frames_saved: 64
    max_clip_frames_saved: 8
    num_video_frames: 8
    num_clip_frames: 8
    eval_type: local
YAML

  uv run python eval.py \
    --model_name "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --lora True \
    --pooling eos \
    --normalize True \
    --per_device_eval_batch_size $BATCH_SIZE \
    --dataset_config "$CHARADES_YAML" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee "${OUTPUT_PATH}/eval_charades_sta.log"

  rm -f "$CHARADES_YAML"
  echo "Done: $CKPT_NAME / Charades-STA"
done

echo ""
echo "All checkpoints done. Check outputs/evaluation/*/Charades-STA_score.json"
