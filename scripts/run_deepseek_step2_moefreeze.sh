#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# DeepSeek-V2-Lite-Chat Step 2: MoE Frozen Component Identification
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_NAME="DeepSeek-V2-Lite-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"
RESUME_JSON="${DATA_DIR}/resume/qwen_summaries_with_race.json"

echo "============================================"
echo "Step 2: Component Identification (MoE Frozen)"
echo "============================================"

BIASED_DIR="${RESULTS_ROOT}/biased_samples"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads_moefreeze"

mkdir -p "${HEADS_DIR}"

python "${PROJECT_ROOT}/2_component_identification/analyze_race_sensitive_heads_moefreeze.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${RESUME_JSON}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --output_dir "${HEADS_DIR}" \
    --batch_size 4 \
    --device "cuda" \
    --model_type "deepseek"

echo ">>> Step 2b (MoE frozen) DONE."
echo ">>> Results: ${HEADS_DIR}"
