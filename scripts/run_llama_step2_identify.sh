#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Llama-V2-Lite Step 2: 组件识别 (Component Identification)
# 使用 GPU 5,6,7
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct"
MODEL_NAME="Llama-3.2-3B-Instruct"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"

RESUME_JSON="${DATA_DIR}/resume/qwen_summaries_with_race.json"

mkdir -p "${RESULTS_ROOT}"

echo "============================================"
echo "Step 2: Component Identification (Sensitive Heads)"
echo "============================================"

# Step 2a: 找到最 biased 的样本
BIASED_DIR="${RESULTS_ROOT}/biased_samples"
mkdir -p "${BIASED_DIR}"

python "${PROJECT_ROOT}/2_component_identification/evaluate_biased_sample.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${RESUME_JSON}" \
    --output_csv_path "${BIASED_DIR}/biased_samples_ranking.csv" \
    --device "cuda" \
    --model_type "llama"

echo ">>> Step 2a (biased samples) DONE."

# Step 2b: 分析种族敏感头
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"
mkdir -p "${HEADS_DIR}"

CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: biased_samples_ranking.csv not found at $CSV_PATH"
    exit 1
fi

python "${PROJECT_ROOT}/2_component_identification/analyze_race_sensitive_heads.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${RESUME_JSON}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --output_dir "${HEADS_DIR}" \
    --batch_size 4 \
    --device "cuda" \
    --model_type "llama"

echo ">>> Step 2b (sensitive heads) DONE."
echo ">>> Results: ${HEADS_DIR}"
