#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Qwen1.5-MoE-A2.7B-Chat  Step 2: Component Identification
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_NAME="Qwen1.5-MoE-A2.7B-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"
RESUME_JSON="${DATA_DIR}/resume/qwen_summaries_with_race.json"

BIASED_DIR="${RESULTS_ROOT}/biased_samples"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"

mkdir -p "${HEADS_DIR}"

echo "============================================"
echo "Step 2: Component Identification — ${MODEL_NAME}"
echo "============================================"

# Step 2a: Calculate biased samples ranking
echo ">>> Step 2a: Evaluate Biased Samples"
python "${PROJECT_ROOT}/2_component_identification/evaluate_biased_sample.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${RESUME_JSON}" \
    --output_csv_path "${CSV_PATH}" \
    --device "cuda" \
    --model_type "qwen"

echo ">>> Step 2a DONE."

echo ">>> Step 2b: Analyze Race Sensitive Heads"
python "${PROJECT_ROOT}/2_component_identification/analyze_race_sensitive_heads.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${RESUME_JSON}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --output_dir "${HEADS_DIR}" \
    --batch_size 4 \
    --device "cuda" \
    --model_type "qwen"

echo ">>> Step 2 DONE. Results: ${HEADS_DIR}"
