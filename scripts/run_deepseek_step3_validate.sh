#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# DeepSeek-V2-Lite Step 3: 干预验证 (Intervention Validation)
# 使用 GPU 5,6,7
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_NAME="DeepSeek-V2-Lite-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"

JSON_PATH="${DATA_DIR}/resume/qwen_summaries_with_race.json"
BIASED_DIR="${RESULTS_ROOT}/biased_samples"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"
VALIDATE_DIR="${RESULTS_ROOT}/intervention_validation"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
GLOBAL_CSV="${PROJECT_ROOT}/results/agg_intervention.csv"

mkdir -p "${VALIDATE_DIR}"

echo "============================================"
echo "Step 3: Intervention Validation"
echo "============================================"

# Baseline
echo ">>> Running Baseline (No Intervention)"
python "${PROJECT_ROOT}/2_component_identification/evaluate_intervention.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --output_dir "${VALIDATE_DIR}" \
    --device "cuda" \
    --model_type "deepseek" \
    --baseline \
    --csv_path "${GLOBAL_CSV}"

# Mean Replacement
echo ">>> Running Mean Replacement Intervention"
python "${PROJECT_ROOT}/2_component_identification/evaluate_intervention.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --sensitive_heads_dir "${HEADS_DIR}" \
    --intervention_mode "mean_replacement" \
    --output_dir "${VALIDATE_DIR}" \
    --device "cuda" \
    --model_type "deepseek" \
    --csv_path "${GLOBAL_CSV}"

# Debias Projection
echo ">>> Running Debias Projection Intervention"
python "${PROJECT_ROOT}/2_component_identification/evaluate_intervention.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --sensitive_heads_dir "${HEADS_DIR}" \
    --intervention_mode "debias_projection" \
    --intervention_strength 1.0 \
    --output_dir "${VALIDATE_DIR}" \
    --device "cuda" \
    --model_type "deepseek" \
    --csv_path "${GLOBAL_CSV}"

# Zero Value
echo ">>> Running Zero Value Intervention"
python "${PROJECT_ROOT}/2_component_identification/evaluate_intervention.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 100 \
    --sensitive_heads_dir "${HEADS_DIR}" \
    --intervention_mode "zero_value" \
    --output_dir "${VALIDATE_DIR}" \
    --device "cuda" \
    --model_type "deepseek" \
    --csv_path "${GLOBAL_CSV}"

echo ">>> Step 3 DONE."
echo ">>> Global CSV updated at: ${GLOBAL_CSV}"
