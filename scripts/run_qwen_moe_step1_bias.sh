#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Qwen1.5-MoE-A2.7B-Chat  Step 1: Bias Evaluation (Discrim-Eval)
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_NAME="Qwen1.5-MoE-A2.7B-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"
DISCRIM_JSON="${DATA_DIR}/discrim-eval/dataset_paired.json"

BIAS_OUTPUT="${RESULTS_ROOT}/bias_evaluation"
mkdir -p "${BIAS_OUTPUT}"

echo "============================================"
echo "Step 1: Bias Evaluation — ${MODEL_NAME}"
echo "============================================"

python "${PROJECT_ROOT}/1_bias_evaluation/evaluate_bias_discrim.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DISCRIM_JSON}" \
    --model_type "qwen" \
    --prompt_type "prompt" \
    --csv_path "${BIAS_OUTPUT}/per_sample_details.csv"

python "${PROJECT_ROOT}/1_bias_evaluation/evaluate_bias_discrim.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DISCRIM_JSON}" \
    --model_type "qwen" \
    --prompt_type "debiased_prompt" \
    --csv_path "${BIAS_OUTPUT}/per_sample_details.csv"

echo ">>> Step 1 DONE. Results: ${BIAS_OUTPUT}"
