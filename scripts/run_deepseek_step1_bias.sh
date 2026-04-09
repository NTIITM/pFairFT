#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# DeepSeek-V2-Lite 全流程实验脚本
# 使用 GPU 5,6,7 运行 pFairFT pipeline
# ============================================================================

export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_PATH="/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_NAME="DeepSeek-V2-Lite-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"

# Resume 数据集路径
RESUME_JSON="${DATA_DIR}/resume/qwen_summaries_with_race.json"
DISCRIM_JSON="${DATA_DIR}/discrim-eval/dataset_paired.json"

mkdir -p "${RESULTS_ROOT}"

echo "============================================"
echo "DeepSeek-V2-Lite pFairFT Experiment Pipeline"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Model: ${MODEL_PATH}"
echo "Results: ${RESULTS_ROOT}"
echo "============================================"

# ============================================================================
# Step 1: 偏见评估 (Bias Evaluation)
# ============================================================================
echo ""
echo ">>> Step 1: Bias Evaluation (Discrim-Eval)"
echo "============================================"

BIAS_OUTPUT="${RESULTS_ROOT}/bias_evaluation"
mkdir -p "${BIAS_OUTPUT}"

# Original prompt
python "${PROJECT_ROOT}/1_bias_evaluation/evaluate_bias_discrim.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DISCRIM_JSON}" \
    --model_type "deepseek" \
    --prompt_type "prompt" \
    --csv_path "${BIAS_OUTPUT}/per_sample_details.csv"

# Debiased prompt
python "${PROJECT_ROOT}/1_bias_evaluation/evaluate_bias_discrim.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DISCRIM_JSON}" \
    --model_type "deepseek" \
    --prompt_type "debiased_prompt" \
    --csv_path "${BIAS_OUTPUT}/per_sample_details.csv"

echo ">>> Step 1 DONE. Results: ${BIAS_OUTPUT}"
