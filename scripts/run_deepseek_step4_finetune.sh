#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# DeepSeek-V2-Lite Step 4: pFairFT 精准微调 (Precision Fine-tuning)
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
FINETUNE_DIR="${RESULTS_ROOT}/precision_fairness"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

mkdir -p "${FINETUNE_DIR}"

echo "============================================"
echo "Step 4: pFairFT Precision Fine-tuning"
echo "============================================"

# LoRA Training Parameters
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
NUM_EPOCHS=3
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
FAIRNESS_LAMBDA=0.1

python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${FINETUNE_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --fairness_lambda "${FAIRNESS_LAMBDA}" \
    --seed 42

echo ">>> Step 4 DONE."
echo ">>> Fine-tuned model saved to: ${FINETUNE_DIR}/final_model"
