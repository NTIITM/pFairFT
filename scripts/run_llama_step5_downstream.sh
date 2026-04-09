#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Llama-3.2-3B-Instruct Step 5: Downstream Evaluation
# GPU: 3,4
# ============================================================================

export CUDA_VISIBLE_DEVICES=3,4

BASE_MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct"
MODEL_NAME="Llama-3.2-3B-Instruct"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"

RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
BIASED_CSV="${RESULTS_ROOT}/biased_samples/biased_samples_ranking.csv"
FINAL_MODEL="${RESULTS_ROOT}/precision_fairness/final_model"
OUT_DIR="${RESULTS_ROOT}/downstream_evaluation"
mkdir -p "${OUT_DIR}"

echo "============================================"
echo "Step 5: Downstream Evaluations"
echo "============================================"

# ---------------------------------------------------------
# 1. MMLU
# ---------------------------------------------------------
echo ">>> Run 1: MMLU"
# Check base model MMLU
MMLU_BASE="${OUT_DIR}/mmlu_baseline.json"
if [ ! -f "${MMLU_BASE}" ]; then
    echo "Running MMLU for Base Model..."
    python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu.py" \
        --model_path "${BASE_MODEL_PATH}" \
        --output_json "${MMLU_BASE}" \
        --device "cuda"
else
    echo "Base MMLU already exists at ${MMLU_BASE}, skipping."
fi

# Finetuned Llama MMLU
MMLU_FT="${OUT_DIR}/mmlu_finetuned.json"
if [ ! -f "${MMLU_FT}" ]; then
    echo "Running MMLU for Finetuned Model..."
    python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu.py" \
        --model_path "${FINAL_MODEL}" \
        --output_json "${MMLU_FT}" \
        --device "cuda"
else
    echo "Finetuned MMLU already exists at ${MMLU_FT}, skipping."
fi

# ---------------------------------------------------------
# 2. Resume Task
# ---------------------------------------------------------
echo ">>> Run 2: Resume Fairness Task (Top 100)"

RESUME_BASE="${OUT_DIR}/resume_baseline.csv"
if [ ! -f "${RESUME_BASE}" ]; then
    echo "Running Resume Baseline..."
    python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
        --mode "baseline" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --biased_csv_path "${BIASED_CSV}" \
        --output_csv_path "${RESUME_BASE}" \
        --device "cuda" \
        --model_type "llama"
else
    echo "Resume Baseline already exists at ${RESUME_BASE}, skipping."
fi

RESUME_FT="${OUT_DIR}/resume_finetuned.csv"
if [ ! -f "${RESUME_FT}" ]; then
    echo "Running Resume Finetuned..."
    python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
        --mode "exp4" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --adapter_path "${FINAL_MODEL}" \
        --biased_csv_path "${BIASED_CSV}" \
        --output_csv_path "${RESUME_FT}" \
        --device "cuda" \
        --model_type "llama"
else
    echo "Resume Finetuned already exists at ${RESUME_FT}, skipping."
fi

# ---------------------------------------------------------
# 3. Discrim-Eval
# ---------------------------------------------------------
echo ">>> Run 3: Discrim-Eval"
# Based on existing results, base model discim_eval output is in bias_evaluation
DISCRIM_BASE="${RESULTS_ROOT}/bias_evaluation/per_sample_details.csv"
if [ ! -f "${DISCRIM_BASE}" ]; then
    echo "ERROR: Discrim Eval Base file not found! Expected at ${DISCRIM_BASE}"
else
    echo "Discrim Eval Baseline already exists, skipping base eval."
    cp "${DISCRIM_BASE}" "${OUT_DIR}/discrim_baseline.csv"
fi

DISCRIM_FT="${OUT_DIR}/discrim_finetuned.csv"
if [ ! -f "${DISCRIM_FT}" ]; then
    echo "Running Discrim-Eval Finetuned..."
    python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --adapter_path "${FINAL_MODEL}" \
        --csv_path "${DISCRIM_FT}" \
        --device "cuda" \
        --model_type "llama"
else
    echo "Discrim-Eval Finetuned already exists at ${DISCRIM_FT}, skipping."
fi

echo ">>> Step 5 DONE."
echo "Results saved to: ${OUT_DIR}"
