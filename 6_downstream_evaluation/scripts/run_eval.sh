#!/usr/bin/env bash
set -euo pipefail

# Path configuration
EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP4_DIR="${EXP_ROOT}/exp4"
EXP5_DIR="${EXP_ROOT}/exp5"
EXP16_DIR="${EXP_ROOT}/exp16"
PY_SCRIPT="${EXP16_DIR}/evaluate_models.py"

# Base model directories
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# Output files
CSV_EXP4="${EXP16_DIR}/per_sample_details_exp4.csv"
CSV_EXP5="${EXP16_DIR}/per_sample_details_exp5.csv"

# Remove existing CSVs to start fresh if needed
rm -f "${CSV_EXP4}" "${CSV_EXP5}"

DEVICE="${DEVICE:-cuda}"

echo "============================================================"
echo "Running exp16 evaluation: exp4 (full) vs exp5 (LoRA)"
echo "============================================================"

shopt -s nullglob

# Iterate through exp4 precision_fairness directories
for PF_DIR in "${EXP4_DIR}"/precision_fairness_*_top100; do
  if [ ! -d "${PF_DIR}" ]; then continue; fi

  PF_NAME="$(basename "${PF_DIR}")"
  FINAL_PF_DIR="${PF_DIR}/final_model"

  if [ ! -d "${FINAL_PF_DIR}" ]; then
    echo "[Skip exp4] ${PF_NAME}: final_model not found."
    continue
  fi

  # Deriving model names
  MODEL_SUFFIX="${PF_NAME#precision_fairness_}"
  MODEL_BASE="${MODEL_SUFFIX%_top100}"

  # Base model path
  BASE_MODEL_DIR=""
  if [[ "${MODEL_BASE}" == Qwen3-* ]]; then
    BASE_MODEL_DIR="${QWEN_DIR}/${MODEL_BASE}"
  else
    BASE_MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_BASE}"
  fi

  # 1. Evaluate Exp4 (Full Finetune)
  echo "Evaluating exp4: ${PF_NAME}"
  python "${PY_SCRIPT}" \
    --base_model_path "${FINAL_PF_DIR}" \
    --device "${DEVICE}" \
    --csv_path "${CSV_EXP4}"

  # 2. Evaluate Exp5 (LoRA)
  LORA_NAME="lora_${MODEL_SUFFIX}"
  FINAL_LORA_DIR="${EXP5_DIR}/${LORA_NAME}/final_model"
  if [ -d "${FINAL_LORA_DIR}" ]; then
    echo "Evaluating exp5: ${LORA_NAME}"
    python "${PY_SCRIPT}" \
      --base_model_path "${BASE_MODEL_DIR}" \
      --adapter_path "${FINAL_LORA_DIR}" \
      --device "${DEVICE}" \
      --csv_path "${CSV_EXP5}"
  else
    echo "[Skip exp5] LoRA model not found for ${MODEL_SUFFIX}"
  fi

done

echo "============================================================"
echo "Evaluation complete. Results saved in:"
echo " - ${CSV_EXP4}"
echo " - ${CSV_EXP5}"
echo "============================================================"
