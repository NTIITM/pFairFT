#!/usr/bin/env bash
set -euo pipefail

# 比较 baseline / exp4 / exp5 三者在 MMLU 上的表现。
# 假设：
#   - baseline: /mnt/nfs/huggingface/LLM-Research/${MODEL_NAME} 或 /mnt/nfs/huggingface/Qwen/${MODEL_NAME}
#   - exp4: precision_fairness_${MODEL_NAME}_top100/final_model
#   - exp5: lora_${MODEL_NAME}_top100/final_model
#
# 输出（位于 exp14/mmlu_results/）：
#   - mmlu_baseline_${MODEL_NAME}_top100.json
#   - mmlu_precision_fairness_${MODEL_NAME}_top100.json
#   - mmlu_lora_${MODEL_NAME}_top100.json

EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP4_DIR="${EXP_ROOT}/exp4"
EXP5_DIR="${EXP_ROOT}/exp5"
EXP14_DIR="${EXP_ROOT}/exp14"
PY_SCRIPT="${EXP14_DIR}/evaluate_mmlu.py"

# 与 exp5/exp_finetune_all.sh 保持一致的基座模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

OUTPUT_ROOT="${EXP14_DIR}/mmlu_results"
mkdir -p "${OUTPUT_ROOT}"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-validation}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

echo "============================================================"
echo "Running MMLU comparison: baseline vs exp4 (precision_fairness) vs exp5 (LoRA)"
echo "EXP4_DIR = ${EXP4_DIR}"
echo "EXP5_DIR = ${EXP5_DIR}"
echo "LLM_RESEARCH_DIR = ${LLM_RESEARCH_DIR}"
echo "QWEN_DIR         = ${QWEN_DIR}"
echo "Output   = ${OUTPUT_ROOT}"
echo "Device   = ${DEVICE}"
echo "Split    = ${SPLIT}"
echo "Max samples = ${MAX_SAMPLES}"
echo "============================================================"

shopt -s nullglob

# 遍历 exp4 中的 precision_fairness_*_top100 目录
for PF_DIR in "${EXP4_DIR}"/precision_fairness_*_top100; do
  if [ ! -d "${PF_DIR}" ]; then
    continue
  fi

  PF_NAME="$(basename "${PF_DIR}")"  # precision_fairness_Llama-3.2-1B-Instruct_top100
  FINAL_PF_DIR="${PF_DIR}/final_model"

  if [ ! -d "${FINAL_PF_DIR}" ]; then
    echo "[Skip exp4] ${PF_NAME}: final_model not found."
    continue
  fi

  # 推导后缀和基础模型名：
  #   PF_NAME      = precision_fairness_Llama-3.2-1B-Instruct_top100
  #   MODEL_SUFFIX = Llama-3.2-1B-Instruct_top100
  #   MODEL_BASE   = Llama-3.2-1B-Instruct
  MODEL_SUFFIX="${PF_NAME#precision_fairness_}"         # Llama-3.2-1B-Instruct_top100
  MODEL_BASE="${MODEL_SUFFIX%_top100}"                  # Llama-3.2-1B-Instruct

  # 对应的 exp5 LoRA 目录名：precision_fairness_ 前缀替换为 lora_
  LORA_NAME="lora_${MODEL_SUFFIX}"                      # lora_Llama-3.2-1B-Instruct_top100
  LORA_DIR="${EXP5_DIR}/${LORA_NAME}"
  FINAL_LORA_DIR="${LORA_DIR}/final_model"

  # 推导 baseline 基座模型路径：根据 MODEL_BASE 选择 LLM-Research 或 Qwen 目录
  BASE_MODEL_DIR=""
  if [[ "${MODEL_BASE}" == Qwen3-* ]]; then
    BASE_MODEL_DIR="${QWEN_DIR}/${MODEL_BASE}"
  else
    BASE_MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_BASE}"
  fi

  echo "------------------------------------------------------------"
  echo "Model family: ${MODEL_SUFFIX}"
  echo "  baseline model dir         : ${BASE_MODEL_DIR}"
  echo "  exp4 precision_fairness dir: ${FINAL_PF_DIR}"
  echo "  exp5 LoRA dir              : ${FINAL_LORA_DIR}"

  # 评估 baseline 模型（如果存在）
  if [ -d "${BASE_MODEL_DIR}" ]; then
    OUTPUT_BASE_JSON="${OUTPUT_ROOT}/mmlu_baseline_${MODEL_SUFFIX}.json"
    if [ -f "${OUTPUT_BASE_JSON}" ]; then
      echo "  [baseline] Result already exists at ${OUTPUT_BASE_JSON}, skip."
    else
      echo "  [baseline] Evaluating base model -> ${OUTPUT_BASE_JSON}"
      python "${PY_SCRIPT}" \
        --model_path "${BASE_MODEL_DIR}" \
        --device "${DEVICE}" \
        --split "${SPLIT}" \
        --max_samples "${MAX_SAMPLES}" \
        --output_json "${OUTPUT_BASE_JSON}"
    fi
  else
    echo "  [baseline] Base model not found at ${BASE_MODEL_DIR}, skip."
  fi

  # 评估 exp4 模型
  OUTPUT_PF_JSON="${OUTPUT_ROOT}/mmlu_${PF_NAME}.json"
  if [ -f "${OUTPUT_PF_JSON}" ]; then
    echo "  [exp4] Result already exists at ${OUTPUT_PF_JSON}, skip."
  else
    echo "  [exp4] Evaluating precision_fairness model -> ${OUTPUT_PF_JSON}"
    python "${PY_SCRIPT}" \
      --model_path "${FINAL_PF_DIR}" \
      --device "${DEVICE}" \
      --split "${SPLIT}" \
      --max_samples "${MAX_SAMPLES}" \
      --output_json "${OUTPUT_PF_JSON}"
  fi

  # 如果存在对应的 exp5 模型，则一并评估
  if [ -d "${FINAL_LORA_DIR}" ]; then
    OUTPUT_LORA_JSON="${OUTPUT_ROOT}/mmlu_${LORA_NAME}.json"
    if [ -f "${OUTPUT_LORA_JSON}" ]; then
      echo "  [exp5] Result already exists at ${OUTPUT_LORA_JSON}, skip."
    else
      echo "  [exp5] Evaluating LoRA model -> ${OUTPUT_LORA_JSON}"
      python "${PY_SCRIPT}" \
        --model_path "${FINAL_LORA_DIR}" \
        --device "${DEVICE}" \
        --split "${SPLIT}" \
        --max_samples "${MAX_SAMPLES}" \
        --output_json "${OUTPUT_LORA_JSON}"
    fi
  else
    echo "  [exp5] LoRA model not found for suffix ${MODEL_SUFFIX}, skip."
  fi
done

echo "============================================================"
echo "MMLU comparison for baseline vs exp4 vs exp5 completed."
echo "Results are saved under: ${OUTPUT_ROOT}"
echo "============================================================"

