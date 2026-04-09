#!/usr/bin/env bash
set -euo pipefail

# 统一评估 exp5 中所有 LoRA 微调模型在 MMLU 上的表现。
#
# 假设：
# - exp5 微调输出目录：exp5/lora_${MODEL_NAME}_top100/final_model
# - 本脚本所在目录：exp14
#
# 该脚本会：
# 1. 遍历 exp5 目录下所有以 lora_*_top100 命名的子目录
# 2. 对其中的 final_model 调用 evaluate_mmlu.py
# 3. 将结果保存到 exp14/mmlu_results/ 对应的 JSON 文件中

EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP5_DIR="${EXP_ROOT}/exp5"
EXP14_DIR="${EXP_ROOT}/exp14"
PY_SCRIPT="${EXP14_DIR}/evaluate_mmlu.py"

OUTPUT_ROOT="${EXP14_DIR}/mmlu_results"
mkdir -p "${OUTPUT_ROOT}"

DEVICE="${DEVICE:-cuda}"          # 可通过环境变量覆盖：DEVICE=cpu bash run_mmlu_exp5.sh
SPLIT="${SPLIT:-validation}"      # 可通过环境变量覆盖：SPLIT=test bash run_mmlu_exp5.sh
MAX_SAMPLES="${MAX_SAMPLES:--1}"  # 可通过环境变量覆盖

echo "============================================================"
echo "Running MMLU evaluation for all exp5 LoRA fine-tuned models"
echo "EXP5_DIR = ${EXP5_DIR}"
echo "Output   = ${OUTPUT_ROOT}"
echo "Device   = ${DEVICE}"
echo "Split    = ${SPLIT}"
echo "Max samples = ${MAX_SAMPLES}"
echo "============================================================"

shopt -s nullglob
for MODEL_DIR in "${EXP5_DIR}"/lora_*_top100; do
  if [ ! -d "${MODEL_DIR}" ]; then
    continue
  fi

  MODEL_NAME="$(basename "${MODEL_DIR}")"
  FINAL_MODEL_DIR="${MODEL_DIR}/final_model"

  if [ ! -d "${FINAL_MODEL_DIR}" ]; then
    echo "[Skip] ${MODEL_NAME}: final_model not found."
    continue
  fi

  OUTPUT_JSON="${OUTPUT_ROOT}/mmlu_${MODEL_NAME}.json"

  echo "------------------------------------------------------------"
  echo "Evaluating model: ${MODEL_NAME}"
  echo "  Model path : ${FINAL_MODEL_DIR}"
  echo "  Output JSON: ${OUTPUT_JSON}"

  python "${PY_SCRIPT}" \
    --model_path "${FINAL_MODEL_DIR}" \
    --device "${DEVICE}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --output_json "${OUTPUT_JSON}"

  echo "Done: ${MODEL_NAME}"
done

echo "============================================================"
echo "All exp5 LoRA models have been evaluated on MMLU (where available)."
echo "Results are saved under: ${OUTPUT_ROOT}"
echo "============================================================"

