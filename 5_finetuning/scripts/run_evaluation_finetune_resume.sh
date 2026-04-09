#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（评估脚本）
EVAL_SCRIPT="/home/common1/hwluo/project/pFairFT/exp5/evaluate_finetune_resume.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验根目录（exp5 目录）
EXP5_DIR="/home/common1/hwluo/project/pFairFT/exp5"
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
# 注意：这里我们遍历的是原始模型路径，以便找到对应的LoRA微调结果
for BASE_MODEL_PATH in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$BASE_MODEL_PATH" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$BASE_MODEL_PATH")"
  LORA_MODEL_DIR="${EXP5_DIR}/lora_${MODEL_NAME}_top100"
  LORA_ADAPTER_PATH="${LORA_MODEL_DIR}/final_model"

  # 检查对应的 LoRA 最终模型目录是否存在（由新的 finetune_model.py 生成）
  if [ ! -d "${LORA_ADAPTER_PATH}" ]; then
    echo "Warning: LoRA final_model directory not found for ${MODEL_NAME} at ${LORA_ADAPTER_PATH}. Skipping."
    continue
  fi

  echo "=========================================="
  echo "Evaluating fine-tuned model: ${MODEL_NAME}"
  echo "Base Model Path: ${BASE_MODEL_PATH}"
  echo "LoRA Adapter Path: ${LORA_ADAPTER_PATH}"
  echo "=========================================="

  # 对应的 biased_samples 目录与 CSV（如果存在则使用 top-100 偏见样本进行评估）
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

  # 为每个模型创建独立的评估结果输出目录
  EVAL_OUTPUT_DIR="${EXP5_DIR}/eval_finetune_resume_results_${MODEL_NAME}"
  mkdir -p "$EVAL_OUTPUT_DIR"

  if [ -f "$CSV_PATH" ]; then
    echo "Using CSV-driven sampling for evaluation from: $CSV_PATH"
    python "$EVAL_SCRIPT" \
      --base_model_path "$BASE_MODEL_PATH" \
      --lora_model_path "$LORA_ADAPTER_PATH" \
      --dataset_json_path "$DATASET_JSON" \
      --output_dir "$EVAL_OUTPUT_DIR" \
      --max_samples 500 \
      --batch_size 8 \
      --balanced \
      --seed 42 \
      --sample_csv_path "$CSV_PATH" \
      --sample_size 100
  else
    python "$EVAL_SCRIPT" \
      --base_model_path "$BASE_MODEL_PATH" \
      --lora_model_path "$LORA_ADAPTER_PATH" \
      --dataset_json_path "$DATASET_JSON" \
      --output_dir "$EVAL_OUTPUT_DIR" \
      --max_samples 500 \
      --batch_size 8 \
      --balanced \
      --seed 42
  fi

  echo "Finished evaluation for model: ${MODEL_NAME}"
  echo "Results saved to: ${EVAL_OUTPUT_DIR}"
  echo ""
done

echo "=========================================="
echo "All done. Evaluation completed for all fine-tuned models."
echo "=========================================="
