#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp2/evaluate_biased_sample.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 输出目录（exp2 目录）
OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp2"

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Evaluating bias level for model: $MODEL_NAME ($MODEL_DIR)"
  echo "=========================================="

  # 为每个模型创建独立的输出目录（区分名字保存）
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/biased_samples_${MODEL_NAME}"
  mkdir -p "$MODEL_OUTPUT_DIR"

  # 输出 CSV 文件路径
  CSV_OUTPUT="${MODEL_OUTPUT_DIR}/biased_samples_ranking.csv"

  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json" \
    --output_csv_path "$CSV_OUTPUT" \
    --device "cuda" \
    --model_type "auto"

  echo "Finished evaluating bias level for model: $MODEL_NAME"
  echo "Results saved to: $CSV_OUTPUT"
  echo ""
done

echo "=========================================="
echo "All done. Bias level evaluation completed for all models."
echo "=========================================="
