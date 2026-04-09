#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp1/evaluate_bias.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 统一输出的 CSV 文件路径
CSV_PATH="per_sample_details_all_models.csv"

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Running analysis for model: $MODEL_NAME ($MODEL_DIR)"
  echo "=========================================="

  # 为每个模型运行两次（prompt 和 debiased_prompt）
  for PROMPT_TYPE in "prompt" "debiased_prompt"; do
    echo "Processing prompt type: $PROMPT_TYPE"
    
    python "$PY_SCRIPT" \
      --model_path "$MODEL_DIR" \
      --dataset_path "/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json" \
      --prompt_type "$PROMPT_TYPE" \
      --csv_path "$CSV_PATH"

    echo "Finished $PROMPT_TYPE for model: $MODEL_NAME"
  done

  echo "Finished model: $MODEL_NAME"
  echo ""
done

echo "=========================================="
echo "All done. Per-sample details CSV: $CSV_PATH"
echo "=========================================="
