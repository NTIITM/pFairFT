#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径：在 discrim-eval 配对数据上做「负向干预」评估
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp8/evaluate_intervention_discrim-eval.py"

# 模型根目录（与其他实验脚本保持一致）
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# discrim-eval 数据集路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json"

# 统一输出的 CSV 文件路径（聚合所有模型的干预结果）
CSV_PATH="per_sample_intervention_negative_all_models.csv"

# 仅使用原始 prompt（如需 debiased_prompt，可再加一层循环）
PROMPT_TYPE="prompt"

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
# for MODEL_DIR in "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Running NEGATIVE head-level intervention discrim-eval for model: $MODEL_NAME"
  echo "Model path: $MODEL_DIR"
  echo "=========================================="

  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_path "$DATASET_JSON" \
    --prompt_type "$PROMPT_TYPE" \
    --csv_path "$CSV_PATH" \
    --intervention_mode "negative"

  echo "Finished negative intervention discrim-eval for model: $MODEL_NAME"
  echo ""
done

echo "=========================================="
echo "All done. Per-sample negative-intervention details CSV: $CSV_PATH"
echo "=========================================="

