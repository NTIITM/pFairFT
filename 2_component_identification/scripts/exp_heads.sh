#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（Attention Heads 种族敏感度分析）
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp2/analyze_race_sensitive_heads.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验根目录（exp2 目录）
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2"

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# Batch size
BATCH_SIZE=8

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Analyzing race-sensitive heads for model: $MODEL_NAME"
  echo "Model path: $MODEL_DIR"
  echo "=========================================="

  # 对应的 biased_samples 目录与 CSV（需要事先由 exp_sample.sh / evaluate_biased_sample.py 生成）
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

  if [ ! -f "$CSV_PATH" ]; then
    echo "WARNING: CSV not found for model $MODEL_NAME: $CSV_PATH"
    echo "Skip heads analysis for this model."
    echo ""
    continue
  fi

  # 为每个模型创建 heads 分析输出目录
  MODEL_OUTPUT_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  mkdir -p "$MODEL_OUTPUT_DIR"

  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "$DATASET_JSON" \
    --sample_csv_path "$CSV_PATH" \
    --sample_size 100 \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto"

  echo "Finished race-sensitive heads analysis for model: $MODEL_NAME"
  echo "Results saved to: $MODEL_OUTPUT_DIR"
  echo ""
done

echo "=========================================="
echo "All done. Race-sensitive heads analysis completed for all models."
echo "=========================================="

