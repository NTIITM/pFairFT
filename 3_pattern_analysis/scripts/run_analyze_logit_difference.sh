#!/usr/bin/env bash
set -euo pipefail

# EXP12: 运行 Logit Difference 分析脚本（Llama-8B）

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp12/analyze_logit_difference.py"

# 模型路径（Llama-8B）
MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Meta-Llama-3-8B-Instruct"

# 数据集路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# 实验根目录（exp12 目录）
EXP12_DIR="/home/common1/hwluo/project/pFairFT/exp12"

# 输出目录
OUTPUT_DIR="${EXP12_DIR}/logit_difference_${MODEL_NAME}"

# 参数设置
MAX_SAMPLES=500
BATCH_SIZE=8

echo "=========================================="
echo "EXP12: Logit Difference Analysis"
echo "Model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行分析脚本
python "$PY_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --dataset_json_path "$DATASET_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto" \
    --balanced \
    --seed 42

echo ""
echo "=========================================="
echo "Analysis completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
