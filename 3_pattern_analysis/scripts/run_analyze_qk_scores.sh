#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp11/analyze_qk_scores.py"

# 模型路径（以LLaMA-8B为例）
MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct"

# exp2的输出目录
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"

# 数据集路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# 输出目录
OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp11/qk_scores_output"

# 敏感头文件路径
SENSITIVE_HEADS_PATH="${EXP2_DIR}/sensitive_heads_Llama-3.2-3B-Instruct_top100/selected_heads_elbow.json"

# 样本CSV路径
SAMPLE_CSV_PATH="${EXP2_DIR}/biased_samples_Llama-3.2-3B-Instruct/biased_samples_ranking.csv"

echo "=========================================="
echo "Analyzing QK scores for sensitive heads"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Sensitive heads: $SENSITIVE_HEADS_PATH"
echo "Sample CSV: $SAMPLE_CSV_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$SENSITIVE_HEADS_PATH" ]; then
    echo "ERROR: Sensitive heads file not found: $SENSITIVE_HEADS_PATH"
    echo "Please run exp2/exp_heads.sh first to generate sensitive heads."
    exit 1
fi

if [ ! -f "$SAMPLE_CSV_PATH" ]; then
    echo "ERROR: Sample CSV file not found: $SAMPLE_CSV_PATH"
    echo "Please run exp2/exp_sample.sh first to generate biased samples."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行Python脚本
python "$PY_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --sensitive_heads_path "$SENSITIVE_HEADS_PATH" \
    --sample_csv_path "$SAMPLE_CSV_PATH" \
    --dataset_json_path "$DATASET_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --model_type "auto"

echo "=========================================="
echo "Done! Results saved to: $OUTPUT_DIR"
echo "=========================================="
