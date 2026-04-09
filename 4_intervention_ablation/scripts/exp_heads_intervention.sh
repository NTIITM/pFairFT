#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（基于敏感 Attention Heads 的负向/正向干预评估）
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp8/evaluate_intervention_heads.py"

# 模型根目录（与 exp2/exp_heads.sh 保持一致）
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验根目录
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2"
EXP8_DIR="/home/common1/hwluo/project/pFairFT/exp8"

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# Batch size
BATCH_SIZE=8

# 方向增强强度（正向干预用）
POSITIVE_STRENGTH=25000

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Running head-level interventions for model: $MODEL_NAME"
  echo "Model path: $MODEL_DIR"
  echo "=========================================="

  # 对应的 biased_samples 目录与 CSV（由 exp2/exp_sample.sh / evaluate_biased_sample.py 生成）
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

  if [ ! -f "$CSV_PATH" ]; then
    echo "WARNING: CSV not found for model $MODEL_NAME: $CSV_PATH"
    echo "Skip interventions for this model."
    echo ""
    continue
  fi

  # exp2 中种族敏感头分析的输出目录（来自 exp2/exp_heads.sh）
  SENSITIVE_HEADS_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"

  if [ ! -d "$SENSITIVE_HEADS_DIR" ]; then
    echo "WARNING: Sensitive heads directory not found for model $MODEL_NAME: $SENSITIVE_HEADS_DIR"
    echo "Skip interventions for this model."
    echo ""
    continue
  fi

  # 在 exp8 中为每个模型创建干预评估输出目录
  MODEL_OUTPUT_DIR="${EXP8_DIR}/intervention_heads_${MODEL_NAME}"
  mkdir -p "$MODEL_OUTPUT_DIR"

  # 基线：使用 CSV 中的 fact_p_yes（无需前向计算）
  echo "[Baseline] Using CSV fact_p_yes for model: $MODEL_NAME"
  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "$DATASET_JSON" \
    --sample_csv_path "$CSV_PATH" \
    --sample_size 100 \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto" \
    --intervention_type "baseline"

  # 负向干预：mean ablation
  echo "[Negative] Evaluating mean ablation for model: $MODEL_NAME"
  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "$DATASET_JSON" \
    --sample_csv_path "$CSV_PATH" \
    --sample_size 100 \
    --sensitive_heads_dir "$SENSITIVE_HEADS_DIR" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto" \
    --intervention_type "negative"

  # 正向干预：方向增强
  echo "[Positive] Evaluating directional amplification for model: $MODEL_NAME"
  python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "$DATASET_JSON" \
    --sample_csv_path "$CSV_PATH" \
    --sample_size 100 \
    --sensitive_heads_dir "$SENSITIVE_HEADS_DIR" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto" \
    --intervention_type "positive" \
    --positive_strength "$POSITIVE_STRENGTH"

  echo "Finished baseline + interventions for model: $MODEL_NAME"
  echo "Results saved to: $MODEL_OUTPUT_DIR"
  echo ""
done

echo "=========================================="
echo "All done. Head-level negative and positive interventions completed for all models."
echo "=========================================="

