#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（干预评估）
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp2/evaluate_intervention.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 输出目录（exp2 目录）
OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp2"

# 统一输出的 CSV 文件路径（追加写入）
CSV_PATH="${OUTPUT_DIR}/intervention_results_heads_top100.csv"

# 如果之前已存在同名 CSV，先删除，重新生成
if [ -f "$CSV_PATH" ]; then
  echo "Removing existing CSV: $CSV_PATH"
  rm "$CSV_PATH"
fi

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# 干预的样本数量（与 sensitive_heads_*_top100 对齐）
SAMPLE_SIZE=100

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Running head intervention evaluation (top-${SAMPLE_SIZE}) for model: $MODEL_NAME"
  echo "Model path: $MODEL_DIR"
  echo "=========================================="

  # 该模型的 biased_samples CSV（exp_sample.sh 输出）
  BIASED_DIR="${OUTPUT_DIR}/biased_samples_${MODEL_NAME}"
  BIASED_CSV="${BIASED_DIR}/biased_samples_ranking.csv"

  if [ ! -f "$BIASED_CSV" ]; then
    echo "Warning: Biased samples CSV not found for $MODEL_NAME"
    echo "  Expected: $BIASED_CSV"
    echo "  Skip this model."
    echo ""
    continue
  fi

  # 使用 exp_heads.sh 产生的敏感头结果（top100）
  SENSITIVE_HEADS_DIR="${OUTPUT_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  SENSITIVE_HEADS_FILE="${SENSITIVE_HEADS_DIR}/selected_heads_elbow.json"
  EMBEDDINGS_FILE="${SENSITIVE_HEADS_DIR}/results.pkl"

  if [ ! -f "$SENSITIVE_HEADS_FILE" ] || [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "Warning: Sensitive heads files not found for $MODEL_NAME"
    echo "  Expected: $SENSITIVE_HEADS_FILE"
    echo "  Expected: $EMBEDDINGS_FILE"
    echo "  Please run exp_heads.sh first to generate these files."
    echo ""
    continue
  fi

  # 为每个模型创建独立的输出目录
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/intervention_results_${MODEL_NAME}_top100"

  # 定义干预模式列表（与 evaluate_intervention.py 一致）
  INTERVENTION_MODES=("mean_replacement" "debias_projection" "zero_value")
  INTERVENTION_STRENGTHS=(1.0 0.5 2.0)  # 对于 debias_projection 使用不同的强度

  # 在同一批 top-100 样本上运行不同干预模式
  for INTERVENTION_MODE in "${INTERVENTION_MODES[@]}"; do
    echo "Running intervention mode: $INTERVENTION_MODE on top-${SAMPLE_SIZE} biased samples"

    if [ "$INTERVENTION_MODE" == "debias_projection" ]; then
      # debias_projection：尝试不同的强度
      for STRENGTH in "${INTERVENTION_STRENGTHS[@]}"; do
        echo "  With strength: $STRENGTH"
        python "$PY_SCRIPT" \
          --model_path "$MODEL_DIR" \
          --dataset_json_path "$DATASET_JSON" \
          --output_dir "$MODEL_OUTPUT_DIR" \
          --batch_size 8 \
          --balanced \
          --seed 42 \
          --device "cuda" \
          --model_type "auto" \
          --sensitive_heads_path "$SENSITIVE_HEADS_FILE" \
          --embeddings_path "$EMBEDDINGS_FILE" \
          --intervention_mode "$INTERVENTION_MODE" \
          --intervention_strength "$STRENGTH" \
          --sample_csv_path "$BIASED_CSV" \
          --sample_size "$SAMPLE_SIZE" \
          --csv_path "$CSV_PATH"
      done
    else
      # 其他模式只运行一次（强度固定为 1.0）
      python "$PY_SCRIPT" \
        --model_path "$MODEL_DIR" \
        --dataset_json_path "$DATASET_JSON" \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --batch_size 8 \
        --balanced \
        --seed 42 \
        --device "cuda" \
        --model_type "auto" \
        --sensitive_heads_path "$SENSITIVE_HEADS_FILE" \
        --embeddings_path "$EMBEDDINGS_FILE" \
        --intervention_mode "$INTERVENTION_MODE" \
        --intervention_strength 1.0 \
        --sample_csv_path "$BIASED_CSV" \
        --sample_size "$SAMPLE_SIZE" \
        --csv_path "$CSV_PATH"
    fi

    echo "Finished intervention mode: $INTERVENTION_MODE for model: $MODEL_NAME"
  done

  echo "Finished all interventions for model: $MODEL_NAME"
  echo ""
done

echo "=========================================="
echo "All done. Aggregated CSV: $CSV_PATH"
echo "=========================================="

