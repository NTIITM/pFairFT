#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp3/evaluate_general.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# exp2 和 exp3 目录
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2"
EXP3_DIR="/home/common1/hwluo/project/pFairFT/exp3"

# 统一输出的 CSV 文件路径（追加写入）
CSV_PATH="${EXP3_DIR}/general_results_all_models.csv"

# 如果之前已存在同名 CSV，先删除，重新生成
if [ -f "$CSV_PATH" ]; then
  echo "Removing existing CSV: $CSV_PATH"
  rm "$CSV_PATH"
fi

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  echo "=========================================="
  echo "Running general evaluation for model: $MODEL_NAME ($MODEL_DIR)"
  echo "=========================================="

  # 查找该模型的敏感头分析结果（从 exp2）
  SENSITIVE_HEADS_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}"
  SENSITIVE_HEADS_FILE="${SENSITIVE_HEADS_DIR}/selected_heads_elbow.json"
  EMBEDDINGS_FILE="${SENSITIVE_HEADS_DIR}/results.pkl"

  # 检查文件是否存在
  if [ ! -f "$SENSITIVE_HEADS_FILE" ] || [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "Warning: Sensitive heads files not found for $MODEL_NAME"
    echo "  Expected: $SENSITIVE_HEADS_FILE"
    echo "  Expected: $EMBEDDINGS_FILE"
    echo "  Please run exp_2_1.sh first to generate these files."
    echo ""
    continue
  fi

  # 为每个模型创建独立的输出目录
  MODEL_OUTPUT_DIR="${EXP3_DIR}/general_results_${MODEL_NAME}"

  # 定义干预模式列表
  INTERVENTION_MODES=("mean_replacement" "debias_projection" "zero_value")
  INTERVENTION_STRENGTHS=(1.0 0.5 2.0)  # 对于 debias_projection 使用不同的强度

  # 定义 prompt 类型列表
  PROMPT_TYPES=("prompt" "debiased_prompt")

  # 对每种 prompt 类型进行评估
  for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    echo "Processing prompt type: $PROMPT_TYPE"
    
    # 首先运行 baseline（无干预）
    echo "  Running baseline (no intervention)..."
    python "$PY_SCRIPT" \
      --model_path "$MODEL_DIR" \
      --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json" \
      --output_dir "$MODEL_OUTPUT_DIR" \
      --prompt_type "$PROMPT_TYPE" \
      --device "cuda" \
      --model_type "auto" \
      --baseline \
      --csv_path "$CSV_PATH"

    # 然后运行不同的干预模式
    for INTERVENTION_MODE in "${INTERVENTION_MODES[@]}"; do
      echo "  Running intervention mode: $INTERVENTION_MODE"
      
      if [ "$INTERVENTION_MODE" == "debias_projection" ]; then
        # 对于 debias_projection，尝试不同的强度
        for STRENGTH in "${INTERVENTION_STRENGTHS[@]}"; do
          echo "    With strength: $STRENGTH"
          python "$PY_SCRIPT" \
            --model_path "$MODEL_DIR" \
            --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json" \
            --output_dir "$MODEL_OUTPUT_DIR" \
            --prompt_type "$PROMPT_TYPE" \
            --device "cuda" \
            --model_type "auto" \
            --sensitive_heads_path "$SENSITIVE_HEADS_FILE" \
            --embeddings_path "$EMBEDDINGS_FILE" \
            --intervention_mode "$INTERVENTION_MODE" \
            --intervention_strength "$STRENGTH" \
            --csv_path "$CSV_PATH"
        done
      else
        # 对于其他模式，只运行一次
        python "$PY_SCRIPT" \
          --model_path "$MODEL_DIR" \
          --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json" \
          --output_dir "$MODEL_OUTPUT_DIR" \
          --prompt_type "$PROMPT_TYPE" \
          --device "cuda" \
          --model_type "auto" \
          --sensitive_heads_path "$SENSITIVE_HEADS_FILE" \
          --embeddings_path "$EMBEDDINGS_FILE" \
          --intervention_mode "$INTERVENTION_MODE" \
          --intervention_strength 1.0 \
          --csv_path "$CSV_PATH"
      fi
      
      echo "  Finished intervention mode: $INTERVENTION_MODE"
    done
    
    echo "Finished prompt type: $PROMPT_TYPE"
  done

  echo "Finished model: $MODEL_NAME"
  echo ""
done

echo "=========================================="
echo "All done. Aggregated CSV: $CSV_PATH"
echo "=========================================="
