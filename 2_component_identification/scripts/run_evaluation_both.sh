#!/bin/bash
# 同时运行敏感头干预与随机头干预；每个模型运行 6 个 step（含 0）：head_count = 0, step, 2*step, ..., 5*step。
# max_head_count = 5*step，以保证恰好 6 个点。
# 结果保存在各模型对应目录下的不同 CSV：
#   - intervention_results_by_head_count.csv         (敏感头)
#   - intervention_results_by_head_count_random.csv  (随机头)
# 模型路径来自 /mnt/nfs/huggingface/LLM-Research/ 与 /mnt/nfs/huggingface/Qwen/
# 敏感头目录来自 exp2_old：sensitive_heads_${MODEL_NAME}_top100

set -e

LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"
EXP2_OLD_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"
EXP9_DIR="/home/common1/hwluo/project/pFairFT/exp9"

# 模型路径 -> step 映射（与手写表一致）
# Qwen 1.7b=3, Qwen 4b=5, Qwen 8b=5; Llama 1b=2, Llama 3b=7, Llama 8b=9
get_step() {
  case "$1" in
    Qwen3-1.7B)                echo 3 ;;
    Qwen3-4B|Qwen3-8B)         echo 5 ;;
    Llama-3.2-1B-Instruct)      echo 2 ;;
    Llama-3.2-3B-Instruct)      echo 7 ;;
    Meta-Llama-3-8B-Instruct)   echo 9 ;;
    *)                         echo 5 ;;
  esac
}

get_model_type() {
  case "$1" in
    Qwen3-*)   echo "qwen" ;;
    *)         echo "auto" ;;
  esac
}

# 遍历 Qwen 与 Llama 模型
for MODEL_DIR in "$QWEN_DIR"/Qwen3-1.7B "$QWEN_DIR"/Qwen3-4B "$QWEN_DIR"/Qwen3-8B \
                 "$LLM_RESEARCH_DIR"/Llama-3.2-1B-Instruct \
                 "$LLM_RESEARCH_DIR"/Llama-3.2-3B-Instruct \
                 "$LLM_RESEARCH_DIR"/Meta-Llama-3-8B-Instruct; do
  if [ ! -d "$MODEL_DIR" ]; then
    echo "Skip (not a directory): $MODEL_DIR"
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  STEP=$(get_step "$MODEL_NAME")
  MODEL_TYPE=$(get_model_type "$MODEL_NAME")
  # 6 个 step 含 0：0, step, 2*step, 3*step, 4*step, 5*step -> max_head_count = 5*step
  MAX_HEAD=$((5 * STEP))

  HEADS_DIR="${EXP2_OLD_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  SAMPLE_CSV="${EXP2_OLD_DIR}/biased_samples_${MODEL_NAME}/biased_samples_ranking.csv"
  OUTPUT_DIR="${EXP9_DIR}/intervention_results_${MODEL_NAME}_top100"

  if [ ! -f "${HEADS_DIR}/results.pkl" ]; then
    echo "Skip $MODEL_NAME: results.pkl not found in $HEADS_DIR (run exp2_old/exp_heads.sh first or use exp2_old output)."
    continue
  fi
  if [ ! -f "$SAMPLE_CSV" ]; then
    echo "Skip $MODEL_NAME: sample CSV not found: $SAMPLE_CSV"
    continue
  fi

  mkdir -p "$OUTPUT_DIR"
  echo "=========================================="
  echo "Model: $MODEL_NAME  step=$STEP  max_head_count=$MAX_HEAD (6 points: 0,$STEP,$((2*STEP)),$((3*STEP)),$((4*STEP)),$MAX_HEAD)  model_type=$MODEL_TYPE"
  echo "  model_path=$MODEL_DIR"
  echo "  heads_dir=$HEADS_DIR"
  echo "  output_dir=$OUTPUT_DIR"
  echo "=========================================="

  echo ""
  echo "========== 1/2 Sensitive heads (negative intervention) =========="
  python "${EXP9_DIR}/evaluate_intervention_by_head_count.py" \
    --model_path "$MODEL_DIR" \
    --model_type "$MODEL_TYPE" \
    --sample_csv_path "$SAMPLE_CSV" \
    --sample_size 100 \
    --sensitive_heads_dir "$HEADS_DIR" \
    --max_head_count "$MAX_HEAD" \
    --step "$STEP" \
    --intervention_type negative \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --device cuda

  echo ""
  echo "========== 2/2 Random heads (negative intervention) =========="
  python "${EXP9_DIR}/evaluate_intervention_by_head_count_random.py" \
    --model_path "$MODEL_DIR" \
    --model_type "$MODEL_TYPE" \
    --sample_csv_path "$SAMPLE_CSV" \
    --sample_size 100 \
    --sensitive_heads_dir "$HEADS_DIR" \
    --max_head_count "$MAX_HEAD" \
    --step "$STEP" \
    --output_dir "$OUTPUT_DIR" \
    --results_csv_name "intervention_results_by_head_count_random.csv" \
    --batch_size 8 \
    --device cuda

  echo ""
  echo "Done for $MODEL_NAME. CSVs in $OUTPUT_DIR:"
  ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || true
  echo ""
done

echo "=========================================="
echo "All models done."
echo "=========================================="
