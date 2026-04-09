#!/bin/bash
# 专门运行 Llama-3-8B-Instruct 模型，进行 exp10 的 discrim-eval 偏见程度干预实验。
# 敏感头干预（negative）在 GPU 2 上运行，随机头干预（negative_random）在 GPU 5 上运行。
# 每个模型运行 6 个 step（含 0）：head_count = 0, step, 2*step, ..., 5*step。
# max_head_count = 5*step，以保证恰好 6 个点。

set -e

# 定义路径
MODEL_DIR="/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Meta-Llama-3-8B-Instruct"
MODEL_TYPE="llama"

EXP2_OLD_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"
EXP10_DIR="/home/common1/hwluo/project/pFairFT/exp10"

HEADS_DIR="${EXP2_OLD_DIR}/sensitive_heads_${MODEL_NAME}_top100"
OUTPUT_DIR="${EXP10_DIR}/intervention_results_${MODEL_NAME}_discrim_eval"

# Llama-3-8B-Instruct 的 step 设置
STEP=9
MAX_HEAD=$((5 * STEP)) # 6 个 step 含 0

# 检查 results.pkl 是否存在
if [ ! -f "${HEADS_DIR}/results.pkl" ]; then
  echo "Error: results.pkl not found in $HEADS_DIR (run exp2_old/exp_heads.sh first or use exp2_old output)."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "=========================================="
echo "Testing Llama-3-8B-Instruct model (exp10 discrim-eval)"
echo "  model_path=$MODEL_DIR"
echo "  heads_dir=$HEADS_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  step=$STEP, max_head_count=$MAX_HEAD (6 points: 0,$STEP,$((2*STEP)),$((3*STEP)),$((4*STEP)),$MAX_HEAD)"
echo "=========================================="

# 敏感头干预 (negative) - 在 GPU 2 上运行
echo ""
echo "========== Running Sensitive heads (negative intervention) on GPU 2 =========="
CUDA_VISIBLE_DEVICES=2 python "${EXP10_DIR}/evaluate_intervention_discrim_eval_head_count.py" \
  --model_path "$MODEL_DIR" \
  --model_type "$MODEL_TYPE" \
  --device cuda \
  --output_dir "$OUTPUT_DIR" \
  --embeddings_path "${HEADS_DIR}/results.pkl" \
  --intervention_mode negative \
  --max_head_count "$MAX_HEAD" \
  --step "$STEP" \
  --results_csv_name "results_sensitive_heads.csv" \
  --seed 42 & # Run in background
PID_SENSITIVE=$!

# 随机头干预 (negative_random) - 在 GPU 5 上运行
echo ""
echo "========== Running Random heads (negative intervention) on GPU 5 =========="
CUDA_VISIBLE_DEVICES=5 python "${EXP10_DIR}/evaluate_intervention_discrim_eval_head_count.py" \
  --model_path "$MODEL_DIR" \
  --model_type "$MODEL_TYPE" \
  --device cuda \
  --output_dir "$OUTPUT_DIR" \
  --embeddings_path "${HEADS_DIR}/results.pkl" \
  --intervention_mode negative_random \
  --max_head_count "$MAX_HEAD" \
  --step "$STEP" \
  --results_csv_name "results_random_heads.csv" \
  --seed 42 & # Run in background
PID_RANDOM=$!

# Wait for both processes to complete
wait $PID_SENSITIVE
wait $PID_RANDOM

echo ""
echo "Done for Llama-3-8B-Instruct. CSVs in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || true
echo ""
echo "=========================================="
echo "All Llama-3-8B-Instruct evaluation done."
echo "=========================================="
