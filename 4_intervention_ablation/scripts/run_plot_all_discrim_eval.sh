#!/usr/bin/env bash
# 遍历 exp10 的所有输出目录，一键调用 plot_discrim_eval_head_count.py 画图。
# 为每个模型生成两种图：
# 1. overall_mean (PNG): 总体偏见随头数量变化的折线图
# 2. by_question_id (PDF): 按问题 ID 排列的详细对比图

set -euo pipefail

EXP10_DIR="/home/common1/hwluo/project/pFairFT/exp10"
PLOT_PY="${EXP10_DIR}/plot_discrim_eval_head_count.py"

# 获取模型的 step 大小（保持与 evaluation 脚本一致）
get_step() {
  case "$1" in
    Qwen3-1.7B)                 echo 3 ;;
    Qwen3-4B|Qwen3-8B)          echo 5 ;;
    Llama-3.2-1B-Instruct)      echo 2 ;;
    Llama-3.2-3B-Instruct)      echo 7 ;;
    Meta-Llama-3-8B-Instruct)   echo 9 ;;
    *)                          echo 5 ;;
  esac
}

echo "Starting plotting for all models in ${EXP10_DIR}..."

# 遍历所有干预结果目录
for RES_DIR in "${EXP10_DIR}"/intervention_results_*_discrim_eval; do
  if [ ! -d "${RES_DIR}" ]; then
    continue
  fi

  # 从目录名提取模型名称
  # 目录名格式: intervention_results_{MODEL_NAME}_discrim_eval
  DIR_NAME=$(basename "${RES_DIR}")
  MODEL_NAME=${DIR_NAME#intervention_results_}
  MODEL_NAME=${MODEL_NAME%_discrim_eval}

  # 检查必要的 CSV 是否存在
  SENSITIVE_CSV="${RES_DIR}/results_sensitive_heads.csv"
  RANDOM_CSV="${RES_DIR}/results_random_heads.csv"

  if [ ! -f "${SENSITIVE_CSV}" ] || [ ! -f "${RANDOM_CSV}" ]; then
    echo "[SKIP] ${MODEL_NAME}: Missing CSV files in ${RES_DIR}"
    continue
  fi

  # 根据 step 计算画图用的 head_counts (0, step, 2*step, ..., 5*step)
  STEP=$(get_step "${MODEL_NAME}")
  TARGET_HC=""
  for i in {0..5}; do
    TARGET_HC="${TARGET_HC} $((i * STEP))"
  done

  echo "=========================================="
  echo "Plotting for Model: ${MODEL_NAME}"
  echo "Target Head Counts: ${TARGET_HC}"
  echo "=========================================="

  # 1. 画总体平均图 (overall_mean)
  echo "Generating overall mean plot..."
  python "${PLOT_PY}" \
    --results_dir "${RES_DIR}" \
    --model_name "${MODEL_NAME}" \
    --target_head_counts ${TARGET_HC} \
    --plot_type "overall_mean" \
    --output "${RES_DIR}/mean_bias_overall_mean.png"

  # 2. 画按问题分布图 (by_question_id)
  echo "Generating by-question plot..."
  python "${PLOT_PY}" \
    --results_dir "${RES_DIR}" \
    --model_name "${MODEL_NAME}" \
    --target_head_counts ${TARGET_HC} \
    --plot_type "by_question_id" \
    --output "${RES_DIR}/mean_bias_by_question.pdf"

  echo "Done for ${MODEL_NAME}."
  echo ""
done

echo "All plotting tasks completed."
