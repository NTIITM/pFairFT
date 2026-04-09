#!/usr/bin/env bash
# 运行 exp10 的 discrim-eval 偏见程度干预实验，对每个模型、每个 step 数量计算结果。
# 每个模型运行 6 个 step（含 0）：head_count = 0, step, 2*step, ..., 5*step。
# max_head_count = 5*step，以保证恰好 6 个点。
# 针对 discrim-eval 数据集，计算每个 decision_question_id 的平均 p(yes) 绝对差。
# 结果保存到各模型对应目录下的不同 CSV：
#   - results_sensitive_heads.csv   (敏感头干预)
#   - results_random_heads.csv      (随机头干预)

set -euo pipefail

LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"
EXP2_OLD_DIR="/home/common1/hwluo/project/pFairFT/exp2"
EXP10_DIR="/home/common1/hwluo/project/pFairFT/exp10"

# GPU 列表：可通过环境变量覆盖，例如：GPUS="0 1 2 3" bash run_...
if [ -n "${GPUS:-}" ]; then
  read -r -a GPU_LIST <<< "${GPUS}"
else
  GPU_LIST=(0 1 2 3 5 7)
fi

# 模型路径 -> step 映射（与手写表一致）
# Qwen 1.7b=3, Qwen 4b=5, Qwen 8b=5; Llama 1b=2, Llama 3b=7, Llama 8b=9
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

get_model_type() {
  case "$1" in
    Qwen3-*)   echo "qwen" ;;
    *)         echo "auto" ;;
  esac
}

MODEL_LIST=(
  "${QWEN_DIR}/Qwen3-1.7B"
  "${QWEN_DIR}/Qwen3-4B"
  "${QWEN_DIR}/Qwen3-8B"
  "${LLM_RESEARCH_DIR}/Llama-3.2-1B-Instruct"
  "${LLM_RESEARCH_DIR}/Llama-3.2-3B-Instruct"
  "${LLM_RESEARCH_DIR}/Meta-Llama-3-8B-Instruct"
)

TASKS=()

for MODEL_DIR in "${MODEL_LIST[@]}"; do
  if [ ! -d "$MODEL_DIR" ]; then
    echo "[SKIP] Missing model dir: $MODEL_DIR"
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  STEP="$(get_step "$MODEL_NAME")"
  MODEL_TYPE="$(get_model_type "$MODEL_NAME")"
  MAX_HEAD=$((5 * STEP))

  HEADS_DIR="${EXP2_OLD_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  OUTPUT_DIR="${EXP10_DIR}/intervention_results_${MODEL_NAME}_discrim_eval"

  if [ ! -f "${HEADS_DIR}/results.pkl" ]; then
    echo "[SKIP] $MODEL_NAME: results.pkl not found in $HEADS_DIR"
    continue
  fi

  mkdir -p "$OUTPUT_DIR"

  # 两个任务：sensitive / random
  if [ ! -f "${OUTPUT_DIR}/results_sensitive_heads.csv" ]; then
    TASKS+=("python ${EXP10_DIR}/evaluate_intervention_discrim_eval_head_count.py --model_path ${MODEL_DIR} --model_type ${MODEL_TYPE} --device cuda --output_dir ${OUTPUT_DIR} --embeddings_path ${HEADS_DIR}/results.pkl --intervention_mode negative --max_head_count ${MAX_HEAD} --step ${STEP} --results_csv_name results_sensitive_heads.csv --seed 42")
  else
    echo "[SKIP] $MODEL_NAME sensitive task: ${OUTPUT_DIR}/results_sensitive_heads.csv already exists."
  fi

  if [ ! -f "${OUTPUT_DIR}/results_random_heads.csv" ]; then
    TASKS+=("python ${EXP10_DIR}/evaluate_intervention_discrim_eval_head_count.py --model_path ${MODEL_DIR} --model_type ${MODEL_TYPE} --device cuda --output_dir ${OUTPUT_DIR} --embeddings_path ${HEADS_DIR}/results.pkl --intervention_mode negative_random --max_head_count ${MAX_HEAD} --step ${STEP} --results_csv_name results_random_heads.csv --seed 42")
  else
    echo "[SKIP] $MODEL_NAME random task: ${OUTPUT_DIR}/results_random_heads.csv already exists."
  fi

done

echo "Total tasks: ${#TASKS[@]}"

declare -A GPU_TO_PID_MAP
idx=0
num_tasks=${#TASKS[@]}

while [ $idx -lt $num_tasks ]; do
  for gpu in "${GPU_LIST[@]}"; do
    if [ $idx -ge $num_tasks ]; then
      break
    fi

    is_free=false
    if [[ ! -v GPU_TO_PID_MAP[$gpu] ]]; then
      is_free=true
    else
      pid="${GPU_TO_PID_MAP[$gpu]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        unset GPU_TO_PID_MAP[$gpu]
        is_free=true
      fi
    fi

    if [ "$is_free" = true ]; then
      cmd="${TASKS[$idx]}"
      echo "[Task $((idx+1))/$num_tasks] GPU $gpu: $cmd"
      CUDA_VISIBLE_DEVICES="$gpu" $cmd &
      GPU_TO_PID_MAP[$gpu]=$!
      idx=$((idx + 1))
      sleep 1
    fi
  done

  if [ $idx -lt $num_tasks ]; then
    sleep 5
  fi
 done

echo "All tasks dispatched. Waiting..."
for gpu in "${GPU_LIST[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
 done

echo "=========================================="
echo "All models done."
echo "=========================================="

# After evaluation, generate grouped 3x1 plots (vertical) for Llama and Qwen
python "${EXP10_DIR}/plot_discrim_eval_head_count.py" \
  --results_dir "${EXP10_DIR}" \
  --plot_type grouped_3x1 \
  --output "${EXP10_DIR}/mean_bias_by_group"
