#!/usr/bin/env bash
set -euo pipefail

# Path configuration
EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP2_DIR="${EXP_ROOT}/exp2_old"
EXP4_DIR="${EXP_ROOT}/exp4"
EXP5_KL_DIR="${EXP_ROOT}/exp5_KL"
EXP5_CE_DIR="${EXP_ROOT}/exp5_CE"
EXP24_DIR="${EXP_ROOT}/exp24"
PY_SCRIPT="${EXP24_DIR}/evaluate_resume_fairness_top100_exp24.py"

# Base model directories
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# Model list (same as used elsewhere in the project)
MODEL_LIST=(
  "Qwen3-1.7B"
  "Qwen3-4B"
  "Qwen3-8B"
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Meta-Llama-3-8B-Instruct"
)

# GPU configuration
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

TASKS=()

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  echo "=========================================="
  echo "Preparing tasks for model: ${MODEL_NAME}"
  echo "=========================================="

  # Biased ranking CSV from exp2
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
  if [ ! -f "${CSV_PATH}" ]; then
    echo "  [SKIP] Biased ranking CSV not found: ${CSV_PATH}"
    continue
  fi

  # Base model dir
  if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
    BASE_MODEL_DIR="${QWEN_DIR}/${MODEL_NAME}"
  else
    BASE_MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_NAME}"
  fi

  if [ ! -d "${BASE_MODEL_DIR}" ]; then
    echo "  [SKIP] Base model dir not found: ${BASE_MODEL_DIR}"
    continue
  fi

  # Model type hint
  if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
    MODEL_TYPE="qwen"
  else
    MODEL_TYPE="llama"
  fi

  # 1) Baseline
  OUT_BASELINE="${EXP24_DIR}/baseline/${MODEL_NAME}/resume_top100.csv"
  mkdir -p "$(dirname "${OUT_BASELINE}")"
  TASKS+=("python ${PY_SCRIPT} --mode baseline --base_model_path ${BASE_MODEL_DIR} --biased_csv_path ${CSV_PATH} --sample_size 100 --output_csv_path ${OUT_BASELINE} --model_type ${MODEL_TYPE}")

done

echo "Total TASKS collected: ${#TASKS[@]}"

# Scheduler (same pattern as exp18/run_eval_multi_gpu.sh)
declare -A GPU_TO_PID_MAP
idx=0
num_tasks=${#TASKS[@]}

while [ $idx -lt $num_tasks ]; do
  for gpu in "${GPUS[@]}"; do
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
      echo "[Task $((idx+1))/$num_tasks] Running on GPU $gpu: $cmd"
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

# Wait for all remaining processes
echo "All tasks dispatched. Waiting for remaining processes to finish..."
for gpu in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
 done

echo "=========================================="
echo "All evaluation tasks completed (exp24)."
echo "=========================================="
