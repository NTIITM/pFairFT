#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP25_DIR="${EXP_ROOT}/exp25"
LOG_DIR="${EXP25_DIR}/logs_mmlu"
mkdir -p "${LOG_DIR}"

LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

MODEL_LIST=(
  "Qwen3-1.7B"
  "Qwen3-4B"
  "Qwen3-8B"
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Meta-Llama-3-8B-Instruct"
)

GPUS=(0 1 2 3 5 6 7)

SPLIT="validation"
MAX_SAMPLES=-1

TASKS=()

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
    MODEL_DIR="${QWEN_DIR}/${MODEL_NAME}"
  else
    MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_NAME}"
  fi

  if [ ! -d "${MODEL_DIR}" ]; then
    echo "[SKIP] Missing model dir: ${MODEL_DIR}"
    continue
  fi

  OUT_DIR="${EXP25_DIR}/results_${MODEL_NAME}"
  mkdir -p "${OUT_DIR}"

  for MODE in baseline partial all; do
    OUT_JSON="${OUT_DIR}/mmlu_${SPLIT}_${MAX_SAMPLES}_${MODE}.json"
    if [ -f "${OUT_JSON}" ]; then
      echo "[SKIP] ${MODEL_NAME} ${MODE}: ${OUT_JSON} exists"
      continue
    fi
    TASKS+=("python ${EXP25_DIR}/evaluate_mmlu_intervention.py --model_path ${MODEL_DIR} --device cuda --split ${SPLIT} --max_samples ${MAX_SAMPLES} --intervention_mode ${MODE} --output_json ${OUT_JSON}")
  done

done

echo "Total tasks: ${#TASKS[@]}"

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
      log_file="${LOG_DIR}/task_${idx}.log"
      echo "[Task $((idx+1))/$num_tasks] GPU $gpu: $cmd"
      CUDA_VISIBLE_DEVICES="$gpu" $cmd > "$log_file" 2>&1 &
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
for gpu in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
 done

echo "MMLU evaluation completed. Summarizing..."
python "${EXP25_DIR}/summarize_mmlu_exp25.py" --exp25_root "${EXP25_DIR}" --split "${SPLIT}" --max_samples "${MAX_SAMPLES}" 

echo "Done."
