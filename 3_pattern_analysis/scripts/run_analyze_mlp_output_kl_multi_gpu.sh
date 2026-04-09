#!/usr/bin/env bash
set -euo pipefail

# One-click MLP output KL/mean-diff analysis for all models (baseline + MLP intervention)

EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP2_DIR="${EXP_ROOT}/exp2_old"
EXP15_DIR="${EXP_ROOT}/exp15"
EXP20_DIR="${EXP_ROOT}/exp20"

PY_BASE="${EXP20_DIR}/analyze_mlp_output_kl_resume.py"
PY_INT_HEAD="${EXP20_DIR}/analyze_mlp_output_kl_resume_with_intervention.py"
PY_PLOT="${EXP20_DIR}/plot_mlp_output_kl_layers.py"
PY_PLOT_RACE="${EXP20_DIR}/plot_mlp_input_p_race_layers.py"

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

TASKS=()

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
  if [ ! -f "${CSV_PATH}" ]; then
    echo "[MLP-OUT] Skip ${MODEL_NAME}: missing ${CSV_PATH}"
    continue
  fi

  if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
    MODEL_DIR="${QWEN_DIR}/${MODEL_NAME}"
  else
    MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_NAME}"
  fi
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "[MLP-OUT] Skip ${MODEL_NAME}: missing model dir ${MODEL_DIR}"
    continue
  fi

  OUT_DIR="${EXP20_DIR}/${MODEL_NAME}"
  mkdir -p "${OUT_DIR}"

  BASE_METRIC="${OUT_DIR}/mlp_kl_p_yes.npy"
  INT_METRIC="${OUT_DIR}/mlp_kl_p_yes_intervened.npy"

  # Note: Always re-run to ensure we capture "input to next module" instead of "MLP output"
  TASKS+=("python ${PY_BASE} --model_path ${MODEL_DIR} --biased_csv_path ${CSV_PATH} --output_dir ${OUT_DIR} --sample_size 100 --batch_size 8")

  SENSITIVE_HEADS_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  SELECTED_HEADS_JSON="${SENSITIVE_HEADS_DIR}/selected_heads_elbow.json"
  EMBEDDINGS_PKL="${SENSITIVE_HEADS_DIR}/results.pkl"

  if [ -f "${SELECTED_HEADS_JSON}" ] && [ -f "${EMBEDDINGS_PKL}" ]; then
    TASKS+=("python ${PY_INT_HEAD} --model_path ${MODEL_DIR} --biased_csv_path ${CSV_PATH} --sensitive_heads_path ${SELECTED_HEADS_JSON} --embeddings_path ${EMBEDDINGS_PKL} --output_dir ${OUT_DIR} --sample_size 100 --batch_size 8")
  else
    echo "[MLP-OUT] Skip head intervention for ${MODEL_NAME}: missing ${SELECTED_HEADS_JSON} or ${EMBEDDINGS_PKL}"
  fi

done

echo "[MLP-OUT] Total tasks: ${#TASKS[@]}"

# Simple GPU scheduler
declare -A GPU_TO_PID_MAP
idx=0
num_tasks=${#TASKS[@]}

# Cleanup function to kill all background tasks on exit
cleanup() {
  echo "[MLP-OUT] Cleaning up background tasks..."
  for gpu in "${!GPU_TO_PID_MAP[@]}"; do
    pid="${GPU_TO_PID_MAP[$gpu]}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing task $pid on GPU $gpu"
      kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

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
      # Optional: Double check if GPU actually has enough memory (at least 2GB free)
      # free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
      # if [ "$free_mem" -lt 2000 ]; then
      #   echo "[MLP-OUT] GPU $gpu looks busy (only ${free_mem}MB free), skipping..."
      #   continue
      # fi

      cmd="${TASKS[$idx]}"
      echo "[Task $((idx+1))/$num_tasks] GPU $gpu: $cmd"
      # Use setsid or subshell to ensure we can track the exact PID
      CUDA_VISIBLE_DEVICES="$gpu" $cmd &
      GPU_TO_PID_MAP[$gpu]=$!
      idx=$((idx+1))
      sleep 2 # Increase sleep to avoid race conditions on GPU initialization
    fi
  done
  if [ $idx -lt $num_tasks ]; then
    sleep 5
  fi
done

# Wait for all
for gpu in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
 done

echo "[MLP-OUT] All analysis tasks completed. Running plotting scripts..."
python "${PY_PLOT}" --exp20_root "${EXP20_DIR}"
python "${PY_PLOT_RACE}" --exp20_root "${EXP20_DIR}"

echo "[MLP-OUT] Done."
