#!/bin/bash

set -euo pipefail

# Use HF mirror and store all HF cache under this experiment directory for reuse
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/common1/hwluo/project/pFairFT/exp26/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false

# Configuration
MODELS=(
    "Qwen3-1.7B"
    "Qwen3-4B"
    "Qwen3-8B"
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Meta-Llama-3-8B-Instruct"
)

PATHS=(
    "/mnt/nfs/huggingface/Qwen/Qwen3-1.7B"
    "/mnt/nfs/huggingface/Qwen/Qwen3-4B"
    "/mnt/nfs/huggingface/Qwen/Qwen3-8B"
    "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct"
    "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct"
    "/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct"
)

OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp26/results"
CACHE_DIR="/home/common1/hwluo/project/pFairFT/exp26/pre_logits_cache"
NUM_SAMPLES=100
MAX_LENGTH=512
STRENGTH=1.0

# GPU configuration (override by exporting GPUS="0 1 2 3")
if [ -n "${GPUS:-}" ]; then
  read -r -a GPU_LIST <<< "${GPUS}"
else
  GPU_LIST=(0 1 2 3 4 5 6 7)
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp26/compute_exp26_metrics.py"

run_on_gpu() {
  local gpu="$1"; shift
  local cmd=("$@")
  echo "[GPU $gpu] ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  echo $!
}

wait_for_slot_and_launch() {
  local -n _gpu_to_pid=$1
  local -n _gpu_list=$2
  shift 2
  local cmd=("$@")

  while true; do
    for gpu in "${_gpu_list[@]}"; do
      local is_free=0
      if [ -z "${_gpu_to_pid[$gpu]+x}" ]; then
        is_free=1
      else
        local pid="${_gpu_to_pid[$gpu]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          wait "$pid" 2>/dev/null || true
          unset _gpu_to_pid[$gpu]
          is_free=1
        fi
      fi

      if [ "$is_free" -eq 1 ]; then
        local pid
        pid=$(run_on_gpu "$gpu" "${cmd[@]}")
        _gpu_to_pid[$gpu]="$pid"
        sleep 1
        return 0
      fi
    done
    sleep 5
  done
}

wait_all() {
  local -n _gpu_to_pid=$1
  for gpu in "${!_gpu_to_pid[@]}"; do
    local pid="${_gpu_to_pid[$gpu]}"
    wait "$pid" 2>/dev/null || true
  done
}

echo "Starting Experiment 26: KL/CE Comparison on OpenWebText"
echo "========================================================"
echo "Using GPUs: ${GPU_LIST[*]}"

declare -A GPU_TO_PID_MAP

# ------------------------
# Phase 1: Baseline FIRST
# ------------------------
echo "[Phase 1] Running baseline for all models (must finish before other tasks)"
for i in "${!MODELS[@]}"; do
  M_NAME="${MODELS[$i]}"
  M_PATH="${PATHS[$i]}"

  cmd=(python "$PY_SCRIPT" \
    --model_path "$M_PATH" \
    --model_name "$M_NAME" \
    --task baseline \
    --num_samples "$NUM_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --intervention_strength "$STRENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR")

  wait_for_slot_and_launch GPU_TO_PID_MAP GPU_LIST "${cmd[@]}"
done

echo "[Phase 1] All baseline tasks dispatched. Waiting for completion..."
wait_all GPU_TO_PID_MAP
unset GPU_TO_PID_MAP

declare -A GPU_TO_PID_MAP

# ------------------------
# Phase 2: Remaining tasks
# ------------------------
echo "[Phase 2] Dispatching remaining tasks in parallel"
TASKS=("lora_ce" "lora_kl" "exp25_partial" "exp25_all" "exp4")

for i in "${!MODELS[@]}"; do
  M_NAME="${MODELS[$i]}"
  M_PATH="${PATHS[$i]}"

  for T in "${TASKS[@]}"; do
    cmd=(python "$PY_SCRIPT" \
      --model_path "$M_PATH" \
      --model_name "$M_NAME" \
      --task "$T" \
      --num_samples "$NUM_SAMPLES" \
      --max_length "$MAX_LENGTH" \
      --intervention_strength "$STRENGTH" \
      --output_dir "$OUTPUT_DIR" \
      --cache_dir "$CACHE_DIR")

    wait_for_slot_and_launch GPU_TO_PID_MAP GPU_LIST "${cmd[@]}"
  done

done

echo "[Phase 2] All tasks dispatched. Waiting for remaining processes to finish..."
wait_all GPU_TO_PID_MAP

echo "========================================================"
echo "Experiment 26 finished. Results are in $OUTPUT_DIR"
echo "Master CSV: $OUTPUT_DIR/exp26_all_results_openwebtext.csv"
echo "========================================================"