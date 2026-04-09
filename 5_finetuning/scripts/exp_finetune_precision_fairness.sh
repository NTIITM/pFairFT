#!/usr/bin/env bash
set -euo pipefail

# EXP4: Precision Fine-tuning with Fairness Constraints (Multi-GPU version)
# 
# This script performs precision fine-tuning with fairness constraints:
# 1. Select race-sensitive heads from exp2_old output.
# 2. Apply LoRA for parameter-efficient fine-tuning.
# 3. Use fairness constraint loss: L = L_task + λ * L_f
#    - L_task: KL divergence loss
#    - L_f: Fairness loss (projection to neutral point)

# Experiment Root
EXP_DIR="/home/common1/hwluo/project/pFairFT/exp4"
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"

# Python Script Path
PY_SCRIPT="${EXP_DIR}/finetune_precision_fairness.py"

# Model Root Directories
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# Resume Dataset JSON Path
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# LoRA Training Parameters
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
NUM_EPOCHS=3
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
FAIRNESS_LAMBDA=0.1

# =========================
# Multi-GPU parallel config (Synced from exp5)
# =========================
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs_finetune_all}"
mkdir -p "${LOG_DIR}"

SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a GPUS_ARR <<< "${GPUS}"
else
  GPUS_ARR=(0 1 2 3 4 5 6 7)
fi

TASKS=()
TASK_NAMES=()
TASK_GPU_NEED=()

for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  if [ ! -d "$MODEL_DIR" ]; then continue; fi

  MODEL_NAME="$(basename "$MODEL_DIR")"
  HEADS_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  
  if [ ! -d "$HEADS_DIR" ] || [ ! -f "${HEADS_DIR}/results.pkl" ]; then
    echo "WARNING: Heads analysis results not found for model ${MODEL_NAME}. Skipping."
    continue
  fi

  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
  MODEL_OUTPUT_DIR="${EXP_DIR}/precision_fairness_${MODEL_NAME}_top100"

  if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -d "${MODEL_OUTPUT_DIR}/final_model" ]]; then
    echo "[SKIP] ${MODEL_NAME} already exists."
    continue
  fi

  gpu_need=1
  # For 8B-class models, allocate at least 3 GPUs to avoid OOM
  if [[ "${MODEL_NAME}" == *"8B"* ]] || [[ "${MODEL_DIR}" == *"8B"* ]]; then
    gpu_need=3
  fi

  if [ -f "$CSV_PATH" ]; then
    cmd="python \"$PY_SCRIPT\" --model_path \"$MODEL_DIR\" --dataset_json_path \"$DATASET_JSON\" --heads_analysis_dir \"$HEADS_DIR\" --output_dir \"$MODEL_OUTPUT_DIR\" --sample_csv_path \"$CSV_PATH\" --sample_size 1000 --lora_rank \"$LORA_RANK\" --lora_alpha \"$LORA_ALPHA\" --lora_dropout \"$LORA_DROPOUT\" --num_epochs \"$NUM_EPOCHS\" --batch_size \"$BATCH_SIZE\" --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" --learning_rate \"$LEARNING_RATE\" --fairness_lambda \"$FAIRNESS_LAMBDA\" --seed 42"
  else
    cmd="python \"$PY_SCRIPT\" --model_path \"$MODEL_DIR\" --dataset_json_path \"$DATASET_JSON\" --heads_analysis_dir \"$HEADS_DIR\" --output_dir \"$MODEL_OUTPUT_DIR\" --max_samples 2000 --balanced --lora_rank \"$LORA_RANK\" --lora_alpha \"$LORA_ALPHA\" --lora_dropout \"$LORA_DROPOUT\" --num_epochs \"$NUM_EPOCHS\" --batch_size \"$BATCH_SIZE\" --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" --learning_rate \"$LEARNING_RATE\" --fairness_lambda \"$FAIRNESS_LAMBDA\" --seed 42"
  fi

  TASKS+=("$cmd")
  TASK_NAMES+=("$MODEL_NAME")
  TASK_GPU_NEED+=("$gpu_need")
done

# Prioritize 8B tasks (gpu_need larger) so they start first.
# This prevents smaller jobs from fragmenting free GPUs and blocking 8B jobs.
if [[ "${#TASKS[@]}" -gt 1 ]]; then
  N=${#TASKS[@]}
  for ((i=0; i<N; i++)); do
    for ((j=i+1; j<N; j++)); do
      if (( TASK_GPU_NEED[j] > TASK_GPU_NEED[i] )); then
        tmp="${TASKS[i]}"; TASKS[i]="${TASKS[j]}"; TASKS[j]="$tmp"
        tmp="${TASK_NAMES[i]}"; TASK_NAMES[i]="${TASK_NAMES[j]}"; TASK_NAMES[j]="$tmp"
        tmp="${TASK_GPU_NEED[i]}"; TASK_GPU_NEED[i]="${TASK_GPU_NEED[j]}"; TASK_GPU_NEED[j]="$tmp"
      fi
    done
  done
fi

GPU_PID=()
for _ in "${GPUS_ARR[@]}"; do GPU_PID+=(""); done
declare -A PID_TO_GPUSET

find_free_gpus() {
  local need="$1"
  FOUND_GPU_IDX=()
  for i in "${!GPUS_ARR[@]}"; do
    if [[ -n "${GPU_PID[$i]}" ]] && ! kill -0 "${GPU_PID[$i]}" 2>/dev/null; then GPU_PID[$i]=""; fi
  done
  local free_list=()
  for i in "${!GPUS_ARR[@]}"; do
    if [[ -z "${GPU_PID[$i]}" ]]; then free_list+=("$i"); fi
  done
  if [[ "${#free_list[@]}" -ge "$need" ]]; then
    for ((j=0; j<need; j++)); do FOUND_GPU_IDX+=("${free_list[$j]}"); done
    return 0
  fi
  return 1
}

idx=0
while [[ $idx -lt ${#TASKS[@]} ]]; do
  need="${TASK_GPU_NEED[$idx]}"
  if find_free_gpus "$need"; then
    gpu_ids=()
    for gi in "${FOUND_GPU_IDX[@]}"; do gpu_ids+=("${GPUS_ARR[$gi]}"); done
    cuda_visible_devices="$(IFS=','; echo "${gpu_ids[*]}")"
    model_name="${TASK_NAMES[$idx]}"
    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/${ts}_${model_name}.log"
    echo "[Task $((idx+1))/${#TASKS[@]}] GPU(${cuda_visible_devices}) ${model_name}"
    ( CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" bash -c "${TASKS[$idx]}" ) >"${log_file}" 2>&1 &
    pid=$!
    for gi in "${FOUND_GPU_IDX[@]}"; do GPU_PID[$gi]="$pid"; done
    PID_TO_GPUSET[$pid]="${FOUND_GPU_IDX[*]}"
    idx=$((idx + 1))
    sleep 1
  else
    sleep 5
  fi
done
wait
echo "All exp4 tasks completed."
