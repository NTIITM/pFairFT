#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（基于事实 / 反事实 KL 散度约束的微调脚本）
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp5/finetune_model.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验根目录（exp5 目录）
EXP5_DIR="/home/common1/hwluo/project/pFairFT/exp5"
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# LoRA 训练参数（可根据需要调整）
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
NUM_EPOCHS=3
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4  # 梯度累积步数，有效batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE=2e-5

# =========================
# Multi-GPU parallel config
# =========================
# 可通过环境变量覆盖：
# - GPUS: 例如 "0,1,2,3"；未设置则默认 0..7
# - LOG_DIR: 日志目录（默认 exp5/logs_finetune_all）
# - SKIP_IF_EXISTS: 1=若 output_dir/final_model 存在则跳过（默认 1）
#
# 注意：
# - 模型名/路径中包含 "8B" 的任务会至少占用 2 张卡
# - 每个任务会独占其分配到的卡（通过 CUDA_VISIBLE_DEVICES 控制）

LOG_DIR="${LOG_DIR:-${EXP5_DIR}/logs_finetune_all}"
mkdir -p "${LOG_DIR}"

SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a GPUS_ARR <<< "${GPUS}"
else
  GPUS_ARR=(0 1 2 3 4 5 6 7)
fi

NUM_GPUS=${#GPUS_ARR[@]}
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] No GPUs configured. Set env GPUS=0,1,..." >&2
  exit 1
fi

# -------------------------
# Collect tasks
# -------------------------
TASKS=()          # each element is a full shell command string
TASK_NAMES=()     # for logging / decision
TASK_GPU_NEED=()  # 1 or 2

shopt -s nullglob

# 遍历两个目录下的所有子目录（每个子目录视为一个模型）
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_DIR")"

  # 对应的 biased_samples 目录与 CSV（可选：如果存在则使用 top-100 偏见样本）
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

  # 为每个模型创建独立的输出目录（统一格式：lora_${MODEL_NAME}_top100）
  MODEL_OUTPUT_DIR="${EXP5_DIR}/lora_${MODEL_NAME}_top100"

  # 若训练目标已经存在则跳过（以 final_model 是否存在为准；如果你的训练产物目录不同，可以改这里）
  FINAL_DIR="${MODEL_OUTPUT_DIR}/final_model"
  if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -d "${FINAL_DIR}" ]]; then
    echo "[SKIP] ${MODEL_NAME}: final_model already exists: ${FINAL_DIR}"
    continue
  fi

  mkdir -p "$MODEL_OUTPUT_DIR"

  # 8B 模型至少 2 卡（根据名字/路径包含 8B 判断）
  gpu_need=1
  if [[ "${MODEL_NAME}" == *"8B"* ]] || [[ "${MODEL_DIR}" == *"8B"* ]]; then
    gpu_need=2
  fi

  # 构建训练命令
  if [ -f "$CSV_PATH" ]; then
    cmd="python \"$PY_SCRIPT\" \\
      --model_path \"$MODEL_DIR\" \\
      --dataset_json_path \"$DATASET_JSON\" \\
      --output_dir \"$MODEL_OUTPUT_DIR\" \\
      --sample_csv_path \"$CSV_PATH\" \\
      --sample_size 1000 \\
      --train_type \"lora\" \\
      --lora_rank \"$LORA_RANK\" \\
      --lora_alpha \"$LORA_ALPHA\" \\
      --lora_dropout \"$LORA_DROPOUT\" \\
      --num_epochs \"$NUM_EPOCHS\" \\
      --batch_size \"$BATCH_SIZE\" \\
      --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \\
      --learning_rate \"$LEARNING_RATE\" \\
      --seed 42"
  else
    cmd="python \"$PY_SCRIPT\" \\
      --model_path \"$MODEL_DIR\" \\
      --dataset_json_path \"$DATASET_JSON\" \\
      --output_dir \"$MODEL_OUTPUT_DIR\" \\
      --max_samples 2000 \\
      --balanced \\
      --train_type \"lora\" \\
      --lora_rank \"$LORA_RANK\" \\
      --lora_alpha \"$LORA_ALPHA\" \\
      --lora_dropout \"$LORA_DROPOUT\" \\
      --num_epochs \"$NUM_EPOCHS\" \\
      --batch_size \"$BATCH_SIZE\" \\
      --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \\
      --learning_rate \"$LEARNING_RATE\" \\
      --seed 42"
  fi

  TASKS+=("$cmd")
  TASK_NAMES+=("$MODEL_NAME")
  TASK_GPU_NEED+=("$gpu_need")
done

num_tasks=${#TASKS[@]}
echo "Total TASKS found: ${num_tasks}"

after_dispatch_sleep_sec=1
poll_interval_sec=5

# -------------------------
# Scheduler (supports 1/2 GPU per task)
# -------------------------
# We'll manage GPU occupancy as a simple array: GPU_PID[i] = pid or empty
# and keep a task->gpu list mapping for cleanup.

# Use numeric-indexed arrays to avoid bash associative pitfalls across versions
GPU_PID=()
for _ in "${GPUS_ARR[@]}"; do GPU_PID+=(""); done

declare -A PID_TO_GPUSET

gpu_is_free() {
  local gi="$1"
  if [[ -z "${GPU_PID[$gi]}" ]]; then
    return 0
  fi
  local pid="${GPU_PID[$gi]}"
  if kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  wait "$pid" 2>/dev/null || true
  GPU_PID[$gi]=""
  return 0
}

# Find a contiguous set of free GPUs of size N; returns indices in global var FOUND_GPU_IDX
FOUND_GPU_IDX=()
find_free_gpus() {
  local need="$1"
  FOUND_GPU_IDX=()

  # refresh states
  local i
  for i in "${!GPUS_ARR[@]}"; do
    gpu_is_free "$i" || true
  done

  if [[ "$need" -le 1 ]]; then
    for i in "${!GPUS_ARR[@]}"; do
      if [[ -z "${GPU_PID[$i]}" ]]; then
        FOUND_GPU_IDX=("$i")
        return 0
      fi
    done
    return 1
  fi

  # need==2: pick any two free GPUs (not necessarily contiguous IDs, just two slots)
  local free_list=()
  for i in "${!GPUS_ARR[@]}"; do
    if [[ -z "${GPU_PID[$i]}" ]]; then
      free_list+=("$i")
    fi
  done

  if [[ "${#free_list[@]}" -ge 2 ]]; then
    FOUND_GPU_IDX=("${free_list[0]}" "${free_list[1]}")
    return 0
  fi
  return 1
}

idx=0
while [[ $idx -lt $num_tasks ]]; do
  need="${TASK_GPU_NEED[$idx]}"

  if find_free_gpus "$need"; then
    # Build CUDA_VISIBLE_DEVICES from selected GPUs
    gpu_ids=()
    for gi in "${FOUND_GPU_IDX[@]}"; do
      gpu_ids+=("${GPUS_ARR[$gi]}")
    done
    cuda_visible_devices="$(IFS=','; echo "${gpu_ids[*]}")"

    model_name="${TASK_NAMES[$idx]}"
    cmd="${TASKS[$idx]}"

    ts="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/${ts}_${model_name}_g${need}.log"

    echo "[Task $((idx+1))/$num_tasks] GPUs(${cuda_visible_devices}) need=${need} model=${model_name}"
    echo "  cmd: ${cmd}"
    echo "  log: ${log_file}"

    # shellcheck disable=SC2086
    ( CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" bash -lc "${cmd}" ) >"${log_file}" 2>&1 &
    pid=$!

    # Mark those GPUs occupied by this pid
    for gi in "${FOUND_GPU_IDX[@]}"; do
      GPU_PID[$gi]="$pid"
    done
    PID_TO_GPUSET[$pid]="${FOUND_GPU_IDX[*]}"

    idx=$((idx + 1))
    sleep "${after_dispatch_sleep_sec}"
  else
    sleep "${poll_interval_sec}"
  fi
done

echo "All tasks dispatched. Waiting for remaining processes to finish..."

# Wait for all unique pids remaining
seen_pids=()
for gi in "${!GPUS_ARR[@]}"; do
  pid="${GPU_PID[$gi]}"
  if [[ -n "$pid" ]]; then
    seen_pids+=("$pid")
  fi
done

# Deduplicate pids
uniq_pids=()
for pid in "${seen_pids[@]}"; do
  already=false
  for upid in "${uniq_pids[@]}"; do
    if [[ "$upid" == "$pid" ]]; then
      already=true
      break
    fi
  done
  if [[ "$already" == false ]]; then
    uniq_pids+=("$pid")
  fi
done

for pid in "${uniq_pids[@]}"; do
  wait "$pid" 2>/dev/null || true
  # cleanup GPU slots associated with this pid
  if [[ -n "${PID_TO_GPUSET[$pid]:-}" ]]; then
    for gi in ${PID_TO_GPUSET[$pid]}; do
      GPU_PID[$gi]=""
    done
  fi
done

echo "=========================================="
echo "All done. KL+LoRA fine-tuning completed for all models."
echo "=========================================="
echo ""
echo "LoRA adapters (and configs) are saved in: ${EXP5_DIR}/lora_*_top100/final_model/"
