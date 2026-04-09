#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp2/evaluate_biased_sample.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 输出目录（exp2 目录）
OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp2"

# GPU 列表（每个GPU跑一个模型）
GPUS=(0 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}

# 收集所有模型目录到数组
MODEL_DIRS=()
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi
  MODEL_DIRS+=("$MODEL_DIR")
done

NUM_MODELS=${#MODEL_DIRS[@]}
echo "=========================================="
echo "Found $NUM_MODELS models to evaluate"
echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"
echo "=========================================="
echo ""

# 函数：在指定GPU上运行模型评估
run_model_on_gpu() {
  local MODEL_DIR="$1"
  local GPU_ID="$2"
  local MODEL_NAME="$(basename "$MODEL_DIR")"
  
  echo "[GPU $GPU_ID] Starting evaluation for model: $MODEL_NAME"
  echo "[GPU $GPU_ID] Model path: $MODEL_DIR"
  
  # 为每个模型创建独立的输出目录（区分名字保存）
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/biased_samples_${MODEL_NAME}"
  mkdir -p "$MODEL_OUTPUT_DIR"
  
  # 输出 CSV 文件路径
  CSV_OUTPUT="${MODEL_OUTPUT_DIR}/biased_samples_ranking.csv"
  
  # 使用 CUDA_VISIBLE_DEVICES 限制只使用指定的GPU
  # 注意：CUDA_VISIBLE_DEVICES 会将可见的GPU重新编号为从0开始
  # 所以这里我们设置 CUDA_VISIBLE_DEVICES=$GPU_ID，这样进程只能看到这个GPU（作为GPU 0）
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json" \
    --output_csv_path "$CSV_OUTPUT" \
    --device "cuda" \
    --model_type "auto"
  
  echo "[GPU $GPU_ID] Finished evaluating bias level for model: $MODEL_NAME"
  echo "[GPU $GPU_ID] Results saved to: $CSV_OUTPUT"
  echo ""
}

# 并行执行：使用后台进程
# 为每个模型分配一个GPU（当GPU可用时分配）

# 初始化GPU到PID的映射（使用关联数组）
declare -A GPU_TO_PID_MAP

for i in "${!MODEL_DIRS[@]}"; do
  MODEL_DIR="${MODEL_DIRS[$i]}"
  MODEL_NAME="$(basename "$MODEL_DIR")"
  
  # 如果所有GPU都在使用，等待一个完成
  while true; do
    # 计算当前使用的GPU数量
    ACTIVE_GPUS=0
    for GPU_ID in "${GPUS[@]}"; do
      if [[ -v GPU_TO_PID_MAP[$GPU_ID] ]] && [ -n "${GPU_TO_PID_MAP[$GPU_ID]:-}" ]; then
        PID="${GPU_TO_PID_MAP[$GPU_ID]}"
        if kill -0 "$PID" 2>/dev/null; then
          ACTIVE_GPUS=$((ACTIVE_GPUS + 1))
        else
          # 进程已完成，清理
          wait "$PID" 2>/dev/null || true
          echo "[GPU $GPU_ID] Process completed, GPU freed"
          unset GPU_TO_PID_MAP[$GPU_ID]
        fi
      fi
    done
    
    # 如果有空闲GPU，跳出循环
    if [ $ACTIVE_GPUS -lt $NUM_GPUS ]; then
      break
    fi
    
    # 如果所有GPU都在使用，等待一小段时间
    sleep 2
  done
  
  # 找到第一个可用的GPU
  GPU_ID=""
  for gpu in "${GPUS[@]}"; do
    if [[ ! -v GPU_TO_PID_MAP[$gpu] ]] || [ -z "${GPU_TO_PID_MAP[$gpu]:-}" ]; then
      GPU_ID="$gpu"
      break
    fi
  done
  
  if [ -z "$GPU_ID" ]; then
    echo "Error: No available GPU found"
    exit 1
  fi
  
  # 在后台运行，并记录PID和GPU
  run_model_on_gpu "$MODEL_DIR" "$GPU_ID" &
  PID=$!
  GPU_TO_PID_MAP[$GPU_ID]=$PID
  
  echo "Assigned model $MODEL_NAME to GPU $GPU_ID (PID: $PID)"
done

# 等待所有剩余的后台进程完成
echo ""
echo "Waiting for all remaining processes to complete..."
for GPU_ID in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$GPU_ID] ]] && [ -n "${GPU_TO_PID_MAP[$GPU_ID]:-}" ]; then
    PID="${GPU_TO_PID_MAP[$GPU_ID]}"
    echo "Waiting for process on GPU $GPU_ID (PID: $PID)..."
    wait "$PID"
    echo "[GPU $GPU_ID] Process completed"
  fi
done

echo "=========================================="
echo "All done. Bias level evaluation completed for all models."
echo "=========================================="
