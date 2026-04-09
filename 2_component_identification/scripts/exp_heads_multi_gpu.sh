#!/usr/bin/env bash
set -euo pipefail

# Python 脚本路径（Attention Heads 种族敏感度分析）
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp2/analyze_race_sensitive_heads.py"

# 模型根目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验根目录（exp2 目录）
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2"

# Resume 数据集 JSON 路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# Batch size
BATCH_SIZE=8

# GPU 列表（每个GPU跑一个模型）
GPUS=(0 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}

# 收集所有模型目录到数组（只包含有对应CSV的模型）
MODEL_DIRS=()
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  # 只处理目录
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi
  
  MODEL_NAME="$(basename "$MODEL_DIR")"
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
  
  # 只添加有CSV文件的模型
  if [ -f "$CSV_PATH" ]; then
    MODEL_DIRS+=("$MODEL_DIR")
  else
    echo "WARNING: CSV not found for model $MODEL_NAME: $CSV_PATH"
    echo "Skip heads analysis for this model."
  fi
done

NUM_MODELS=${#MODEL_DIRS[@]}
echo "=========================================="
echo "Found $NUM_MODELS models with CSV files to analyze"
echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"
echo "=========================================="
echo ""

# 函数：在指定GPU上运行模型分析
run_model_on_gpu() {
  local MODEL_DIR="$1"
  local GPU_ID="$2"
  local MODEL_NAME="$(basename "$MODEL_DIR")"
  
  echo "[GPU $GPU_ID] Starting race-sensitive heads analysis for model: $MODEL_NAME"
  echo "[GPU $GPU_ID] Model path: $MODEL_DIR"
  
  # 对应的 biased_samples 目录与 CSV（需要事先由 exp_sample.sh / evaluate_biased_sample.py 生成）
  BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
  CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"
  
  # 再次检查CSV文件是否存在（双重保险）
  if [ ! -f "$CSV_PATH" ]; then
    echo "[GPU $GPU_ID] ERROR: CSV not found for model $MODEL_NAME: $CSV_PATH"
    echo "[GPU $GPU_ID] Skip heads analysis for this model."
    return 1
  fi
  
  # 为每个模型创建 heads 分析输出目录
  MODEL_OUTPUT_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  mkdir -p "$MODEL_OUTPUT_DIR"
  
  # 使用 CUDA_VISIBLE_DEVICES 限制只使用指定的GPU
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$PY_SCRIPT" \
    --model_path "$MODEL_DIR" \
    --dataset_json_path "$DATASET_JSON" \
    --sample_csv_path "$CSV_PATH" \
    --sample_size 100 \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda" \
    --model_type "auto"
  
  echo "[GPU $GPU_ID] Finished race-sensitive heads analysis for model: $MODEL_NAME"
  echo "[GPU $GPU_ID] Results saved to: $MODEL_OUTPUT_DIR"
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
echo "All done. Race-sensitive heads analysis completed for all models."
echo "=========================================="
