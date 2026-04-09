#!/usr/bin/env bash
set -euo pipefail

# 脚本路径
PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp17/evaluate_intervention_projection.py"

# 模型目录
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 实验目录
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2_old"
EXP17_DIR="/home/common1/hwluo/project/pFairFT/exp17"

# 数据集路径
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json"

# 公共 CSV 输出路径（用于绘图读取新结果）
CSV_PATH="${EXP17_DIR}/per_sample_intervention_projection_all_models.csv"

# 清理旧结果
if [ -f "$CSV_PATH" ]; then
  rm "$CSV_PATH"
fi

# 可用 GPU 列表（默认使用 2,3,4,5,6,7 卡；可通过 GPU_IDS 环境变量覆盖）
GPU_IDS_DEFAULT="2,3,4,5,6,7"
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 并发上限（默认等于 GPU 数量；可通过 MAX_JOBS 环境变量覆盖）
MAX_JOBS_DEFAULT="$NUM_GPUS"
MAX_JOBS="${MAX_JOBS:-$MAX_JOBS_DEFAULT}"

# 临时目录：存每张卡一个 CSV，最后再合并，避免并发写同一个 CSV
TMP_DIR="${EXP17_DIR}/tmp_parallel_csv"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

# 收集所有模型路径
MODEL_DIRS=()
for MODEL_DIR in "$LLM_RESEARCH_DIR"/* "$QWEN_DIR"/*; do
  if [ -d "$MODEL_DIR" ]; then
    MODEL_DIRS+=("$MODEL_DIR")
  fi
done

# 轮询分配：第 i 个模型 -> 第 (i % NUM_GPUS) 张卡
for IDX in "${!MODEL_DIRS[@]}"; do
  MODEL_DIR="${MODEL_DIRS[$IDX]}"
  MODEL_NAME="$(basename "$MODEL_DIR")"

  GPU_ID="${GPU_ARRAY[$((IDX % NUM_GPUS))]}"

  # 检查是否有 exp2 产出的敏感头数据
  SENSITIVE_HEADS_DIR="${EXP2_DIR}/sensitive_heads_${MODEL_NAME}_top100"
  if [ ! -d "$SENSITIVE_HEADS_DIR" ]; then
    echo "Skipping $MODEL_NAME: No sensitive heads data in $SENSITIVE_HEADS_DIR"
    continue
  fi

  # 并发控制：达到 MAX_JOBS 就等待任意一个结束
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  echo "[GPU $GPU_ID] Running Projection Intervention for: $MODEL_NAME"

  PER_GPU_CSV="${TMP_DIR}/per_sample_${MODEL_NAME}.csv"

  (
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python "$PY_SCRIPT" \
      --model_path "$MODEL_DIR" \
      --dataset_path "$DATASET_JSON" \
      --sensitive_heads_dir "$SENSITIVE_HEADS_DIR" \
      --output_dir "${EXP17_DIR}/results_${MODEL_NAME}" \
      --csv_path "$PER_GPU_CSV" \
      --intervention_strength 1.0 \
      --device "cuda"
  ) &
done

# 等待所有后台任务完成
wait

# 合并所有临时 CSV 到总 CSV（只保留一次表头）
FIRST=1
> "$CSV_PATH"
for PART in "$TMP_DIR"/*.csv; do
  if [ ! -f "$PART" ]; then
    continue
  fi
  if [ "$FIRST" -eq 1 ]; then
    cat "$PART" >> "$CSV_PATH"
    FIRST=0
  else
    tail -n +2 "$PART" >> "$CSV_PATH" || true
  fi
done

echo "All models completed. Results saved to $CSV_PATH"
