#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="/home/common1/hwluo/project/pFairFT"
EXP2_DIR="${EXP_ROOT}/exp2_old"
EXP15_DIR="${EXP_ROOT}/exp15"
EXP20_DIR="${EXP_ROOT}/exp20"

PY_BASELINE="${EXP20_DIR}/analyze_head_kl_resume.py"
PY_MLP="${EXP20_DIR}/analyze_head_kl_resume_mlp.py"

LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

RESUME_JSON="${EXP_ROOT}/data/resume/qwen_summaries_with_race.json"

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
    echo "[SKIP] Missing biased ranking CSV: ${CSV_PATH}"
    continue
  fi

  if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
    MODEL_DIR="${QWEN_DIR}/${MODEL_NAME}"
  else
    MODEL_DIR="${LLM_RESEARCH_DIR}/${MODEL_NAME}"
  fi
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "[SKIP] Missing model dir: ${MODEL_DIR}"
    continue
  fi

  OUT_DIR="${EXP20_DIR}/${MODEL_NAME}"
  mkdir -p "${OUT_DIR}"

  # 1) Baseline analysis (heads) - skip if kl_p_yes.npy already exists
  BASELINE_OUT="${OUT_DIR}/kl_p_yes.npy"
  if [ -f "${BASELINE_OUT}" ]; then
    echo "[BASELINE] Skip ${MODEL_NAME}: ${BASELINE_OUT} already exists"
  else
    TASKS+=("python ${PY_BASELINE} --model_path ${MODEL_DIR} --dataset_json_path ${RESUME_JSON} --biased_csv_path ${CSV_PATH} --sample_size 100 --batch_size 1 --output_dir ${OUT_DIR}")
  fi

  # 2) MLP intervention analysis (heads) - skip if kl_p_yes_mlp.npy already exists
  MLP_SELECTED_JSON="${EXP15_DIR}/mlp_elbow_${MODEL_NAME}/selected_mlp_layers_elbow.json"
  MLP_OUT="${OUT_DIR}/kl_p_yes_mlp.npy"
  if [ -f "${MLP_OUT}" ]; then
    echo "[MLP] Skip ${MODEL_NAME}: ${MLP_OUT} already exists"
  else
    if [ -f "${MLP_SELECTED_JSON}" ]; then
      TASKS+=("python ${PY_MLP} --model_path ${MODEL_DIR} --dataset_json_path ${RESUME_JSON} --biased_csv_path ${CSV_PATH} --sample_size 100 --batch_size 1 --output_dir ${OUT_DIR} --mlp_selected_path ${MLP_SELECTED_JSON}")
    else
      echo "[MLP] Skip ${MODEL_NAME}: missing ${MLP_SELECTED_JSON}"
    fi
  fi

done

echo "Total tasks: ${#TASKS[@]}"

# Scheduler (same pattern as exp16/exp18)
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
for gpu in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
 done

echo "=========================================="
echo "exp20 analysis completed (baseline + MLP)."
echo "=========================================="
