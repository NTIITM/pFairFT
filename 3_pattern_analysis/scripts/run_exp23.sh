#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
MODEL_NAME="${MODEL_NAME:-Meta-Llama-3-8B-Instruct}"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct/}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/discrim-eval/dataset_paired.json}"
FIRST_ADAPTER="${FIRST_ADAPTER:-${RESULTS_ROOT}/pfairft/final_model}"
SECOND_ADAPTER="${SECOND_ADAPTER:-baseline}"
FIRST_LABEL="${FIRST_LABEL:-PFairFT}"
SECOND_LABEL="${SECOND_LABEL:-baseline}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/downstream_head_analysis/qid33}"
QID="${QID:-33}"

SENSITIVE_HEADS_JSON="${SENSITIVE_HEADS_JSON:-${RESULTS_ROOT}/sensitive_heads/selected_heads_elbow.json}"

mkdir -p "${OUT_DIR}"

export TRANSFORMERS_NO_SKLEARN=1

"${PY}" "${PROJECT_ROOT}/3_pattern_analysis/model_comparison/analyze_exp23.py" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --first_adapter "${FIRST_ADAPTER}" \
  --second_adapter "${SECOND_ADAPTER}" \
  --first_label "${FIRST_LABEL}" \
  --second_label "${SECOND_LABEL}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUT_DIR}" \
  --qid "${QID}"

"${PY}" "${PROJECT_ROOT}/3_pattern_analysis/model_comparison/plot_exp23.py" \
  --input_dir "${OUT_DIR}" \
  --output_path "${OUT_DIR}/head_pfairft_vs_global_lora_ce_qid33.pdf" \
  --sensitive_heads_json "${SENSITIVE_HEADS_JSON}" \
  --first_label "${FIRST_LABEL}" \
  --second_label "${SECOND_LABEL}"

echo "Done. Outputs in ${OUT_DIR}"
