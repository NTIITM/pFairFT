#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct/"
DATASET_PATH="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json"
EXP4_ADAPTER="/home/common1/hwluo/project/pFairFT/exp4/precision_fairness_Meta-Llama-3-8B-Instruct_top100/final_model"
EXP5_ADAPTER="/home/common1/hwluo/project/pFairFT/exp5/lora_Meta-Llama-3-8B-Instruct_top100/final_model"
OUT_DIR="/home/common1/hwluo/project/pFairFT/exp23/output_llama3_8b_qid33"
QID=33

SENSITIVE_HEADS_JSON="/home/common1/hwluo/project/pFairFT/exp2_old/sensitive_heads_Meta-Llama-3-8B-Instruct_top100/selected_heads_elbow.json"

mkdir -p "${OUT_DIR}"

export TRANSFORMERS_NO_SKLEARN=1

python /home/common1/hwluo/project/pFairFT/exp23/analyze_exp23.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --exp4_adapter "${EXP4_ADAPTER}" \
  --exp5_adapter "${EXP5_ADAPTER}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUT_DIR}" \
  --qid "${QID}"

python /home/common1/hwluo/project/pFairFT/exp23/plot_exp23.py \
  --input_dir "${OUT_DIR}" \
  --output_path "${OUT_DIR}/head_exp4_vs_exp5_qid33.pdf" \
  --sensitive_heads_json "${SENSITIVE_HEADS_JSON}"

echo "Done. Outputs in ${OUT_DIR}"
