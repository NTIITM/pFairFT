#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen3-4B"
DATASET_PATH="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json"
OUT_DIR="/home/common1/hwluo/project/pFairFT/exp21/output_qwen3_4b_qid12"

mkdir -p "${OUT_DIR}"

python /home/common1/hwluo/project/pFairFT/exp21/analyze_head_patterns.py \
  --model_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUT_DIR}" \
  --qid 12

SENSITIVE_HEADS_JSON="/home/common1/hwluo/project/pFairFT/exp2_old/sensitive_heads_Qwen3-4B_top100/selected_heads_elbow.json"

python /home/common1/hwluo/project/pFairFT/exp21/plot_comparison.py \
  --input_dir "${OUT_DIR}" \
  --output_path "${OUT_DIR}/head_prompt_vs_debiased_qid12.pdf" \
  --sensitive_heads_json "${SENSITIVE_HEADS_JSON}"

echo "Done. Outputs in ${OUT_DIR}"