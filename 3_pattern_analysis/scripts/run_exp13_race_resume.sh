#!/usr/bin/env bash
set -euo pipefail

# Example runner for exp13 (Resume dataset, race terms).
# Adjust MODEL_PATH before running.

python /home/common1/hwluo/project/pFairFT/exp13/analyze_mlp_in_out_similarity.py \
  --model_path "/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct" \
  --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json" \
  --output_dir "/home/common1/hwluo/project/pFairFT/exp13/mlp_in_out_race_resume" \
  --attribute_token "White" \
  --sensitive_terms White Black \
  --max_length 512 \
  --max_samples 200 \
  --batch_size 8

