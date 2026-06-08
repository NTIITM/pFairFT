#!/usr/bin/env bash
export PATH="/home/common1/hwluo/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=3,4
python /home/common1/hwluo/project/pFairFT/4_intervention_ablation/projection_intervention/train_igbp_probes.py \
  --model_path "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct" \
  --dataset_json_path "/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json" \
  --output_dir "/home/common1/hwluo/project/pFairFT/results/Llama-3.2-3B-Instruct/igbp" \
  --num_iterations 5
