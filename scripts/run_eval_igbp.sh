#!/usr/bin/env bash
export PATH="/home/common1/hwluo/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=3,4

python /home/common1/hwluo/project/pFairFT/4_intervention_ablation/projection_intervention/evaluate_intervention_igbp.py \
    --model_path "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct" \
    --dataset_path "/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json" \
    --csv_path "/home/common1/hwluo/project/pFairFT/results/Llama-3.2-3B-Instruct/downstream_evaluation/discrim_igbp.csv"
