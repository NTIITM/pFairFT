#!/bin/bash
# 执行exp9评估脚本
# 从biased_samples_ranking.csv选取前100个样本
# 测试max heads=10, step=5 (即测试5和10两个头数量)


CUDA_VISIBLE_DEVICES=5 python /home/common1/hwluo/project/pFairFT/exp9/evaluate_intervention_by_head_count.py \
  --model_path /mnt/nfs/huggingface/Qwen/Qwen3-4B \
  --model_type qwen \
  --sample_csv_path /home/common1/hwluo/project/pFairFT/exp2_old/biased_samples_Qwen3-4B/biased_samples_ranking.csv \
  --sample_size 100 \
  --sensitive_heads_dir /home/common1/hwluo/project/pFairFT/exp2_old/sensitive_heads_Qwen3-4B_top100 \
  --max_head_count 35 \
  --step 5 \
  --intervention_type negative \
  --output_dir /home/common1/hwluo/project/pFairFT/exp9/intervention_results_qwen3-4b_top100 \
  --batch_size 8 \
  --device cuda
