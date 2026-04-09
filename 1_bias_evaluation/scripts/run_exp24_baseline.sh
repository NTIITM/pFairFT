#!/bin/bash
# exp24: Evaluate baseline model on top 100 biased resume samples with debiased prompt suffix

BASE_MODEL="/home/common1/hwluo/model/Meta-Llama-3-8B-Instruct"
BIASED_CSV="/home/common1/hwluo/project/pFairFT/exp2/biased_samples_llama3_8b_qid33/biased_samples_ranking.csv"
OUTPUT_DIR="pFairFT/1_bias_evaluation/baseline"
mkdir -p $OUTPUT_DIR

python pFairFT/1_bias_evaluation/evaluate_resume_fairness_top100_exp24.py \
    --mode baseline \
    --base_model_path $BASE_MODEL \
    --biased_csv_path $BIASED_CSV \
    --sample_size 100 \
    --output_csv_path "$OUTPUT_DIR/resume_top100_p_yes.csv" \
    --model_type llama
