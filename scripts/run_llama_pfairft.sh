#!/usr/bin/env bash
set -euo pipefail
# Set path to include anaconda python
export PATH="/home/common1/hwluo/anaconda3/bin:$PATH"

# GPU allocations: 3,4
export CUDA_VISIBLE_DEVICES=3,4

MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct"
MODEL_NAME="Llama-3.2-3B-Instruct"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"

JSON_PATH="${DATA_DIR}/resume/qwen_summaries_with_race.json"
BIASED_DIR="${RESULTS_ROOT}/biased_samples"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

# ================================================================
# PFairFT (CE + affine-transform fairness loss on selected heads)
# ================================================================
PFAIRFT_DIR="${RESULTS_ROOT}/pfairft"
echo "============================================"
echo "Fine-tuning Llama-3 with PFairFT..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --loss_type fairness_ce --fairness_lambda 0.1 --seed 42

echo "Evaluating PFairFT Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${RESULTS_ROOT}/downstream_evaluation/resume_pfairft.csv" \
    --device "cuda" --model_type "llama"

echo "Evaluating PFairFT MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --out_json "${RESULTS_ROOT}/downstream_evaluation/mmlu_ce_pfairft.json"

echo "Evaluating PFairFT Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --csv_path "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft.csv" \
    --device "cuda" --model_type "llama"

echo "============================================"
echo "Generate Figure 8 with PFairFT..."
python "${PROJECT_ROOT}/6_downstream_evaluation/plot_figure8.py" \
    --baseline_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_baseline.csv" \
    --pfairft_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft.csv" \
    --pfairft_kl_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft_kl.csv" \
    --global_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_global.csv" \
    --out_pdf "${RESULTS_ROOT}/downstream_evaluation/Figure8_Llama_with_CE.pdf" \
    --model_label "Llama-3.2-3B" || echo "Plotting failed, usually because previous CSVs are missing"

echo "All done! Results at: ${RESULTS_ROOT}/downstream_evaluation"
