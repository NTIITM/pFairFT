#!/usr/bin/env bash
set -euo pipefail
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

# PfairFT-KL (Already completed during Step 4, we will copy its results for plotting)
PFAIRFT_KL_DIR="${RESULTS_ROOT}/precision_fairness"
DISCRIM_KL="${RESULTS_ROOT}/downstream_evaluation/discrim_finetuned.csv"
RESUME_KL="${RESULTS_ROOT}/downstream_evaluation/resume_finetuned.csv" 
# Create a copy so we can name it neatly for plotting
cp "${DISCRIM_KL}" "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft_kl.csv" 2>/dev/null || true
cp "${RESUME_KL}" "${RESULTS_ROOT}/downstream_evaluation/resume_pfairft_kl.csv" 2>/dev/null || true

# PfairFT Variant (MSE Anchor Projection)
PFAIRFT_DIR="${RESULTS_ROOT}/pfairft"
echo "============================================"
echo "Running finetuning for PFairFT..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --loss_type fairness --fairness_lambda 0.1 --seed 42

echo "Evaluating Resume for PFairFT..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${RESULTS_ROOT}/downstream_evaluation/resume_pfairft.csv" \
    --device "cuda" --model_type "llama"

echo "Evaluating MMLU CE for PFairFT..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --out_json "${RESULTS_ROOT}/downstream_evaluation/mmlu_ce_pfairft.json"

echo "Evaluating Discrim-Eval for PFairFT..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --csv_path "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft.csv" \
    --device "cuda" --model_type "llama"


# Global Variant
GLOBAL_DIR="${RESULTS_ROOT}/global"
echo "============================================"
echo "Running finetuning for Global..."
python "${PROJECT_ROOT}/5_finetuning/finetune_global_lora.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --output_dir "${GLOBAL_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --train_type lora \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --seed 42

echo "Evaluating Resume for Global..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode global \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${RESULTS_ROOT}/downstream_evaluation/resume_global.csv" \
    --device "cuda" --model_type "llama"

echo "Evaluating MMLU CE for Global..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --out_json "${RESULTS_ROOT}/downstream_evaluation/mmlu_ce_global.json"

echo "Evaluating Discrim-Eval for Global..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode global \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --csv_path "${RESULTS_ROOT}/downstream_evaluation/discrim_global.csv" \
    --device "cuda" --model_type "llama"

echo "============================================"
echo "Plotting Figure 8..."
python "${PROJECT_ROOT}/6_downstream_evaluation/plot_figure8.py" \
    --baseline_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_baseline.csv" \
    --pfairft_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft.csv" \
    --pfairft_kl_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_pfairft_kl.csv" \
    --global_csv "${RESULTS_ROOT}/downstream_evaluation/discrim_global.csv" \
    --out_pdf "${RESULTS_ROOT}/downstream_evaluation/Figure8_Llama.pdf"

echo "Finished reproducing Table 2 values and Figure 8!"
echo "Check ${RESULTS_ROOT}/downstream_evaluation/ for all results and Figure8_Llama.pdf."
