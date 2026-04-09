#!/usr/bin/env bash
# ============================================================
# Qwen1.5-MoE-A2.7B-Chat: Reproduce Table 2 & Figure 8
# Metrics: Resume DP, MMLU LM-CE, Discrim DP
# Fine-tuning: PFairFT, PFairFT-KL, Global
# ============================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6,7
export HF_DATASETS_OFFLINE=1

MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_NAME="Qwen1.5-MoE-A2.7B-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"
EVAL_DIR="${RESULTS_ROOT}/downstream_evaluation"

mkdir -p "${EVAL_DIR}"

JSON_PATH="${DATA_DIR}/resume/qwen_summaries_with_race.json"
BIASED_DIR="${RESULTS_ROOT}/biased_samples"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

# ================================================================
# Baseline Evaluation (no fine-tuning)
# ================================================================
echo "============================================"
# echo "Evaluating Baseline Resume DP..."
# python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
#     --mode baseline \
#     --base_model_path "${MODEL_PATH}" \
#     --biased_csv_path "${CSV_PATH}" \
#     --output_csv_path "${EVAL_DIR}/resume_baseline.csv" \
#     --device "cuda" --model_type "qwen"

# echo "Evaluating Baseline MMLU LM-CE..."
# python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
#     --model_path "${MODEL_PATH}" \
#     --out_json "${EVAL_DIR}/mmlu_ce_baseline.json"

# echo "Evaluating Baseline Discrim-Eval..."
# python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
#     --mode baseline \
#     --base_model_path "${MODEL_PATH}" \
#     --csv_path "${EVAL_DIR}/discrim_baseline.csv" \
#     --device "cuda" --model_type "qwen"

# ================================================================
# PFairFT-KL (KL divergence on sensitive heads)
# ================================================================
PFAIRFT_KL_DIR="${RESULTS_ROOT}/precision_fairness"
echo "============================================"
echo "Fine-tuning PFairFT-KL..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_KL_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --loss_type kl --fairness_lambda 0.1 --seed 42

echo "Evaluating PFairFT-KL Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft_kl \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${EVAL_DIR}/resume_pfairft_kl.csv" \
    --device "cuda" --model_type "qwen"

echo "Evaluating PFairFT-KL MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_DIR}/final_model" \
    --out_json "${EVAL_DIR}/mmlu_ce_pfairft_kl.json"

echo "Evaluating PFairFT-KL Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft_kl \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_DIR}/final_model" \
    --csv_path "${EVAL_DIR}/discrim_pfairft_kl.csv" \
    --device "cuda" --model_type "qwen"

# ================================================================
# PFairFT (MSE/affine projection on sensitive heads)
# ================================================================
PFAIRFT_DIR="${RESULTS_ROOT}/pfairft"
echo "============================================"
echo "Fine-tuning PFairFT..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --loss_type fairness --fairness_lambda 0.1 --seed 42

echo "Evaluating PFairFT Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${EVAL_DIR}/resume_pfairft.csv" \
    --device "cuda" --model_type "qwen"

echo "Evaluating PFairFT MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --out_json "${EVAL_DIR}/mmlu_ce_pfairft.json"

echo "Evaluating PFairFT Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_DIR}/final_model" \
    --csv_path "${EVAL_DIR}/discrim_pfairft.csv" \
    --device "cuda" --model_type "qwen"

# ================================================================
# Global (LoRA + CE)
# ================================================================
GLOBAL_DIR="${RESULTS_ROOT}/global"
echo "============================================"
echo "Fine-tuning Global..."
python "${PROJECT_ROOT}/5_finetuning/finetune_global_lora.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --output_dir "${GLOBAL_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --train_type lora \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --seed 42

echo "Evaluating Global Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode global \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${EVAL_DIR}/resume_global.csv" \
    --device "cuda" --model_type "qwen"

echo "Evaluating Global MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --out_json "${EVAL_DIR}/mmlu_ce_global.json"

echo "Evaluating Global Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode global \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${GLOBAL_DIR}/final_model" \
    --csv_path "${EVAL_DIR}/discrim_global.csv" \
    --device "cuda" --model_type "qwen"

# ================================================================
# PFairFT-KL-CE (Combined KL + CE on sensitive heads)
# ================================================================
PFAIRFT_KL_CE_DIR="${RESULTS_ROOT}/pfairft_kl_ce"
echo "============================================"
echo "Fine-tuning PFairFT-KL-CE..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_KL_CE_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --loss_type kl_ce --fairness_lambda 0.1 --seed 42

echo "Evaluating PFairFT-KL-CE Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft_kl_ce \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${EVAL_DIR}/resume_pfairft_kl_ce.csv" \
    --device "cuda" --model_type "qwen"

echo "Evaluating PFairFT-KL-CE MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --out_json "${EVAL_DIR}/mmlu_ce_pfairft_kl_ce.json"

echo "Evaluating PFairFT-KL-CE Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft_kl_ce \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --csv_path "${EVAL_DIR}/discrim_pfairft_kl_ce.csv" \
    --device "cuda" --model_type "qwen"

# ================================================================
# Generate Figure 8 (including the new variant)
# ================================================================
echo "============================================"
echo "Generating Figure 8..."
python "${PROJECT_ROOT}/6_downstream_evaluation/plot_figure8.py" \
    --baseline_csv "${EVAL_DIR}/discrim_baseline.csv" \
    --pfairft_csv "${EVAL_DIR}/discrim_pfairft.csv" \
    --pfairft_kl_csv "${EVAL_DIR}/discrim_pfairft_kl.csv" \
    --global_csv "${EVAL_DIR}/discrim_global.csv" \
    --pfairft_kl_ce_csv "${EVAL_DIR}/discrim_pfairft_kl_ce.csv" \
    --out_pdf "${EVAL_DIR}/Figure8_Qwen_MoE_with_KLCE.pdf" \
    --model_label "Qwen1.5-MoE 2.7B"

echo "All done! Results at: ${EVAL_DIR}"
