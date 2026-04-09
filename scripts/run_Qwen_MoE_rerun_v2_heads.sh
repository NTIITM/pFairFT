#!/usr/bin/env bash
# ============================================================
# Qwen1.5-MoE-A2.7B-Chat: Re-run with NEW head selection (v2)
# Uses the acceleration (2nd derivative) elbow method: 5 heads
# ============================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export HF_DATASETS_OFFLINE=1

MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen1.5-MoE-A2.7B-Chat"
MODEL_NAME="Qwen1.5-MoE-A2.7B-Chat"
PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
RESULTS_ROOT="${PROJECT_ROOT}/results/${MODEL_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"

JSON_PATH="${DATA_DIR}/resume/qwen_summaries_with_race.json"
BIASED_DIR="${RESULTS_ROOT}/biased_samples"
HEADS_DIR="${RESULTS_ROOT}/sensitive_heads"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

# ================================================================
# Step 0: Backup old results
# ================================================================
BACKUP_DIR="${RESULTS_ROOT}/_backup_$(date +%Y%m%d_%H%M%S)"
echo "============================================"
echo "Backing up old results to ${BACKUP_DIR}..."
mkdir -p "${BACKUP_DIR}"

for d in pfairft precision_fairness pfairft_ce global downstream_evaluation; do
    if [ -d "${RESULTS_ROOT}/${d}" ]; then
        echo "  Backing up ${d}..."
        cp -r "${RESULTS_ROOT}/${d}" "${BACKUP_DIR}/${d}"
    fi
done

# Also backup the run logs
for f in run_log.txt run_log_ce.txt; do
    if [ -f "${RESULTS_ROOT}/${f}" ]; then
        cp "${RESULTS_ROOT}/${f}" "${BACKUP_DIR}/${f}"
    fi
done

# Backup old selected_heads_elbow.json
cp "${HEADS_DIR}/selected_heads_elbow.json" "${BACKUP_DIR}/selected_heads_elbow_original.json"
echo "Backup complete."

# ================================================================
# Step 1: Update results.pkl with new head selection (v2)
# ================================================================
echo "============================================"
echo "Updating results.pkl with v2 head selection (acceleration method, 5 heads)..."
python -c "
import pickle, json
import numpy as np

pkl_path = '${HEADS_DIR}/results.pkl'
v2_json = '${HEADS_DIR}/selected_heads_elbow_v2.json'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

with open(v2_json, 'r') as f:
    new_heads = json.load(f)

print(f'Old selected heads: {len(data[\"selected_heads\"])}')
data['selected_heads'] = new_heads

# Recompute elbow values
flat_kl = data['heatmap'].flatten()
valid = flat_kl[np.isfinite(flat_kl)]
sorted_scores = np.sort(valid)[::-1]

# Find the score of the least-significant selected head
min_score = min(data['heatmap'][h['layer'], h['head']] for h in new_heads)
data['elbow_score'] = float(min_score)

# Find elbow_idx (rank - 1)
elbow_idx = np.searchsorted(-sorted_scores, -min_score)
data['elbow_idx'] = int(elbow_idx)
data['elbow_rank'] = int(elbow_idx) + 1
data['elbow_kl_value'] = float(min_score)

print(f'New selected heads: {len(new_heads)}')
print(f'New elbow_score: {data[\"elbow_score\"]:.6f}')
print(f'New elbow_rank: {data[\"elbow_rank\"]}')

with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)
print('Updated results.pkl successfully.')
"

# Also update selected_heads_elbow.json
cp "${HEADS_DIR}/selected_heads_elbow_v2.json" "${HEADS_DIR}/selected_heads_elbow.json"
echo "Updated selected_heads_elbow.json."

EVAL_DIR="${RESULTS_ROOT}/downstream_evaluation"
mkdir -p "${EVAL_DIR}"

# ================================================================
# PFairFT-KL (KL divergence on sensitive heads)
# ================================================================
PFAIRFT_KL_DIR="${RESULTS_ROOT}/precision_fairness"
echo "============================================"
echo "Fine-tuning PFairFT-KL (v2: 5 heads)..."
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
echo "Fine-tuning PFairFT (v2: 5 heads)..."
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
# Global (LoRA + CE) — does NOT depend on head selection, skip if exists
# ================================================================
GLOBAL_DIR="${RESULTS_ROOT}/global"
if [ -d "${GLOBAL_DIR}/final_model" ]; then
    echo "============================================"
    echo "Global fine-tuning already exists, skipping (no head dependency)."
else
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
fi

# ================================================================
# PFairFT-CE (Combined KL + CE on sensitive heads)
# ================================================================
PFAIRFT_KL_CE_DIR="${RESULTS_ROOT}/pfairft_ce"
echo "============================================"
echo "Fine-tuning PFairFT-CE (v2: 5 heads)..."
python "${PROJECT_ROOT}/5_finetuning/finetune_precision_fairness.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_json_path "${JSON_PATH}" \
    --heads_analysis_dir "${HEADS_DIR}" \
    --output_dir "${PFAIRFT_KL_CE_DIR}" \
    --sample_csv_path "${CSV_PATH}" \
    --sample_size 1000 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --loss_type fairness_ce --fairness_lambda 0.1 --seed 42

echo "Evaluating PFairFT-CE Resume DP..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_resume_fairness_top100.py" \
    --mode pfairft_ce \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --biased_csv_path "${CSV_PATH}" \
    --output_csv_path "${EVAL_DIR}/resume_pfairft_ce.csv" \
    --device "cuda" --model_type "qwen"

echo "Evaluating PFairFT-CE MMLU LM-CE..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_mmlu_ce.py" \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --out_json "${EVAL_DIR}/mmlu_ce_pfairft_ce.json"

echo "Evaluating PFairFT-CE Discrim-Eval..."
python "${PROJECT_ROOT}/6_downstream_evaluation/evaluate_models_discrim.py" \
    --mode pfairft_ce \
    --base_model_path "${MODEL_PATH}" \
    --adapter_path "${PFAIRFT_KL_CE_DIR}/final_model" \
    --csv_path "${EVAL_DIR}/discrim_pfairft_ce.csv" \
    --device "cuda" --model_type "qwen"

# ================================================================
# Generate Figure 8
# ================================================================
echo "============================================"
echo "Generating Figure 8..."
python "${PROJECT_ROOT}/6_downstream_evaluation/plot_figure8.py" \
    --baseline_csv "${EVAL_DIR}/discrim_baseline.csv" \
    --pfairft_csv "${EVAL_DIR}/discrim_pfairft.csv" \
    --pfairft_kl_csv "${EVAL_DIR}/discrim_pfairft_kl.csv" \
    --global_csv "${EVAL_DIR}/discrim_global.csv" \
    --pfairft_ce_csv "${EVAL_DIR}/discrim_pfairft_ce.csv" \
    --out_pdf "${EVAL_DIR}/Figure8_Qwen_MoE_with_KLCE.pdf" \
    --model_label "Qwen1.5-MoE 2.7B"

echo "============================================"
echo "All done! Results at: ${EVAL_DIR}"
echo "Backup of old results at: ${BACKUP_DIR}"
echo "============================================"
