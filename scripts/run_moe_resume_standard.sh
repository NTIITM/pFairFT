#!/usr/bin/env bash
set -euo pipefail

# Standard MOE resume-ranking protocol driver.
# Default is dry-run so the command plan is visible before expensive jobs start.

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"

MODEL_NAME="${MODEL_NAME:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_NAME}}"

DRY_RUN="${DRY_RUN:-1}"
RUN_ALL="${RUN_ALL:-1}"
RUN_RANKING="${RUN_RANKING:-$RUN_ALL}"
RUN_HEADS="${RUN_HEADS:-$RUN_ALL}"
RUN_TRAIN="${RUN_TRAIN:-$RUN_ALL}"
RUN_RESUME_EVAL="${RUN_RESUME_EVAL:-$RUN_ALL}"
RUN_DISCRIM_EVAL="${RUN_DISCRIM_EVAL:-$RUN_ALL}"
RUN_MMLU="${RUN_MMLU:-$RUN_ALL}"
RUN_PLOTS="${RUN_PLOTS:-$RUN_ALL}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export CUDA_VISIBLE_DEVICES
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_JSON="${DATASET_JSON:-$PROJECT_ROOT/data/resume/qwen_summaries_with_race.json}"
DISCRIM_JSON="${DISCRIM_JSON:-$PROJECT_ROOT/data/discrim-eval/dataset_paired.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/$MODEL_NAME}"
DOWNSTREAM_DIR="$RESULTS_ROOT/downstream_evaluation"

RANKING_CSV="$RESULTS_ROOT/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv"
HEADS_DIR="$RESULTS_ROOT/sensitive_heads_moefreeze_top100_summary_only_current_ranking"

GLOBAL_DIR="$RESULTS_ROOT/global_lora_ce_summary_only_current_ranking_full"
PFAIRFT_DIR="$RESULTS_ROOT/pfairft_fairness_ce_summary_only_current_ranking_full"
PFAIRFT_KL_DIR="$RESULTS_ROOT/pfairft_fairness_kl_summary_only_current_ranking_full"
PFAIRFT_KL_CE_DIR="$RESULTS_ROOT/pfairft_fairness_kl_ce_summary_only_current_ranking_full"

RESUME_BASE="$DOWNSTREAM_DIR/resume_baseline_top100_summary_only_current_ranking_fresh.csv"
RESUME_GLOBAL="$DOWNSTREAM_DIR/resume_global_lora_ce_summary_only_current_ranking_full_fresh.csv"
RESUME_PFAIRFT="$DOWNSTREAM_DIR/resume_pfairft_summary_only_current_ranking_full_fresh.csv"
RESUME_PFAIRFT_KL="$DOWNSTREAM_DIR/resume_pfairft_kl_summary_only_current_ranking_full_fresh.csv"
RESUME_PFAIRFT_KL_CE="$DOWNSTREAM_DIR/resume_pfairft_kl_ce_summary_only_current_ranking_full_fresh.csv"

DISCRIM_BASE="$DOWNSTREAM_DIR/discrim_baseline_resume_standard_fresh.csv"
DISCRIM_GLOBAL="$DOWNSTREAM_DIR/discrim_global_lora_ce_resume_standard_fresh.csv"
DISCRIM_PFAIRFT="$DOWNSTREAM_DIR/discrim_pfairft_resume_standard_fresh.csv"
DISCRIM_PFAIRFT_KL="$DOWNSTREAM_DIR/discrim_pfairft_kl_resume_standard_fresh.csv"
DISCRIM_PFAIRFT_KL_CE="$DOWNSTREAM_DIR/discrim_pfairft_kl_ce_resume_standard_fresh.csv"

MMLU_BASE="$DOWNSTREAM_DIR/mmlu_ce_baseline_resume_standard_fresh.json"
MMLU_GLOBAL="$DOWNSTREAM_DIR/mmlu_ce_global_lora_ce_resume_standard_fresh.json"
MMLU_PFAIRFT="$DOWNSTREAM_DIR/mmlu_ce_pfairft_resume_standard_fresh.json"
MMLU_PFAIRFT_KL="$DOWNSTREAM_DIR/mmlu_ce_pfairft_kl_resume_standard_fresh.json"
MMLU_PFAIRFT_KL_CE="$DOWNSTREAM_DIR/mmlu_ce_pfairft_kl_ce_resume_standard_fresh.json"

RESUME_FIG_PDF="$DOWNSTREAM_DIR/Figure8_${MODEL_NAME}_resume_standard_fresh_klce.pdf"
RESUME_FIG_PNG="$DOWNSTREAM_DIR/Figure8_${MODEL_NAME}_resume_standard_fresh_klce.png"
DISCRIM_FIG_PDF="$DOWNSTREAM_DIR/Figure8_${MODEL_NAME}_discrim_resume_standard_fresh_klce.pdf"

if [[ -z "$MODEL_NAME" || -z "$MODEL_PATH" ]]; then
  cat <<'USAGE'
Usage:
  MODEL_NAME=JetMoE-8B-Chat \
  MODEL_PATH=/mnt/nfs/models/JetMoE-8B-Chat \
  MODEL_TYPE=jetmoe \
  DRY_RUN=0 \
  bash scripts/run_moe_resume_standard.sh

Defaults:
  DRY_RUN=1 prints commands without running them.
  RUN_ALL=1 enables every phase; override RUN_RANKING/RUN_HEADS/RUN_TRAIN/
  RUN_RESUME_EVAL/RUN_DISCRIM_EVAL/RUN_MMLU/RUN_PLOTS to select phases.

Branches:
  baseline
  Global LoRA CE
  PFairFT        = precise selected heads + affine fairness + CE
  PFairFT-KL     = precise selected heads + affine fairness + KL
  PFairFT-KL-CE  = precise selected heads + affine fairness + KL + CE
USAGE
  exit 2
fi

cd "$PROJECT_ROOT"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

fresh_file() {
  if [[ "$DRY_RUN" == "0" ]]; then
    rm -f "$1"
  else
    printf '+ rm -f %q\n' "$1"
  fi
}

train_pfairft() {
  local loss_type="$1"
  local out_dir="$2"
  run_cmd "$PY" 5_finetuning/finetune_precision_fairness.py \
    --model_path "$MODEL_PATH" \
    --dataset_json_path "$DATASET_JSON" \
    --heads_analysis_dir "$HEADS_DIR" \
    --output_dir "$out_dir" \
    --sample_csv_path "$RANKING_CSV" \
    --sample_size 0 \
    --max_samples 0 \
    --no-balanced \
    --resume_prompt_mode summary_only \
    --loss_type "$loss_type" \
    --num_epochs "${NUM_EPOCHS:-1}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
    --learning_rate "${LR:-2e-5}" \
    --warmup_steps "${WARMUP_STEPS:-0}" \
    ${TRAIN_PRECISION_FLAG:-}
}

eval_resume() {
  local mode="$1"
  local adapter_path="$2"
  local out_csv="$3"
  fresh_file "$out_csv"
  local args=(
    "$PY" 6_downstream_evaluation/evaluate_resume_fairness_top100.py
    --mode "$mode"
    --base_model_path "$MODEL_PATH"
    --dataset_json_path "$DATASET_JSON"
    --biased_csv_path "$RANKING_CSV"
    --sample_size "${RESUME_EVAL_SAMPLE_SIZE:-100}"
    --output_csv_path "$out_csv"
    --model_type "$MODEL_TYPE"
    --resume_prompt_mode summary_only
  )
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter_path "$adapter_path")
  fi
  run_cmd "${args[@]}"
}

eval_discrim() {
  local mode="$1"
  local suffix="$2"
  local adapter_path="$3"
  local out_csv="$4"
  fresh_file "$out_csv"
  local args=(
    "$PY" 6_downstream_evaluation/evaluate_models_discrim.py
    --dataset_path "$DISCRIM_JSON"
    --base_model_path "$MODEL_PATH"
    --mode "$mode"
    --model_type "$MODEL_TYPE"
    --csv_path "$out_csv"
    --model_name_suffix "$suffix"
  )
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter_path "$adapter_path")
  fi
  run_cmd "${args[@]}"
}

eval_mmlu() {
  local adapter_path="$1"
  local out_json="$2"
  fresh_file "$out_json"
  local args=(
    "$PY" 6_downstream_evaluation/evaluate_mmlu_ce.py
    --model_path "$MODEL_PATH"
    --out_json "$out_json"
    --split "${MMLU_SPLIT:-validation}"
    --max_samples "${MMLU_MAX_SAMPLES:-0}"
  )
  if [[ -n "$adapter_path" ]]; then
    args+=(--adapter_path "$adapter_path")
  fi
  run_cmd "${args[@]}"
}

if [[ "$RUN_RANKING" == "1" ]]; then
  run_cmd "$PY" 2_component_identification/evaluate_biased_sample.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" \
    --output_csv_path "$RANKING_CSV" \
    --resume_prompt_mode summary_only
fi

if [[ "$RUN_HEADS" == "1" ]]; then
  run_cmd "$PY" 2_component_identification/analyze_race_sensitive_heads_moefreeze.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" \
    --output_dir "$HEADS_DIR" \
    --sample_csv_path "$RANKING_CSV" \
    --sample_size 100 \
    --no-balanced \
    --resume_prompt_mode summary_only \
    --batch_size "${HEAD_BATCH_SIZE:-8}"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  run_cmd "$PY" 5_finetuning/finetune_global_lora.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" \
    --output_dir "$GLOBAL_DIR" \
    --sample_csv_path "$RANKING_CSV" \
    --sample_size 0 \
    --max_samples 0 \
    --no-balanced \
    --num_epochs "${NUM_EPOCHS:-1}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
    --learning_rate "${LR:-2e-5}" \
    --warmup_steps "${WARMUP_STEPS:-0}" \
    ${TRAIN_PRECISION_FLAG:-}
  train_pfairft fairness_ce "$PFAIRFT_DIR"
  train_pfairft fairness_kl "$PFAIRFT_KL_DIR"
  train_pfairft fairness_kl_ce "$PFAIRFT_KL_CE_DIR"
fi

if [[ "$RUN_RESUME_EVAL" == "1" ]]; then
  eval_resume baseline "" "$RESUME_BASE"
  eval_resume global_lora_ce "$GLOBAL_DIR/final_model" "$RESUME_GLOBAL"
  eval_resume pfairft "$PFAIRFT_DIR/final_model" "$RESUME_PFAIRFT"
  eval_resume pfairft_kl "$PFAIRFT_KL_DIR/final_model" "$RESUME_PFAIRFT_KL"
  eval_resume pfairft_kl_ce "$PFAIRFT_KL_CE_DIR/final_model" "$RESUME_PFAIRFT_KL_CE"
fi

if [[ "$RUN_DISCRIM_EVAL" == "1" ]]; then
  eval_discrim baseline baseline "" "$DISCRIM_BASE"
  eval_discrim global_lora_ce global_lora_ce_resume_standard "$GLOBAL_DIR/final_model" "$DISCRIM_GLOBAL"
  eval_discrim pfairft pfairft_resume_standard "$PFAIRFT_DIR/final_model" "$DISCRIM_PFAIRFT"
  eval_discrim pfairft_kl pfairft_kl_resume_standard "$PFAIRFT_KL_DIR/final_model" "$DISCRIM_PFAIRFT_KL"
  eval_discrim pfairft_kl_ce pfairft_kl_ce_resume_standard "$PFAIRFT_KL_CE_DIR/final_model" "$DISCRIM_PFAIRFT_KL_CE"
fi

if [[ "$RUN_MMLU" == "1" ]]; then
  eval_mmlu "" "$MMLU_BASE"
  eval_mmlu "$GLOBAL_DIR/final_model" "$MMLU_GLOBAL"
  eval_mmlu "$PFAIRFT_DIR/final_model" "$MMLU_PFAIRFT"
  eval_mmlu "$PFAIRFT_KL_DIR/final_model" "$MMLU_PFAIRFT_KL"
  eval_mmlu "$PFAIRFT_KL_CE_DIR/final_model" "$MMLU_PFAIRFT_KL_CE"
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
  fresh_file "$RESUME_FIG_PDF"
  fresh_file "$RESUME_FIG_PNG"
  run_cmd "$PY" 6_downstream_evaluation/plot_resume_figure8.py \
    --baseline_csv "$RESUME_BASE" \
    --global_csv "$RESUME_GLOBAL" \
    --pfairft_csv "$RESUME_PFAIRFT" \
    --pfairft_kl_csv "$RESUME_PFAIRFT_KL" \
    --pfairft_kl_ce_csv "$RESUME_PFAIRFT_KL_CE" \
    --out_pdf "$RESUME_FIG_PDF" \
    --out_png "$RESUME_FIG_PNG" \
    --model_label "$MODEL_LABEL" \
    --pfairft_label "PFairFT"

  fresh_file "$DISCRIM_FIG_PDF"
  run_cmd "$PY" 6_downstream_evaluation/plot_figure8.py \
    --baseline_csv "$DISCRIM_BASE" \
    --global_csv "$DISCRIM_GLOBAL" \
    --pfairft_csv "$DISCRIM_PFAIRFT" \
    --pfairft_kl_csv "$DISCRIM_PFAIRFT_KL" \
    --pfairft_kl_ce_csv "$DISCRIM_PFAIRFT_KL_CE" \
    --out_pdf "$DISCRIM_FIG_PDF" \
    --model_label "$MODEL_LABEL"
fi

