#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
GPU="${GPU:-}"
DRY_RUN="${DRY_RUN:-1}"

RUN_HEAD_RESUME="${RUN_HEAD_RESUME:-1}"
RUN_HEAD_DISCRIM_ALL="${RUN_HEAD_DISCRIM_ALL:-1}"
RUN_HEAD_DISCRIM_TOPK="${RUN_HEAD_DISCRIM_TOPK:-1}"
RUN_HEAD_PLOTS="${RUN_HEAD_PLOTS:-1}"
RUN_ATTENTION="${RUN_ATTENTION:-1}"
RUN_HEAD_LOGIT="${RUN_HEAD_LOGIT:-1}"
RUN_MLP="${RUN_MLP:-1}"
RUN_ROUTER="${RUN_ROUTER:-1}"
RUN_PROMPT_HEAD="${RUN_PROMPT_HEAD:-1}"
RUN_INFERENCE_TIME="${RUN_INFERENCE_TIME:-1}"
RUN_FIGURE8="${RUN_FIGURE8:-1}"
RUN_A13="${RUN_A13:-1}"
RUN_RESULT_PLOTS="${RUN_RESULT_PLOTS:-1}"

RANDOM_REPEATS="${RANDOM_REPEATS:-1}"
RESUME_SAMPLE_SIZE="${RESUME_SAMPLE_SIZE:-100}"
HEAD_STEP="${HEAD_STEP:-5}"
HEAD_MAX="${HEAD_MAX:-25}"
HEAD_COUNTS="${HEAD_COUNTS:-}"
DISCRIM_HEAD_STEP="${DISCRIM_HEAD_STEP:-5}"
DISCRIM_HEAD_MAX="${DISCRIM_HEAD_MAX:-25}"
PROMPT_QID="${PROMPT_QID:-33}"
A13_QID="${A13_QID:-33}"

if [[ -z "$MODEL_NAME" || -z "$MODEL_PATH" || -z "$GPU" ]]; then
  cat <<'USAGE'
Usage:
  MODEL_NAME=JetMoE-8B-Chat \
  MODEL_PATH=/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat \
  MODEL_TYPE=jetmoe GPU=6,7 DRY_RUN=0 \
  bash scripts/run_moe_paper_suite.sh

GPU must contain only the allowed physical GPU indices: 6, 7, 6,7, or 7,6.
USAGE
  exit 2
fi

if [[ "$GPU" != "6" && "$GPU" != "7" && "$GPU" != "6,7" && "$GPU" != "7,6" ]]; then
  echo "Refusing to run: GPU must use only physical GPUs 6 and/or 7, got $GPU" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"

RESULTS_ROOT="${RESULTS_ROOT:-results/$MODEL_NAME}"
DATASET_JSON="${DATASET_JSON:-data/resume/qwen_summaries_with_race.json}"
DISCRIM_JSON="${DISCRIM_JSON:-data/discrim-eval/dataset_paired.json}"

RANKING_CSV="${RANKING_CSV:-$RESULTS_ROOT/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv}"
if [[ ! -f "$RANKING_CSV" && -f "$RESULTS_ROOT/biased_samples/biased_samples_ranking_current_prompt.csv" ]]; then
  RANKING_CSV="$RESULTS_ROOT/biased_samples/biased_samples_ranking_current_prompt.csv"
fi
HEADS_DIR="${HEADS_DIR:-$RESULTS_ROOT/sensitive_heads_moefreeze_top100_summary_only_current_ranking}"
SELECTED_HEADS="$HEADS_DIR/selected_heads_elbow.json"
HEAD_EMBEDDINGS="$HEADS_DIR/results.pkl"

ABLATION_ROOT="$RESULTS_ROOT/intervention_ablation"
HEAD_RESUME_ROOT="${HEAD_RESUME_ROOT:-$ABLATION_ROOT/head_resume_topk}"
PATTERN_ROOT="$RESULTS_ROOT/pattern_analysis"
MLP_ROOT="$RESULTS_ROOT/mlp_analysis"
INFERENCE_ROOT="$RESULTS_ROOT/inference_time"
LOG_ROOT="$RESULTS_ROOT/logs/paper_suite"
mkdir -p "$LOG_ROOT"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required standard-protocol artifact: $1" >&2
    exit 3
  fi
}

require_file "$RANKING_CSV"
require_file "$SELECTED_HEADS"
require_file "$HEAD_EMBEDDINGS"

head_count_args=(--max_head_count "$HEAD_MAX" --step "$HEAD_STEP")
if [[ -n "$HEAD_COUNTS" ]]; then
  IFS=',' read -r -a explicit_head_counts <<< "$HEAD_COUNTS"
  head_count_args=(--head_counts "${explicit_head_counts[@]}")
fi

if [[ "$RUN_HEAD_RESUME" == "1" ]]; then
  out="$HEAD_RESUME_ROOT/sensitive"
  run_cmd "$PY" 2_component_identification/evaluate_intervention_by_head_count.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --sample_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --sensitive_heads_dir "$HEADS_DIR" \
    --resume_prompt_mode summary_only \
    --batch_size 1 \
    --output_dir "$out" --intervention_type negative \
    "${head_count_args[@]}"

  for ((repeat=0; repeat<RANDOM_REPEATS; repeat++)); do
    seed=$((42 + repeat))
    out="$HEAD_RESUME_ROOT/random_seed_${seed}"
    run_cmd "$PY" 2_component_identification/evaluate_intervention_by_head_count_random.py \
      --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --dataset_json_path "$DATASET_JSON" --sample_csv_path "$RANKING_CSV" \
      --sample_size "$RESUME_SAMPLE_SIZE" --sensitive_heads_dir "$HEADS_DIR" \
      --resume_prompt_mode summary_only \
      --batch_size 1 \
      --output_dir "$out" --seed "$seed" \
      "${head_count_args[@]}"
  done
fi

if [[ "$RUN_HEAD_DISCRIM_ALL" == "1" ]]; then
  for mode in negative negative_random; do
    repeats=1
    [[ "$mode" == "negative_random" ]] && repeats="$RANDOM_REPEATS"
    for ((repeat=0; repeat<repeats; repeat++)); do
      seed=$((42 + repeat))
      out="$ABLATION_ROOT/head_discrim_all/${mode}_seed_${seed}"
      run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_discrim_eval.py \
        --dataset_path "$DISCRIM_JSON" --model_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" --prompt_type prompt \
        --sensitive_heads_dir "$HEADS_DIR" --intervention_mode "$mode" \
        --seed "$seed" --output_dir "$out" --csv_path "$out/per_sample.csv"
    done
  done
fi

if [[ "$RUN_HEAD_DISCRIM_TOPK" == "1" ]]; then
  for mode in negative negative_random; do
    repeats=1
    [[ "$mode" == "negative_random" ]] && repeats="$RANDOM_REPEATS"
    for ((repeat=0; repeat<repeats; repeat++)); do
      seed=$((42 + repeat))
      out="$ABLATION_ROOT/head_discrim_topk/${mode}_seed_${seed}"
      run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_discrim_eval_head_count.py \
        --dataset_path "$DISCRIM_JSON" --model_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" --sensitive_heads_dir "$HEADS_DIR" \
        --intervention_mode "$mode" --seed "$seed" --output_dir "$out" \
        --batch_size 1 \
        --results_csv_name results.csv --max_head_count "$DISCRIM_HEAD_MAX" \
        --step "$DISCRIM_HEAD_STEP"
    done
  done
fi

if [[ "$RUN_HEAD_PLOTS" == "1" ]]; then
  run_cmd "$PY" 4_intervention_ablation/head_intervention/aggregate_moe_head_ablation.py \
    --model_name "$MODEL_NAME" --ablation_root "$ABLATION_ROOT" \
    --output_dir "$ABLATION_ROOT/head_ablation_plots"
fi

if [[ "$RUN_ATTENTION" == "1" ]]; then
  for source in fixed resume_rank1; do
    out="$PATTERN_ROOT/attention_pattern/$source"
    run_cmd "$PY" 3_pattern_analysis/head_attention_pattern/analyze_qk_scores.py \
      --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --sensitive_heads_path "$SELECTED_HEADS" --sample_csv_path "$RANKING_CSV" \
      --dataset_json_path "$DATASET_JSON" --prompt_source "$source" --output_dir "$out"
    run_cmd "$PY" 3_pattern_analysis/head_attention_pattern/visualize_qk_scores.py \
      --qk_scores_json "$out/qk_scores_full.json" --model_path "$MODEL_PATH" \
      --output_dir "$out/plots"
  done
fi

if [[ "$RUN_HEAD_LOGIT" == "1" ]]; then
  out="$PATTERN_ROOT/head_logit_resume_top100"
  run_cmd "$PY" 3_pattern_analysis/head_logit_analysis/analyze_head_kl_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --biased_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --batch_size 1 --output_dir "$out"
fi

if [[ "$RUN_MLP" == "1" ]]; then
  identify="$MLP_ROOT/identification_top100"
  selected="$MLP_ROOT/selected_layers_top100"
  means="$MLP_ROOT/mlp_means_resume.pkl"
  exp20="$MLP_ROOT/exp20_residual_top100"
  run_cmd "$PY" 2_component_identification/analyze_race_sensitive_MLPs.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --output_dir "$identify" \
    --sample_csv_path "$RANKING_CSV" --sample_size "$RESUME_SAMPLE_SIZE" \
    --batch_size 1
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/select_race_sensitive_MLPs.py \
    --results_path "$identify/results_mlp.pkl" --output_dir "$selected"
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/collect_race_mean_MLPs_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --output_path "$means" \
    --sample_csv_path "$RANKING_CSV" --sample_size "$RESUME_SAMPLE_SIZE" \
    --resume_prompt_mode summary_only --batch_size 1
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/evaluate_intervention_MLP_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --sample_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --resume_prompt_mode summary_only \
    --sensitive_mlp_path "$selected/selected_mlp_layers_elbow.json" \
    --mlp_embeddings_path "$means" --output_dir "$ABLATION_ROOT/mlp_resume" \
    --csv_path "$ABLATION_ROOT/mlp_resume/per_sample.csv"
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/evaluate_intervention_MLP_discrim_eval.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_path "$DISCRIM_JSON" \
    --sensitive_mlp_path "$selected/selected_mlp_layers_elbow.json" \
    --mlp_embeddings_path "$means" --output_dir "$ABLATION_ROOT/mlp_discrim" \
    --csv_path "$ABLATION_ROOT/mlp_discrim/per_sample.csv"
  run_cmd "$PY" 3_pattern_analysis/mlp_analysis/analyze_mlp_output_kl_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --biased_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --batch_size 1 --output_dir "$exp20"
  run_cmd "$PY" 3_pattern_analysis/mlp_analysis/analyze_mlp_output_kl_resume_with_intervention.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" \
    --biased_csv_path "$RANKING_CSV" --sensitive_heads_path "$SELECTED_HEADS" \
    --embeddings_path "$HEAD_EMBEDDINGS" --sample_size "$RESUME_SAMPLE_SIZE" \
    --batch_size 1 --output_dir "$exp20"
  run_cmd "$PY" 3_pattern_analysis/head_logit_analysis/analyze_head_kl_resume_mlp.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --biased_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" \
    --mlp_selected_path "$selected/selected_mlp_layers_elbow.json" \
    --batch_size 1 \
    --output_dir "$PATTERN_ROOT/head_logit_with_mlp_intervention"
fi

if [[ "$RUN_ROUTER" == "1" ]]; then
  run_cmd "$PY" 3_pattern_analysis/mlp_analysis/analyze_moe_router_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --sample_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --sensitive_heads_path "$SELECTED_HEADS" \
    --embeddings_path "$HEAD_EMBEDDINGS" --batch_size 1 \
    --output_dir "$MLP_ROOT/router_top100"
fi

if [[ "$RUN_PROMPT_HEAD" == "1" ]]; then
  prompt_out="$PATTERN_ROOT/debiased_prompt_qid${PROMPT_QID}"
  run_cmd "$PY" 3_pattern_analysis/debiased_prompt_analysis/analyze_head_patterns.py \
    --model_path "$MODEL_PATH" --dataset_path "$DISCRIM_JSON" \
    --qid "$PROMPT_QID" --batch_size 1 --output_dir "$prompt_out"
  run_cmd "$PY" 3_pattern_analysis/debiased_prompt_analysis/plot_debiased_prompt_head_l2.py \
    --analysis_dir "$prompt_out" --selected_heads_json "$SELECTED_HEADS" \
    --out_path "$prompt_out/head_prompt_vs_debiased_l2.pdf"
fi

if [[ "$RUN_INFERENCE_TIME" == "1" ]]; then
  run_cmd "$PY" 4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_resume.py \
    --base_model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --biased_csv_path "$RANKING_CSV" \
    --sample_size "$RESUME_SAMPLE_SIZE" --resume_prompt_mode summary_only \
    --sensitive_heads_dir "$HEADS_DIR" --intervention_mode partial \
    --output_csv_path "$INFERENCE_ROOT/resume_partial.csv"
  run_cmd "$PY" 4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_discrim.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_path "$DISCRIM_JSON" \
    --sensitive_heads_dir "$HEADS_DIR" --intervention_mode partial \
    --output_dir "$INFERENCE_ROOT" --csv_path "$INFERENCE_ROOT/discrim_partial.csv"
  run_cmd "$PY" 4_intervention_ablation/projection_intervention/evaluate_mmlu_intervention.py \
    --model_path "$MODEL_PATH" --sensitive_heads_dir "$HEADS_DIR" \
    --intervention_mode partial --output_json "$INFERENCE_ROOT/mmlu_partial.json"
fi

if [[ "$RUN_FIGURE8" == "1" ]]; then
  downstream="$RESULTS_ROOT/downstream_evaluation"
  baseline="$downstream/discrim_baseline_resume_standard_fresh.csv"
  global_csv="$downstream/discrim_global_lora_ce_resume_standard_fresh.csv"
  pfairft_csv="$downstream/discrim_pfairft_ce_resume_standard_fresh.csv"
  pfairft_kl_csv="$downstream/discrim_pfairft_kl_resume_standard_fresh.csv"
  pfairft_kl_ce_csv="$downstream/discrim_pfairft_kl_ce_resume_standard_fresh.csv"
  debiased_csv="$downstream/discrim_baseline_debiased_prompt_paper_suite.csv"
  require_file "$baseline"
  require_file "$global_csv"
  require_file "$pfairft_csv"
  require_file "$pfairft_kl_csv"
  run_cmd "$PY" 6_downstream_evaluation/evaluate_models_discrim.py \
    --dataset_path "$DISCRIM_JSON" --base_model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" --mode baseline_debiased_prompt \
    --prompt_type debiased_prompt --csv_path "$debiased_csv" \
    --model_name_suffix baseline_debiased_prompt
  run_cmd "$PY" 6_downstream_evaluation/plot_figure8.py \
    --baseline_csv "$baseline" --global_csv "$global_csv" \
    --pfairft_csv "$pfairft_csv" --pfairft_kl_csv "$pfairft_kl_csv" \
    --pfairft_kl_ce_csv "$pfairft_kl_ce_csv" \
    --debiased_prompt_csv "$debiased_csv" \
    --inference_time_csv "$INFERENCE_ROOT/discrim_partial.csv" \
    --out_pdf "$downstream/Figure8_${MODEL_NAME}_paper_suite_extended.pdf" \
    --model_label "$MODEL_NAME"
fi

if [[ "$RUN_A13" == "1" ]]; then
  pfairft="${PFAIRFT_ADAPTER:-$RESULTS_ROOT/pfairft_fairness_ce_summary_only_current_ranking_full/final_model}"
  global="${GLOBAL_ADAPTER:-$RESULTS_ROOT/global_lora_ce_summary_only_current_ranking_full/final_model}"
  require_file "$pfairft/adapter_config.json"
  require_file "$global/adapter_config.json"
  out="$RESULTS_ROOT/downstream_head_analysis/qid${A13_QID}_pfairft_vs_global"
  run_cmd "$PY" 3_pattern_analysis/model_comparison/compare_adapter_head_fairness_gap.py \
    --base_model_path "$MODEL_PATH" --first_adapter "$pfairft" --second_adapter "$global" \
    --first_label PFairFT --second_label "Global LoRA CE" --dataset_path "$DISCRIM_JSON" \
    --qid "$A13_QID" --batch_size 1 --output_dir "$out"
  run_cmd "$PY" 3_pattern_analysis/model_comparison/plot_adapter_head_fairness_gap.py \
    --input_dir "$out" --sensitive_heads_json "$SELECTED_HEADS" \
    --output_path "$out/head_pfairft_vs_global_qid${A13_QID}.pdf" \
    --first_label PFairFT --second_label "Global LoRA CE"
fi

if [[ "$RUN_RESULT_PLOTS" == "1" ]]; then
  run_cmd "$PY" 3_pattern_analysis/plots/plot_moe_paper_results.py \
    --model_name "$MODEL_NAME" --results_root "$RESULTS_ROOT" \
    --selected_heads_json "$SELECTED_HEADS" \
    --output_dir "$RESULTS_ROOT/paper_plots"
fi

echo "Paper suite completed for $MODEL_NAME on physical GPU $GPU"
