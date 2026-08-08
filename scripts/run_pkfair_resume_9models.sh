#!/usr/bin/env bash
set -euo pipefail

# Reproduce the nine-model full-resume Global/PKFair comparison.
# This driver is dry-run by default. It only removes explicitly allowlisted
# comparison adapters when CLEAN_OLD_CHECKPOINTS=1.

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
DATASET_JSON="${DATASET_JSON:-$PROJECT_ROOT/data/resume/qwen_summaries_with_race.json}"
DISCRIM_JSON="${DISCRIM_JSON:-$PROJECT_ROOT/data/discrim-eval/dataset_paired.json}"
DRY_RUN="${DRY_RUN:-1}"
CLEAN_OLD_CHECKPOINTS="${CLEAN_OLD_CHECKPOINTS:-1}"
RUN_UPSTREAM="${RUN_UPSTREAM:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"
MODEL_FILTER="${MODEL_FILTER:-}"
GPU_OVERRIDE="${GPU_OVERRIDE:-}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODELS=(
  DeepSeek-V2-Lite-Chat
  JetMoE-8B-Chat
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Meta-Llama-3-8B-Instruct
  OLMoE-1B-7B-0924-Instruct
  Qwen3-1.7B
  Qwen3-4B
  Qwen3-8B
)

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

fresh_file() {
  local path="$1"
  if [[ "$DRY_RUN" == "0" ]]; then
    rm -f -- "$path"
  else
    printf '+ rm -f %q\n' "$path"
  fi
}

remove_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    return
  fi
  if [[ "$DRY_RUN" == "0" ]]; then
    rm -rf -- "$path"
  else
    printf '+ rm -rf %q\n' "$path"
  fi
}

configure_model() {
  local name="$1"
  TRAIN_MEMORY_ARGS=()
  case "$name" in
    DeepSeek-V2-Lite-Chat)
      MODEL_PATH=/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat
      MODEL_TYPE=deepseek; GPU_SET=6,7; BATCH_SIZE=1; GRAD_ACCUM=16 ;;
    JetMoE-8B-Chat)
      MODEL_PATH=/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat
      MODEL_TYPE=jetmoe; GPU_SET=6,7; BATCH_SIZE=1; GRAD_ACCUM=16 ;;
    Llama-3.2-1B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct
      MODEL_TYPE=llama; GPU_SET=6; BATCH_SIZE=4; GRAD_ACCUM=4 ;;
    Llama-3.2-3B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct
      MODEL_TYPE=llama; GPU_SET=6; BATCH_SIZE=2; GRAD_ACCUM=8 ;;
    Meta-Llama-3-8B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct
      MODEL_TYPE=llama; GPU_SET=6,7; BATCH_SIZE=1; GRAD_ACCUM=16 ;;
    OLMoE-1B-7B-0924-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/allenai/OLMoE-1B-7B-0924-Instruct
      MODEL_TYPE=olmoe; GPU_SET=6; BATCH_SIZE=2; GRAD_ACCUM=8 ;;
    Qwen3-1.7B)
      MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-1.7B
      MODEL_TYPE=qwen; GPU_SET=6; BATCH_SIZE=4; GRAD_ACCUM=4
      TRAIN_MEMORY_ARGS=(--gradient_checkpointing) ;;
    Qwen3-4B)
      MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-4B
      MODEL_TYPE=qwen; GPU_SET=6; BATCH_SIZE=2; GRAD_ACCUM=8 ;;
    Qwen3-8B)
      MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-8B
      MODEL_TYPE=qwen; GPU_SET=6,7; BATCH_SIZE=1; GRAD_ACCUM=16 ;;
    *) echo "Unsupported model: $name" >&2; return 2 ;;
  esac
}

is_current_heads_valid() {
  local heads_dir="$1"; local ranking="$2"; local model_type="$3"
  local dataset_rel ranking_rel
  dataset_rel="$(realpath -m --relative-to="$PROJECT_ROOT" "$DATASET_JSON")"
  ranking_rel="$(realpath -m --relative-to="$PROJECT_ROOT" "$ranking")"
  [[ -f "$heads_dir/results.pkl" && -f "$heads_dir/selected_heads_elbow.json" && -f "$heads_dir/metadata.json" ]] || return 1
  jq -e --arg dataset "$DATASET_JSON" --arg dataset_rel "$dataset_rel" \
    --arg ranking "$ranking" --arg ranking_rel "$ranking_rel" --arg type "$model_type" \
    '(.dataset_kind == "resume") and
     ((.dataset_json_path == $dataset) or (.dataset_json_path == $dataset_rel)) and
     ((.sample_csv_path == $ranking) or (.sample_csv_path == $ranking_rel)) and
     (.sample_size == 100) and
     (.resume_prompt_mode == "summary_only") and (.model_type == $type)' \
    "$heads_dir/metadata.json" >/dev/null
}

ensure_upstream() {
  local name="$1"; local root="$PROJECT_ROOT/results/$name"
  local ranking="$root/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv"
  local heads="$root/sensitive_heads_moefreeze_top100_summary_only_current_ranking"

  if [[ ! -s "$ranking" ]]; then
    run_cmd "$PY" 2_component_identification/evaluate_biased_sample.py \
      --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --dataset_json_path "$DATASET_JSON" --output_csv_path "$ranking" \
      --resume_prompt_mode summary_only
  else
    echo "[$name] Reusing validated/current ranking: $ranking"
  fi

  if is_current_heads_valid "$heads" "$ranking" "$MODEL_TYPE"; then
    echo "[$name] Reusing validated/current top100 heads: $heads"
  else
    run_cmd "$PY" 2_component_identification/analyze_race_sensitive_heads_moefreeze.py \
      --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --dataset_json_path "$DATASET_JSON" --output_dir "$heads" \
      --sample_csv_path "$ranking" --sample_size 100 --no-balanced \
      --resume_prompt_mode summary_only --batch_size 8
  fi
}

cleanup_comparison_checkpoints() {
  local root="$1"; local name="$2"
  local old_dirs=(
    "$root/global_lora_ce_summary_only_current_ranking_full"
    "$root/pfairft_fairness_ce_summary_only_current_ranking_full"
    "$root/pfairft_fairness_kl_summary_only_current_ranking_full"
    "$root/pfairft_fairness_kl_ce_summary_only_current_ranking_full"
    "$root/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch"
    "$root/pkfair_fairness_ce_yesno_summary_only_current_ranking_full_3epoch"
    "$root/pkfair_fairness_kl_yesno_summary_only_current_ranking_full_3epoch"
    "$root/pkfair_fairness_kl_ce_yesno_summary_only_current_ranking_full_3epoch"
    "$root/pfairft_fairness_ce_summary_only_full"
  )
  # These are the four legacy comparison directories in the older 3B run.
  if [[ "$name" == "Llama-3.2-3B-Instruct" ]]; then
    old_dirs+=("$root/global" "$root/pfairft" "$root/pfairft_ce" "$root/precision_fairness")
  fi
  for path in "${old_dirs[@]}"; do
    remove_dir "$path"
  done
}

branch_is_done() {
  local dir="$1"
  [[ "$RESUME_COMPLETED" == "1" && -f "$dir/training_timing.json" && \
     -f "$dir/final_model/adapter_model.safetensors" ]]
}

train_global() {
  local root="$1"; local ranking="$2"; local out="$root/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch"
  if branch_is_done "$out"; then echo "[$MODEL_NAME] Global already complete: $out"; return; fi
  run_cmd "$PY" 5_finetuning/finetune_global_lora.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$DATASET_JSON" --output_dir "$out" \
    --sample_csv_path "$ranking" --sample_size 0 --max_samples 0 --no-balanced \
    --resume_prompt_mode summary_only --num_epochs 3 --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --learning_rate 2e-5 \
    --warmup_steps 0 "${TRAIN_MEMORY_ARGS[@]}"
}

train_pkfair() {
  local root="$1"; local ranking="$2"; local heads="$3"; local loss_type="$4"; local out_name="$5"
  local out="$root/$out_name"
  if branch_is_done "$out"; then echo "[$MODEL_NAME] $loss_type already complete: $out"; return; fi
  run_cmd "$PY" 5_finetuning/finetune_precision_fairness.py \
    --model_path "$MODEL_PATH" \
    --dataset_json_path "$DATASET_JSON" --heads_analysis_dir "$heads" \
    --output_dir "$out" --sample_csv_path "$ranking" --sample_size 0 --max_samples 0 \
    --no-balanced --resume_prompt_mode summary_only --loss_type "$loss_type" \
    --num_epochs 3 --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 2e-5 --warmup_steps 0 "${TRAIN_MEMORY_ARGS[@]}"
}

evaluate_and_plot() {
  local root="$1"; local ranking="$2"; local downstream="$root/downstream_evaluation"
  local global="$root/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
  local pkce="$root/pkfair_fairness_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
  local pkkl="$root/pkfair_fairness_kl_yesno_summary_only_current_ranking_full_3epoch/final_model"
  local pk="$root/pkfair_fairness_kl_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
  local base_resume="$downstream/resume_baseline_top100_summary_only_pkfair_3epoch_fresh.csv"
  local global_resume="$downstream/resume_global_yesno_pkfair_3epoch_fresh.csv"
  local pk_resume="$downstream/resume_pkfair_pkfair_3epoch_fresh.csv"
  local pkkl_resume="$downstream/resume_pkfair_kl_pkfair_3epoch_fresh.csv"
  local pkce_resume="$downstream/resume_pkfair_ce_pkfair_3epoch_fresh.csv"
  local base_discrim="$downstream/discrim_baseline_pkfair_3epoch_fresh.csv"
  local global_discrim="$downstream/discrim_global_yesno_pkfair_3epoch_fresh.csv"
  local pk_discrim="$downstream/discrim_pkfair_pkfair_3epoch_fresh.csv"
  local pkkl_discrim="$downstream/discrim_pkfair_kl_pkfair_3epoch_fresh.csv"
  local pkce_discrim="$downstream/discrim_pkfair_ce_pkfair_3epoch_fresh.csv"

  if [[ "$RUN_EVAL" == "1" ]]; then
    local mode adapter out
    for mode in baseline global pkfair pkfair_kl pkfair_ce; do
      adapter=""; out="$base_resume"
      case "$mode" in
        global) adapter="$global"; out="$global_resume";;
        pkfair) adapter="$pk"; out="$pk_resume";;
        pkfair_kl) adapter="$pkkl"; out="$pkkl_resume";;
        pkfair_ce) adapter="$pkce"; out="$pkce_resume";;
      esac
      fresh_file "$out"
      local args=("$PY" 6_downstream_evaluation/evaluate_resume_fairness_top100.py
        --mode "$mode" --base_model_path "$MODEL_PATH" --dataset_json_path "$DATASET_JSON"
        --biased_csv_path "$ranking" --sample_size 100 --output_csv_path "$out"
        --model_type "$MODEL_TYPE" --resume_prompt_mode summary_only)
      [[ -n "$adapter" ]] && args+=(--adapter_path "$adapter")
      run_cmd "${args[@]}"
    done

    for mode in baseline global pkfair pkfair_kl pkfair_ce; do
      adapter=""; out="$base_discrim"
      case "$mode" in
        global) adapter="$global"; out="$global_discrim";;
        pkfair) adapter="$pk"; out="$pk_discrim";;
        pkfair_kl) adapter="$pkkl"; out="$pkkl_discrim";;
        pkfair_ce) adapter="$pkce"; out="$pkce_discrim";;
      esac
      fresh_file "$out"
      local discrim_args=("$PY" 6_downstream_evaluation/evaluate_models_discrim.py
        --dataset_path "$DISCRIM_JSON" --base_model_path "$MODEL_PATH" --mode "$mode"
        --model_type "$MODEL_TYPE" --csv_path "$out" --model_name_suffix "${mode}_pkfair_3epoch")
      [[ -n "$adapter" ]] && discrim_args+=(--adapter_path "$adapter")
      run_cmd "${discrim_args[@]}"
    done
  fi

  if [[ "$RUN_PLOTS" == "1" ]]; then
    local plot_dir="$downstream"
    local resume_pdf="$plot_dir/Figure8_${MODEL_NAME}_resume_pkfair_3epoch.pdf"
    local resume_png="$plot_dir/Figure8_${MODEL_NAME}_resume_pkfair_3epoch.png"
    local discrim_pdf="$plot_dir/Figure8_${MODEL_NAME}_discrim_pkfair_3epoch.pdf"
    fresh_file "$resume_pdf"; fresh_file "$resume_png"; fresh_file "$discrim_pdf"
    run_cmd "$PY" 6_downstream_evaluation/plot_resume_figure8.py \
      --baseline_csv "$base_resume" --global_csv "$global_resume" \
      --pfairft_csv "$pk_resume" --pfairft_kl_csv "$pkkl_resume" \
      --pfairft_kl_ce_csv "$pkce_resume" --out_pdf "$resume_pdf" --out_png "$resume_png" \
      --model_label "$MODEL_NAME" --pfairft_label PKFair --pfairft_kl_label PKFair-KL \
      --pfairft_kl_ce_label PKFair-CE --global_label Global
    run_cmd "$PY" 6_downstream_evaluation/plot_figure8.py \
      --baseline_csv "$base_discrim" --global_csv "$global_discrim" \
      --pfairft_csv "$pk_discrim" --pfairft_kl_csv "$pkkl_discrim" \
      --pfairft_kl_ce_csv "$pkce_discrim" --out_pdf "$discrim_pdf" \
      --model_label "$MODEL_NAME" --pfairft_label PKFair --pfairft_kl_label PKFair-KL \
      --pfairft_kl_ce_label PKFair-CE --global_label Global
    local paper_dir="$root/paper_plots/figure8_downstream"
    run_cmd mkdir -p "$paper_dir"
    run_cmd cp "$resume_pdf" "$paper_dir/$(basename "$resume_pdf")"
    run_cmd cp "$resume_png" "$paper_dir/$(basename "$resume_png")"
    run_cmd cp "$discrim_pdf" "$paper_dir/$(basename "$discrim_pdf")"
  fi
}

for MODEL_NAME in "${MODELS[@]}"; do
  if [[ -n "$MODEL_FILTER" && ",$MODEL_FILTER," != *",$MODEL_NAME,"* ]]; then
    continue
  fi
  configure_model "$MODEL_NAME"
  if [[ -n "$GPU_OVERRIDE" ]]; then
    GPU_SET="$GPU_OVERRIDE"
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_SET"
  root="$PROJECT_ROOT/results/$MODEL_NAME"
  ranking="$root/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv"
  heads="$root/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
  echo "===== $MODEL_NAME | type=$MODEL_TYPE | GPUs=$GPU_SET | batch=$BATCH_SIZE x accum=$GRAD_ACCUM ====="

  if [[ "$RUN_UPSTREAM" == "1" ]]; then ensure_upstream "$MODEL_NAME"; fi
  if [[ "$CLEAN_OLD_CHECKPOINTS" == "1" ]]; then cleanup_comparison_checkpoints "$root" "$MODEL_NAME"; fi
  if [[ "$RUN_TRAIN" == "1" ]]; then
    train_global "$root" "$ranking"
    train_pkfair "$root" "$ranking" "$heads" fairness_kl_ce pkfair_fairness_kl_ce_yesno_summary_only_current_ranking_full_3epoch
    train_pkfair "$root" "$ranking" "$heads" fairness_kl pkfair_fairness_kl_yesno_summary_only_current_ranking_full_3epoch
    train_pkfair "$root" "$ranking" "$heads" fairness_ce pkfair_fairness_ce_yesno_summary_only_current_ranking_full_3epoch
  fi
  evaluate_and_plot "$root" "$ranking"
done

echo "Nine-model PKFair resume workflow completed."
