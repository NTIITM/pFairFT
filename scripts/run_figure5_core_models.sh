#!/usr/bin/env bash
set -euo pipefail
# Resumable corrected-Global + core Figure 5 workflow for non-Llama-8B models.
PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
GPU="${GPU:-0}"; DRY_RUN="${DRY_RUN:-1}"; RUN_TRAIN="${RUN_TRAIN:-1}"; RUN_EVAL="${RUN_EVAL:-1}"; RUN_PLOT="${RUN_PLOT:-1}"; GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"; GRADIENT_CHECKPOINTING_REENTRANT="${GRADIENT_CHECKPOINTING_REENTRANT:-0}"
DATASET="$PROJECT_ROOT/data/discrim-eval/dataset_paired.json"; RESUME="$PROJECT_ROOT/data/resume/qwen_summaries_with_race.json"
GLOBAL_NAME="global_lora_raw_summary_qv_current_ranking_full_3epoch"
DEFAULT_MODELS=(Qwen3-1.7B Qwen3-4B Qwen3-8B Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct DeepSeek-V2-Lite-Chat JetMoE-8B-Chat OLMoE-1B-7B-0924-Instruct)
DEFAULT_MODELS_CSV="Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Llama-3.2-1B-Instruct,Llama-3.2-3B-Instruct,DeepSeek-V2-Lite-Chat,JetMoE-8B-Chat,OLMoE-1B-7B-0924-Instruct"
IFS=',' read -r -a MODELS <<< "${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
declare -A PATHS TYPES TARGETS
PATHS[Qwen3-1.7B]=/mnt/nfs/huggingface/Qwen/Qwen3-1.7B; PATHS[Qwen3-4B]=/mnt/nfs/huggingface/Qwen/Qwen3-4B; PATHS[Qwen3-8B]=/mnt/nfs/huggingface/Qwen/Qwen3-8B
PATHS[Llama-3.2-1B-Instruct]=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct; PATHS[Llama-3.2-3B-Instruct]=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct
PATHS[DeepSeek-V2-Lite-Chat]=/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat; PATHS[JetMoE-8B-Chat]=/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat; PATHS[OLMoE-1B-7B-0924-Instruct]=/mnt/nfs/huggingface/allenai/OLMoE-1B-7B-0924-Instruct
for m in Qwen3-1.7B Qwen3-4B Qwen3-8B; do TYPES[$m]=qwen; TARGETS[$m]=q_proj,v_proj; done
for m in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct; do TYPES[$m]=llama; TARGETS[$m]=q_proj,v_proj; done
TYPES[DeepSeek-V2-Lite-Chat]=deepseek; TARGETS[DeepSeek-V2-Lite-Chat]=q_proj,v_proj; TYPES[JetMoE-8B-Chat]=jetmoe; TARGETS[JetMoE-8B-Chat]=kv_proj; TYPES[OLMoE-1B-7B-0924-Instruct]=olmoe; TARGETS[OLMoE-1B-7B-0924-Instruct]=q_proj,v_proj
run(){ printf '+'; printf ' %q' "$@"; printf '\n'; if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi; }
csv_complete(){
  local f="$1"
  [[ -s "$f" ]] || return 1
  # Discrim-Eval must contain exactly 2520 data rows (header + 2520).
  [[ "$(wc -l < "$f")" -eq 2521 ]] || return 1
  "$PY" - "$f" <<'PY'
import csv,sys
p=sys.argv[1]
with open(p,newline='',encoding='utf-8') as h:
    rows=list(csv.DictReader(h))
if len(rows)!=2520 or not {'sample_id','matched_id','decision_question_id','p_yes'}.issubset(rows[0]):
    raise SystemExit(1)
ids={int(r['sample_id']) for r in rows}
if len(ids)!=2520 or any(int(r['matched_id']) not in ids for r in rows): raise SystemExit(1)
PY
}
for MODEL in "${MODELS[@]}"; do
 ROOT="$PROJECT_ROOT/results/$MODEL"; MODEL_PATH="${PATHS[$MODEL]}"; TYPE="${TYPES[$MODEL]}"; TARGET="${TARGETS[$MODEL]}"; RANK="$ROOT/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv"; HEADS="$ROOT/sensitive_heads_moefreeze_top100_summary_only_current_ranking"; [[ -d "$HEADS" ]] || HEADS="$ROOT/sensitive_heads_moefreeze"; [[ -d "$HEADS" ]] || HEADS="$ROOT/sensitive_heads"; GLOBAL="$ROOT/$GLOBAL_NAME/final_model"; DOWN="$ROOT/downstream_evaluation"; if [[ "$MODEL" == OLMoE-1B-7B-0924-Instruct && ! -s "$GLOBAL/adapter_model.safetensors" ]]; then GLOBAL="$ROOT/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"; fi; if [[ "$DRY_RUN" == 0 ]]; then mkdir -p "$DOWN"; fi
 for f in "$MODEL_PATH/config.json" "$RANK" "$HEADS/selected_heads_elbow.json" "$HEADS/results.pkl"; do [[ -s "$f" ]] || { echo "Missing $f" >&2; exit 1; }; done
 if [[ "$RUN_TRAIN" == 1 && ! -s "$GLOBAL/adapter_model.safetensors" ]]; then
  TRAIN_ARGS=(5_finetuning/finetune_global_lora.py --model_path "$MODEL_PATH" --model_type "$TYPE" --dataset_json_path "$RESUME" --output_dir "$ROOT/$GLOBAL_NAME" --sample_csv_path "$RANK" --sample_size 0 --max_samples 0 --no-balanced --resume_prompt_mode summary_only --input_prompt_style raw_summary --lora_target_modules "$TARGET" --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-5 --warmup_steps 0)
  [[ "$GRADIENT_CHECKPOINTING" == 1 ]] && TRAIN_ARGS+=(--gradient_checkpointing)
  [[ "$GRADIENT_CHECKPOINTING_REENTRANT" == 1 ]] && TRAIN_ARGS+=(--gradient_checkpointing_reentrant)
  run env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 "$PY" "${TRAIN_ARGS[@]}"
 fi
 if [[ "$RUN_EVAL" == 1 ]]; then
  for spec in "global:$GLOBAL:discrim_${GLOBAL_NAME}.csv"; do IFS=: read -r mode adapter out <<< "$spec"; csv_complete "$DOWN/$out" || run env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 "$PY" 6_downstream_evaluation/evaluate_models_discrim.py --dataset_path "$DATASET" --base_model_path "$MODEL_PATH" --adapter_path "$adapter" --model_type "$TYPE" --mode "$GLOBAL_NAME" --prompt_type prompt --csv_path "$DOWN/$out" --model_name_suffix "$GLOBAL_NAME"; done
  csv_complete "$DOWN/discrim_baseline_debiased_prompt_figure5_fresh.csv" || run env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 "$PY" 6_downstream_evaluation/evaluate_models_discrim.py --dataset_path "$DATASET" --base_model_path "$MODEL_PATH" --model_type "$TYPE" --mode baseline_debiased_prompt_figure5 --prompt_type debiased_prompt --csv_path "$DOWN/discrim_baseline_debiased_prompt_figure5_fresh.csv" --model_name_suffix baseline_debiased_prompt_figure5
  csv_complete "$ROOT/inference_time_figure5/discrim_partial.csv" || run env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 "$PY" 4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_discrim.py --model_path "$MODEL_PATH" --model_type "$TYPE" --dataset_path "$DATASET" --sensitive_heads_dir "$HEADS" --intervention_mode partial --intervention_strength 1.0 --seed 42 --output_dir "$ROOT/inference_time_figure5" --csv_path "$ROOT/inference_time_figure5/discrim_partial.csv";
 fi
done
if [[ "$RUN_PLOT" == 1 ]]; then run env PROJECT_ROOT="$PROJECT_ROOT" "$PY" nmi_plot/figure5/make_core_appendix.py --project_root "$PROJECT_ROOT" --global_name "$GLOBAL_NAME"; fi
