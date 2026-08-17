#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
QWEN_PY="${QWEN_PY:-/home/common1/hwluo/anaconda3/envs/RL/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/table2_nine_models}"
DATASET="$PROJECT_ROOT/data/resume/qwen_summaries_with_race.json"
DRY_RUN="${DRY_RUN:-1}"
RUN_RESUME="${RUN_RESUME:-1}"
RUN_MMLU="${RUN_MMLU:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
MODEL_FILTER="${MODEL_FILTER:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export CUDA_VISIBLE_DEVICES TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODELS=(
  Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3-8B-Instruct
  Qwen3-1.7B Qwen3-4B Qwen3-8B
  DeepSeek-V2-Lite-Chat JetMoE-8B-Chat OLMoE-1B-7B-0924-Instruct
)

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then "$@"; fi
}

resume_complete() {
  local path="$1" prompt_type="$2"
  [[ -s "$path" && -s "$path.metadata.json" ]] || return 1
  "$PY" - "$path" "$prompt_type" <<'PY'
import csv,json,sys
p,prompt=sys.argv[1:]
with open(p,newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
with open(p+'.metadata.json',encoding='utf-8') as h: metadata=json.load(h)
if len(rows)!=100 or len({int(r['index']) for r in rows})!=100: raise SystemExit(1)
if metadata.get('prompt_type')!=prompt or metadata.get('resume_prompt_mode')!='summary_only': raise SystemExit(1)
PY
}

intervention_resume_complete() {
  local path="$1"
  [[ -s "$path" && -s "$path.metadata.json" ]] || return 1
  "$PY" - "$path" <<'PY'
import csv,json,sys
with open(sys.argv[1],newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
if len(rows)!=100 or len({int(r['index']) for r in rows})!=100: raise SystemExit(1)
with open(sys.argv[1]+'.metadata.json',encoding='utf-8') as h: metadata=json.load(h)
if metadata.get('intervention_mode')!='partial' or metadata.get('seed')!=42: raise SystemExit(1)
PY
}

mmlu_complete() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PY" - "$path" <<'PY'
import json,math,sys
with open(sys.argv[1],encoding='utf-8') as h: d=json.load(h)
if int(d.get('count',d.get('total',-1)))!=1531 or not math.isfinite(float(d['ce'])): raise SystemExit(1)
PY
}

intervention_mmlu_complete() {
  local path="$1"
  mmlu_complete "$path" || return 1
  "$PY" - "$path" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as h: d=json.load(h)
if d.get('intervention_scope')!='all_teacher_forcing_positions': raise SystemExit(1)
PY
}

configure_model() {
  local name="$1"
  MODEL_PY="$PY"
  case "$name" in
    Llama-3.2-1B-Instruct) MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct; MODEL_TYPE=llama ;;
    Llama-3.2-3B-Instruct) MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct; MODEL_TYPE=llama ;;
    Meta-Llama-3-8B-Instruct) MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct; MODEL_TYPE=llama ;;
    Qwen3-1.7B) MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-1.7B; MODEL_TYPE=qwen; MODEL_PY="$QWEN_PY" ;;
    Qwen3-4B) MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-4B; MODEL_TYPE=qwen; MODEL_PY="$QWEN_PY" ;;
    Qwen3-8B) MODEL_PATH=/mnt/nfs/huggingface/Qwen/Qwen3-8B; MODEL_TYPE=qwen; MODEL_PY="$QWEN_PY" ;;
    DeepSeek-V2-Lite-Chat) MODEL_PATH=/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat; MODEL_TYPE=deepseek ;;
    JetMoE-8B-Chat) MODEL_PATH=/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat; MODEL_TYPE=jetmoe ;;
    OLMoE-1B-7B-0924-Instruct) MODEL_PATH=/mnt/nfs/huggingface/allenai/OLMoE-1B-7B-0924-Instruct; MODEL_TYPE=olmoe ;;
    *) echo "Unsupported model $name" >&2; return 2 ;;
  esac
}

for MODEL_NAME in "${MODELS[@]}"; do
  [[ -z "$MODEL_FILTER" || "$MODEL_NAME" == "$MODEL_FILTER" ]] || continue
  configure_model "$MODEL_NAME"
  RESULT_ROOT="$PROJECT_ROOT/results/$MODEL_NAME"
  RANKING="$RESULT_ROOT/biased_samples/biased_samples_ranking_summary_only_current_prompt.csv"
  HEADS="$RESULT_ROOT/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
  GLOBAL="$RESULT_ROOT/global_lora_raw_summary_qv_current_ranking_full_3epoch/final_model"
  if [[ "$MODEL_NAME" == "Meta-Llama-3-8B-Instruct" ]]; then
    GLOBAL="$RESULT_ROOT/global_lora_oldtarget_raw_summary_full_3epoch/final_model"
  elif [[ "$MODEL_NAME" == "OLMoE-1B-7B-0924-Instruct" && ! -s "$GLOBAL/adapter_model.safetensors" ]]; then
    GLOBAL="$RESULT_ROOT/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
  fi
  PFAIRFT="$RESULT_ROOT/pkfair_fairness_kl_yesno_summary_only_current_ranking_full_3epoch/final_model"
  PFAIRFT_KL="$RESULT_ROOT/pkfair_fairness_kl_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
  MODEL_OUT="$OUTPUT_ROOT/$MODEL_NAME"
  for required in "$MODEL_PATH/config.json" "$RANKING" "$HEADS/results.pkl" "$HEADS/selected_heads_elbow.json" "$GLOBAL/adapter_model.safetensors" "$PFAIRFT/adapter_model.safetensors" "$PFAIRFT_KL/adapter_model.safetensors"; do
    [[ -s "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 1; }
  done
  run mkdir -p "$MODEL_OUT"
  if [[ "$DRY_RUN" == "0" ]]; then printf '%s\n' "$GLOBAL" > "$MODEL_OUT/global_adapter_path.txt"; else printf '+ write %q\n' "$MODEL_OUT/global_adapter_path.txt"; fi

  if [[ "$RUN_RESUME" == "1" ]]; then
    debiased="$MODEL_OUT/resume_debiased_prompt.csv"
    if ! resume_complete "$debiased" debiased_prompt; then
      run "$MODEL_PY" 6_downstream_evaluation/evaluate_resume_fairness_top100.py \
        --mode table2_debiased_prompt --base_model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
        --dataset_json_path "$DATASET" --biased_csv_path "$RANKING" --sample_size 100 \
        --resume_prompt_mode summary_only --prompt_type debiased_prompt --output_csv_path "$debiased"
    fi
    global_resume="$MODEL_OUT/resume_global.csv"
    if ! resume_complete "$global_resume" prompt; then
      run "$MODEL_PY" 6_downstream_evaluation/evaluate_resume_fairness_top100.py \
        --mode table2_global --base_model_path "$MODEL_PATH" --adapter_path "$GLOBAL" --model_type "$MODEL_TYPE" \
        --dataset_json_path "$DATASET" --biased_csv_path "$RANKING" --sample_size 100 \
        --resume_prompt_mode summary_only --prompt_type prompt --output_csv_path "$global_resume"
    fi
    inference_resume="$MODEL_OUT/resume_inference_time.csv"
    if ! intervention_resume_complete "$inference_resume"; then
      run "$MODEL_PY" 4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_resume.py \
        --base_model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_json_path "$DATASET" \
        --biased_csv_path "$RANKING" --sample_size 100 --resume_prompt_mode summary_only \
        --sensitive_heads_dir "$HEADS" --intervention_mode partial --intervention_strength 1.0 \
        --seed 42 --output_csv_path "$inference_resume"
    fi
  fi

  if [[ "$RUN_MMLU" == "1" ]]; then
    for spec in "base::mmlu_base.json" "global:$GLOBAL:mmlu_global.json" "pfairft:$PFAIRFT:mmlu_pfairft.json" "pfairft_kl:$PFAIRFT_KL:mmlu_pfairft_kl.json"; do
      IFS=: read -r mode adapter output <<< "$spec"
      output="$MODEL_OUT/$output"
      if ! mmlu_complete "$output"; then
        args=("$MODEL_PY" 6_downstream_evaluation/evaluate_mmlu_ce.py --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --split validation --max_samples 0 --out_json "$output")
        [[ -n "$adapter" ]] && args+=(--adapter_path "$adapter")
        run "${args[@]}"
      fi
    done
    inference_mmlu="$MODEL_OUT/mmlu_inference_time_teacher_forcing_all.json"
    if ! intervention_mmlu_complete "$inference_mmlu"; then
      run "$MODEL_PY" 4_intervention_ablation/projection_intervention/evaluate_mmlu_intervention.py \
        --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --split validation --max_samples -1 \
        --sensitive_heads_dir "$HEADS" --intervention_mode partial --intervention_strength 1.0 \
        --output_json "$inference_mmlu"
    fi
  fi
done

if [[ "$RUN_BUILD" == "1" && -z "$MODEL_FILTER" ]]; then
  run "$PY" 6_downstream_evaluation/build_table2_nine_models.py \
    --project_root "$PROJECT_ROOT" --output_root "$OUTPUT_ROOT"
fi
