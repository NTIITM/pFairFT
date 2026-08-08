#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
QWEN_PY="${QWEN_PY:-/home/common1/hwluo/anaconda3/envs/cognitive_mirrors_py39/bin/python}"
MODEL_NAME="${MODEL_NAME:-all}"
DATASET_PATH="${DATASET_PATH:-data/adult_datasets/adult_race_paired.json}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PAIRS="${MAX_PAIRS:-0}"
DRY_RUN="${DRY_RUN:-0}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"

resolve_model() {
  local name="$1"
  MODEL_PY="$PY"
  case "$name" in
    Llama-3.2-1B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      ;;
    Llama-3.2-3B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      ;;
    Meta-Llama-3-8B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      ;;
    Qwen3-1.7B|Qwen3-4B|Qwen3-8B)
      MODEL_PATH="/mnt/nfs/huggingface/Qwen/$name"
      MODEL_TYPE=qwen
      MODEL_PY="$QWEN_PY"
      HEADS_DIR="results/$name/sensitive_heads"
      ;;
    DeepSeek-V2-Lite-Chat)
      MODEL_PATH=/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat
      MODEL_TYPE=deepseek
      HEADS_DIR="results/$name/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
      ;;
    JetMoE-8B-Chat)
      MODEL_PATH=/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat
      MODEL_TYPE=jetmoe
      HEADS_DIR="results/$name/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
      ;;
    OLMoE-1B-7B-0924-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/allenai/OLMoE-1B-7B-0924-Instruct
      MODEL_TYPE=olmoe
      HEADS_DIR="results/$name/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
      ;;
    *)
      echo "Unknown model: $name" >&2
      return 2
      ;;
  esac
  SELECTED_MLP="results/$name/mlp_analysis/selected_layers_top100/selected_mlp_layers_elbow.json"
  MLP_MEANS="results/$name/mlp_analysis/mlp_means_resume.pkl"
  BASELINE_DIR="results/$name/intervention_ablation/adult_race_yesno_full_baseline_seed_42"
  TOP_DIR="results/$name/intervention_ablation/adult_race_yesno_high_gap_top${TOP_K}_seed_42"
  EVALUATION_DIR="$TOP_DIR/evaluation"
}

validate_inputs() {
  for required in \
    "$DATASET_PATH" \
    "$MODEL_PATH/config.json" \
    "$HEADS_DIR/selected_heads_elbow.json" \
    "$HEADS_DIR/results.pkl" \
    "$SELECTED_MLP" \
    "$MLP_MEANS"; do
    if [[ ! -f "$required" ]]; then
      echo "Missing required input: $required" >&2
      return 3
    fi
  done
}

evaluation_complete() {
  local path="$EVALUATION_DIR/metadata.json"
  [[ -f "$path" ]] || return 1
  [[ "$(jq -r '.status' "$path")" == "complete" ]] || return 1
  [[ "$(jq -r '.pairs' "$path")" == "$TOP_K" ]] || return 1
  [[ "$(jq -r '.seed' "$path")" == "$SEED" ]] || return 1
  [[ "$(jq -r '.evaluation_protocol' "$path")" == "yes_no_income_gt_50k_v1" ]] || return 1
  [[ "$(jq -r '.head_sweep_complete' "$path")" == "true" ]] || return 1
  [[ "$(jq -r '.completed_conditions | sort | join(" ")' "$path")" == \
    "base key_heads key_mlps random_heads" ]]
}

run_model() {
  local name="$1"
  local gpu="$2"
  resolve_model "$name"
  validate_inputs
  if evaluation_complete; then
    echo "Skipping complete Adult race run for $name"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "PLAN model=$name gpu=$gpu python=$MODEL_PY baseline=$BASELINE_DIR top=$TOP_DIR"
    return 0
  fi

  mkdir -p "$BASELINE_DIR" "$TOP_DIR" "$EVALUATION_DIR" "results/$name/logs"
  local log_path="results/$name/logs/adult_race_yesno_top${TOP_K}_seed_${SEED}.log"
  echo "Starting $name on physical GPU $gpu; log=$log_path"
  {
    CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" \
      data/adult_datasets/evaluate_fairness_score.py \
      --dataset_path "$DATASET_PATH" \
      --model_path "$MODEL_PATH" \
      --model_type "$MODEL_TYPE" \
      --model_name "$name" \
      --batch_size "$BATCH_SIZE" \
      --max_pairs "$MAX_PAIRS" \
      --device cuda \
      --resume \
      --output_dir "$BASELINE_DIR"

    "$PY" data/adult_datasets/select_high_gap_pairs.py \
      --dataset_path "$DATASET_PATH" \
      --baseline_dir "$BASELINE_DIR" \
      --output_dir "$TOP_DIR" \
      --top_k "$TOP_K" \
      --sensitive_heads_dir "$HEADS_DIR" \
      --selected_mlp_path "$SELECTED_MLP" \
      --mlp_embeddings_path "$MLP_MEANS"

    CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" \
      4_intervention_ablation/head_intervention/evaluate_intervention_adult_race.py \
      --dataset_path "$TOP_DIR/selected_top_${TOP_K}.json" \
      --model_path "$MODEL_PATH" \
      --model_type "$MODEL_TYPE" \
      --model_name "$name" \
      --sensitive_heads_dir "$TOP_DIR/component_snapshot/heads" \
      --selected_mlp_path "$TOP_DIR/component_snapshot/mlps/selected_mlp_layers_elbow.json" \
      --mlp_embeddings_path "$TOP_DIR/component_snapshot/mlps/mlp_means_resume.pkl" \
      --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --device cuda \
      --run_head_sweep \
      --resume \
      --output_dir "$EVALUATION_DIR"
  } >"$log_path" 2>&1
}

models=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Meta-Llama-3-8B-Instruct
  Qwen3-1.7B
  Qwen3-4B
  Qwen3-8B
  DeepSeek-V2-Lite-Chat
  JetMoE-8B-Chat
  OLMoE-1B-7B-0924-Instruct
)
gpus=(
  "${GPU_LLAMA_1B:-0}"
  "${GPU_LLAMA_3B:-0}"
  "${GPU_LLAMA_8B:-1}"
  "${GPU_QWEN_1B:-2}"
  "${GPU_QWEN_4B:-3}"
  "${GPU_QWEN_8B:-4}"
  "${GPU_DEEPSEEK:-0,7}"
  "${GPU_JETMOE:-5}"
  "${GPU_OLMOE:-6}"
)

if [[ "$MODEL_NAME" != "all" ]]; then
  for idx in "${!models[@]}"; do
    if [[ "${models[$idx]}" == "$MODEL_NAME" ]]; then
      run_model "$MODEL_NAME" "${GPU:-${gpus[$idx]}}"
      exit 0
    fi
  done
  echo "Unknown model: $MODEL_NAME" >&2
  exit 2
fi

pids=()
names=()
llama_1b_pid=""
for idx in "${!models[@]}"; do
  if [[ "${models[$idx]}" == "Llama-3.2-3B-Instruct" || \
        "${models[$idx]}" == "DeepSeek-V2-Lite-Chat" ]]; then
    continue
  fi
  run_model "${models[$idx]}" "${gpus[$idx]}" &
  pid="$!"
  if [[ "${models[$idx]}" == "Llama-3.2-1B-Instruct" ]]; then
    llama_1b_pid="$pid"
  else
    pids+=("$pid")
    names+=("${models[$idx]}")
  fi
done

status=0
if wait "$llama_1b_pid"; then
  echo "Completed Llama-3.2-1B-Instruct"
else
  echo "Failed Llama-3.2-1B-Instruct" >&2
  status=1
fi

run_model DeepSeek-V2-Lite-Chat "${GPU_DEEPSEEK:-0,7}" &
pids+=("$!")
names+=("DeepSeek-V2-Lite-Chat")

for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "Completed ${names[$idx]}"
  else
    echo "Failed ${names[$idx]}" >&2
    status=1
  fi
done
if run_model Llama-3.2-3B-Instruct "${GPU_LLAMA_3B:-0}"; then
  echo "Completed Llama-3.2-3B-Instruct"
else
  echo "Failed Llama-3.2-3B-Instruct" >&2
  status=1
fi
exit "$status"
