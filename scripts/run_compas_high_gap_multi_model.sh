#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
MODEL_NAME="${MODEL_NAME:-all}"
OUTPUT_NAME="${OUTPUT_NAME:-compas_high_gap_top100_seed_42}"
TOP_K="${TOP_K:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DRY_RUN="${DRY_RUN:-0}"
CURVE_K="${CURVE_K:-50 100 200 500 1000}"

cd "$PROJECT_ROOT"

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
  "${GPU_DEEPSEEK:-7}"
  "${GPU_JETMOE:-5}"
  "${GPU_OLMOE:-6}"
)

is_complete() {
  local metadata_path="$1"
  [[ -f "$metadata_path" ]] || return 1
  [[ "$(jq -r '.status' "$metadata_path")" == "complete" ]] || return 1
  [[ "$(jq -r '.pairs' "$metadata_path")" == "$TOP_K" ]] || return 1
  [[ "$(jq -r '.seed' "$metadata_path")" == "$SEED" ]] || return 1
  [[ "$(jq -r '.batch_size' "$metadata_path")" == "$BATCH_SIZE" ]] || return 1
  [[ "$(jq -r '.completed_conditions | join(" ")' "$metadata_path")" == \
    "base key_heads random_heads key_mlps" ]]
}

run_model() {
  local name="$1"
  local gpu="$2"
  local full_dir="results/$name/intervention_ablation/compas_full_seed_42"
  local high_gap_dir="results/$name/intervention_ablation/$OUTPUT_NAME"
  local selected_path="$high_gap_dir/selected_top_${TOP_K}.json"
  local evaluation_dir="$high_gap_dir/evaluation"
  local metadata_path="$evaluation_dir/metadata.json"
  local snapshot_dir="$high_gap_dir/component_snapshot"

  if is_complete "$metadata_path"; then
    echo "Skipping complete $name: $metadata_path"
    return 0
  fi
  if [[ ! -f "$full_dir/metadata.json" || ! -f "$full_dir/per_pair_base.csv" ]]; then
    echo "Missing complete full result for $name: $full_dir" >&2
    return 3
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "PLAN model=$name gpu=$gpu full=$full_dir selected=$selected_path output=$evaluation_dir"
    return 0
  fi

  mkdir -p "$high_gap_dir" "results/$name/logs"
  "$PY" data/compas/select_high_gap_pairs.py \
    --dataset_path data/compas/compas_paired.json \
    --full_result_dir "$full_dir" \
    --output_dir "$high_gap_dir" \
    --top_k "$TOP_K" \
    --curve_k $CURVE_K

  MODEL_NAME="$name" \
  GPU="$gpu" \
  OUTPUT_NAME="$OUTPUT_NAME/evaluation" \
  DATASET_PATH="$selected_path" \
  HEADS_DIR_OVERRIDE="$snapshot_dir/heads" \
  SELECTED_MLP_OVERRIDE="$snapshot_dir/mlps/selected_mlp_layers_elbow.json" \
  MLP_MEANS_OVERRIDE="$snapshot_dir/mlps/mlp_means_resume.pkl" \
  MAX_PAIRS=0 \
  BATCH_SIZE="$BATCH_SIZE" \
  SEED="$SEED" \
    bash scripts/run_compas_full_multi_model.sh
}

if [[ "$MODEL_NAME" != "all" ]]; then
  found=0
  for idx in "${!models[@]}"; do
    if [[ "${models[$idx]}" == "$MODEL_NAME" ]]; then
      run_model "$MODEL_NAME" "${GPU:-${gpus[$idx]}}"
      found=1
      break
    fi
  done
  if [[ "$found" == "0" ]]; then
    echo "Unknown model: $MODEL_NAME" >&2
    exit 2
  fi
  exit 0
fi

pids=()
names=()
for idx in "${!models[@]}"; do
  if [[ "${models[$idx]}" == "Llama-3.2-3B-Instruct" ]]; then
    continue
  fi
  run_model "${models[$idx]}" "${gpus[$idx]}" &
  pids+=("$!")
  names+=("${models[$idx]}")
done

status=0
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
