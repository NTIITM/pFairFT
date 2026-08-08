#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
LEGACY_ROOT="${LEGACY_ROOT:-/home/common1/hwluo/project/fairness_llm_new}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
QWEN_PY="${QWEN_PY:-/home/common1/hwluo/anaconda3/envs/cognitive_mirrors_py39/bin/python}"
MODEL_NAME="${MODEL_NAME:-all}"
MAX_PAIRS="${MAX_PAIRS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_NAME="${OUTPUT_NAME:-compas_full_seed_42}"
DATASET_PATH="${DATASET_PATH:-data/compas/compas_paired.json}"
SEED="${SEED:-42}"
CONDITIONS="${CONDITIONS:-}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"

prepare_dense_artifacts() {
  local name="$1"
  local head_source="$LEGACY_ROOT/exp2/sensitive_heads_${name}_top100"
  local mlp_selected_source="$LEGACY_ROOT/exp15/mlp_elbow_${name}/selected_mlp_layers_elbow.json"
  local mlp_means_source="$LEGACY_ROOT/exp15/mlp_means_resume_${name}.pkl"
  local head_target="results/$name/sensitive_heads"
  local mlp_target="results/$name/mlp_analysis/selected_layers_top100"

  mkdir -p "$head_target" "$mlp_target" "results/$name/mlp_analysis"
  cp -p "$head_source/selected_heads_elbow.json" "$head_target/selected_heads_elbow.json"
  cp -p "$head_source/results.pkl" "$head_target/results.pkl"
  cp -p "$mlp_selected_source" "$mlp_target/selected_mlp_layers_elbow.json"
  cp -p "$mlp_means_source" "results/$name/mlp_analysis/mlp_means_resume.pkl"
}

resolve_model() {
  local name="$1"
  MODEL_PY="$PY"
  case "$name" in
    Llama-3.2-1B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      prepare_dense_artifacts "$name"
      ;;
    Llama-3.2-3B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      prepare_dense_artifacts "$name"
      ;;
    Meta-Llama-3-8B-Instruct)
      MODEL_PATH=/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct
      MODEL_TYPE=llama
      HEADS_DIR="results/$name/sensitive_heads"
      prepare_dense_artifacts "$name"
      ;;
    Qwen3-1.7B|Qwen3-4B|Qwen3-8B)
      MODEL_PATH="/mnt/nfs/huggingface/Qwen/$name"
      MODEL_TYPE=qwen
      MODEL_PY="$QWEN_PY"
      HEADS_DIR="results/$name/sensitive_heads"
      prepare_dense_artifacts "$name"
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
  OUTPUT_DIR="results/$name/intervention_ablation/$OUTPUT_NAME"
}

run_model() {
  local name="$1"
  local gpu="$2"
  resolve_model "$name"
  HEADS_DIR="${HEADS_DIR_OVERRIDE:-$HEADS_DIR}"
  SELECTED_MLP="${SELECTED_MLP_OVERRIDE:-$SELECTED_MLP}"
  MLP_MEANS="${MLP_MEANS_OVERRIDE:-$MLP_MEANS}"
  for required in \
    "$MODEL_PATH/config.json" \
    "$HEADS_DIR/selected_heads_elbow.json" \
    "$HEADS_DIR/results.pkl" \
    "$SELECTED_MLP" \
    "$MLP_MEANS"; do
    if [[ ! -f "$required" ]]; then
      echo "Missing required artifact for $name: $required" >&2
      return 3
    fi
  done
  mkdir -p "$OUTPUT_DIR" "results/$name/logs"
  local log_name="${OUTPUT_NAME//\//_}"
  local log_path="results/$name/logs/${log_name}.log"
  local condition_args=()
  if [[ -n "$CONDITIONS" ]]; then
    read -r -a selected_conditions <<< "$CONDITIONS"
    condition_args=(--conditions "${selected_conditions[@]}")
  fi
  echo "Starting $name on physical GPU $gpu; log=$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$MODEL_PY" \
    4_intervention_ablation/head_intervention/evaluate_intervention_compas_full.py \
    --dataset_path "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --sensitive_heads_dir "$HEADS_DIR" \
    --selected_mlp_path "$SELECTED_MLP" \
    --mlp_embeddings_path "$MLP_MEANS" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --max_pairs "$MAX_PAIRS" \
    --device cuda \
    --resume \
    "${condition_args[@]}" \
    --output_dir "$OUTPUT_DIR" >"$log_path" 2>&1
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
  "${GPU_DEEPSEEK:-7}"
  "${GPU_JETMOE:-5}"
  "${GPU_OLMOE:-6}"
)

if [[ "$MODEL_NAME" != "all" ]]; then
  gpu="${GPU:-0}"
  run_model "$MODEL_NAME" "$gpu"
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
