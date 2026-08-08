#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/common1/hwluo/project/pFairFT}"
PY="${PY:-/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python}"
MODEL_NAME="${MODEL_NAME:-Meta-Llama-3-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
GPU="${GPU:-6,7}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
RUN_DEBIASED="${RUN_DEBIASED:-1}"
RUN_INFERENCE_TIME="${RUN_INFERENCE_TIME:-1}"
RUN_HEAD_ANALYSIS="${RUN_HEAD_ANALYSIS:-1}"
RUN_SNAPSHOT="${RUN_SNAPSHOT:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESULT_ROOT="$PROJECT_ROOT/results/$MODEL_NAME"
DOWNSTREAM="$RESULT_ROOT/downstream_evaluation"
DATASET="$PROJECT_ROOT/data/discrim-eval/dataset_paired.json"
HEADS_DIR="$RESULT_ROOT/sensitive_heads_moefreeze_top100_summary_only_current_ranking"
SELECTED_HEADS="$HEADS_DIR/selected_heads_elbow.json"
GLOBAL_ADAPTER="$RESULT_ROOT/global_lora_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
PFAIRFT_ADAPTER="$RESULT_ROOT/pkfair_fairness_kl_ce_yesno_summary_only_current_ranking_full_3epoch/final_model"
BASELINE_CSV="$DOWNSTREAM/discrim_baseline_pkfair_3epoch_fresh.csv"
GLOBAL_CSV="$DOWNSTREAM/discrim_global_yesno_pkfair_3epoch_fresh.csv"
PFAIRFT_CSV="$DOWNSTREAM/discrim_pkfair_pkfair_3epoch_fresh.csv"
DEBIASED_CSV="$DOWNSTREAM/discrim_baseline_debiased_prompt_figure5_fresh.csv"
INFERENCE_ROOT="$RESULT_ROOT/inference_time_figure5"
INFERENCE_CSV="$INFERENCE_ROOT/discrim_partial.csv"
HEAD_OUTPUT="$RESULT_ROOT/figure5_analysis/head_conditions_current"
FIGURE_ROOT="$PROJECT_ROOT/nmi_plot/figure5"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required input is missing or empty: $1" >&2
    exit 1
  fi
}

csv_complete() {
  local path="$1"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -eq 2521 ]]
}

head_analysis_complete() {
  [[ -s "$HEAD_OUTPUT/metadata.json" ]] || return 1
  "$PY" - "$HEAD_OUTPUT" <<'PY' >/dev/null
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
qid = int(metadata["selected_comparison_qid"])
paths = [
    root / "panel_b_qid90/debiased_prompt_md.npy",
    root / "panel_b_qid90/debiased_prompt_with_context_md.npy",
    root / f"candidates/qid{qid}/original_md.npy",
    root / f"candidates/qid{qid}/global_md.npy",
    root / f"candidates/qid{qid}/pfairft_md.npy",
]
assert metadata["num_selected_heads"] == 52 and metadata["batch_size"] == 1
assert metadata["candidate_count"] == 51 and len(metadata["candidate_qids"]) == 51
assert all(path.is_file() and np.load(path).shape == (32, 32) for path in paths)
for label in ("original", "global", "pfairft"):
    assert len(list((root / "candidates").glob(f"qid*/{label}_md.npy"))) == 51
PY
}

for path in \
  "$MODEL_PATH/config.json" \
  "$DATASET" \
  "$SELECTED_HEADS" \
  "$HEADS_DIR/results.pkl" \
  "$GLOBAL_ADAPTER/adapter_model.safetensors" \
  "$PFAIRFT_ADAPTER/adapter_model.safetensors" \
  "$BASELINE_CSV" \
  "$GLOBAL_CSV" \
  "$PFAIRFT_CSV"; do
  require_file "$path"
done

echo "Figure 5 model: $MODEL_NAME"
echo "GPUs: $GPU | dry_run=$DRY_RUN | force=$FORCE"

if [[ "$RUN_DEBIASED" == "1" ]]; then
  if [[ "$FORCE" == "0" ]] && csv_complete "$DEBIASED_CSV"; then
    echo "Reusing complete debiased-prompt CSV: $DEBIASED_CSV"
  else
    run_cmd "$PY" "$PROJECT_ROOT/6_downstream_evaluation/evaluate_models_discrim.py" \
      --dataset_path "$DATASET" --base_model_path "$MODEL_PATH" \
      --model_type "$MODEL_TYPE" --mode baseline_debiased_prompt_figure5 \
      --prompt_type debiased_prompt --csv_path "$DEBIASED_CSV" \
      --model_name_suffix baseline_debiased_prompt_figure5
  fi
fi

if [[ "$RUN_INFERENCE_TIME" == "1" ]]; then
  if [[ "$FORCE" == "0" ]] && csv_complete "$INFERENCE_CSV" && [[ -s "$INFERENCE_ROOT/discrim_metadata.json" ]]; then
    echo "Reusing complete inference-time CSV: $INFERENCE_CSV"
  else
    run_cmd "$PY" "$PROJECT_ROOT/4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_discrim.py" \
      --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_path "$DATASET" \
      --sensitive_heads_dir "$HEADS_DIR" --intervention_mode partial \
      --intervention_strength 1.0 --seed 42 --output_dir "$INFERENCE_ROOT" \
      --csv_path "$INFERENCE_CSV"
  fi
fi

if [[ "$RUN_HEAD_ANALYSIS" == "1" ]]; then
  if [[ "$FORCE" == "0" ]] && head_analysis_complete; then
    echo "Reusing complete Figure 5 head analysis: $HEAD_OUTPUT"
  else
    run_cmd "$PY" "$PROJECT_ROOT/3_pattern_analysis/model_comparison/analyze_figure5_head_conditions.py" \
      --base_model_path "$MODEL_PATH" --dataset_path "$DATASET" \
      --selected_heads_json "$SELECTED_HEADS" --baseline_csv "$BASELINE_CSV" \
      --global_csv "$GLOBAL_CSV" --pfairft_csv "$PFAIRFT_CSV" \
      --global_adapter "$GLOBAL_ADAPTER" --pfairft_adapter "$PFAIRFT_ADAPTER" \
      --candidate_count 51 --panel_b_qid 90 --batch_size 1 --output_dir "$HEAD_OUTPUT"
  fi
fi

if [[ "$RUN_SNAPSHOT" == "1" ]]; then
  if [[ "$DRY_RUN" == "0" ]]; then
    csv_complete "$DEBIASED_CSV" || { echo "Debiased CSV is incomplete" >&2; exit 1; }
    csv_complete "$INFERENCE_CSV" || { echo "Inference-time CSV is incomplete" >&2; exit 1; }
    head_analysis_complete || { echo "Head analysis is incomplete" >&2; exit 1; }
  fi
  run_cmd "$PY" "$FIGURE_ROOT/prepare_figure5_data.py" \
    --project_root "$PROJECT_ROOT" --model_name "$MODEL_NAME" \
    --output_dir "$FIGURE_ROOT/data/current"
fi

if [[ "$RUN_PLOT" == "1" ]]; then
  run_cmd "$PY" "$FIGURE_ROOT/plot_figure5.py"
fi

echo "Figure 5 workflow completed."
