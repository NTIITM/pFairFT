#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python"
MODEL_ID="LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Meta-Llama-3-8B-Instruct"
MODEL_TYPE="llama"
MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct"
RESULT_ROOT="$PROJECT_ROOT/results/Meta-Llama-3-8B-Instruct-figures-v1"
GPU="6,7"
STAGE=""
DRY_RUN=0
FORCE_STAGE=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_llama3_8b_figures.sh --stage figure1 [options]
  bash scripts/run_llama3_8b_figures.sh --stage figure1-figure5 [options]

Stages are deliberately sequential:
  figure1  Resume ranking and Head/MLP component identification
  figure2  Resume Head/MLP intervention analysis
  figure3  Head/MLP/attention mechanism analysis
  figure4  Discrim-Eval, COMPAS, and Adult intervention analysis
  figure5  3-epoch fine-tuning and mitigation-mechanism analysis

Options:
  --stage NAME          One figureN stage or a forward range such as figure2-figure4
  --dry-run             Print commands without executing experiments
  --force-stage NAME    Archive NAME and all downstream outputs before running
  --gpu IDS             CUDA device list (default: 6,7)
  --python PATH         Python interpreter
  --model-dir PATH      Local ModelScope checkpoint directory
  --result-root PATH    Isolated result root for this five-figure run
  -h, --help            Show this help

Method binding used everywhere in this workflow:
  PFairFT     = fairness_kl
  PFairFT-KL  = fairness_kl_ce
  PFairFT-CE  = fairness_ce
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --stage) [[ $# -ge 2 ]] || die "--stage requires a value"; STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-stage) [[ $# -ge 2 ]] || die "--force-stage requires a value"; FORCE_STAGE="$2"; shift 2 ;;
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; GPU="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PY="$2"; shift 2 ;;
    --model-dir) [[ $# -ge 2 ]] || die "--model-dir requires a value"; MODEL_PATH="$2"; shift 2 ;;
    --result-root) [[ $# -ge 2 ]] || die "--result-root requires a value"; RESULT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$STAGE" ]] || { usage >&2; exit 2; }
[[ -x "$PY" ]] || die "Python interpreter is not executable: $PY"
[[ "$GPU" =~ ^[0-7](,[0-7])*$ ]] || die "--gpu must contain physical GPU indices 0-7"

stage_number() {
  [[ "$1" =~ ^figure([1-5])$ ]] || return 1
  printf '%s' "${BASH_REMATCH[1]}"
}

parse_stage_range() {
  local first last
  if [[ "$STAGE" =~ ^figure([1-5])$ ]]; then
    first="${BASH_REMATCH[1]}"; last="$first"
  elif [[ "$STAGE" =~ ^figure([1-5])-figure([1-5])$ ]]; then
    first="${BASH_REMATCH[1]}"; last="${BASH_REMATCH[2]}"
    ((first <= last)) || die "stage range must be forward: $STAGE"
  else
    die "invalid stage '$STAGE'; expected figure1..figure5 or a forward range"
  fi
  SELECTED_STAGES=()
  local number
  for ((number=first; number<=last; number++)); do
    SELECTED_STAGES+=("figure${number}")
  done
}

parse_stage_range
if [[ -n "$FORCE_STAGE" ]]; then
  stage_number "$FORCE_STAGE" >/dev/null || die "invalid --force-stage value: $FORCE_STAGE"
  force_number="$(stage_number "$FORCE_STAGE")"
  selected_first="$(stage_number "${SELECTED_STAGES[0]}")"
  selected_last="$(stage_number "${SELECTED_STAGES[-1]}")"
  ((force_number >= selected_first && force_number <= selected_last)) || \
    die "--force-stage must be included in --stage selection"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RESUME_JSON="$PROJECT_ROOT/data/resume/qwen_summaries_with_race.json"
DISCRIM_JSON="$PROJECT_ROOT/data/discrim-eval/dataset_paired.json"
ADULT_JSON="$PROJECT_ROOT/data/adult_datasets/adult_race_paired.json"
COMPAS_JSON="$PROJECT_ROOT/data/compas/compas_paired.json"
STAGE_ROOT="$RESULT_ROOT/stages"
FIG1="$RESULT_ROOT/figure1"
FIG2="$RESULT_ROOT/figure2"
FIG3="$RESULT_ROOT/figure3"
FIG4="$RESULT_ROOT/figure4"
FIG5="$RESULT_ROOT/figure5"

RANKING="$FIG1/biased_samples/biased_samples_ranking_summary_only.csv"
HEADS_DIR="$FIG1/components/heads"
SELECTED_HEADS="$HEADS_DIR/selected_heads_elbow.json"
HEAD_EMBEDDINGS="$HEADS_DIR/results.pkl"
MLP_IDENTIFICATION="$FIG1/components/mlps/identification"
SELECTED_MLPS="$FIG1/components/mlps/selected/selected_mlp_layers_elbow.json"
MLP_MEANS="$FIG1/components/mlps/mlp_means_resume.pkl"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

run_python_snippet() {
  local label="$1"; shift
  printf '+ python-snippet %q' "$label"
  printf ' %q' "$@"
  printf '\n'
}

valid_model() {
  "$PY" - "$MODEL_PATH" <<'PY' >/dev/null 2>&1
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
assert config.get("model_type") == "llama"
assert int(config.get("num_hidden_layers", -1)) == 32
assert int(config.get("num_attention_heads", -1)) == 32
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 4
assert all((root / shard).is_file() and (root / shard).stat().st_size > 0 for shard in shards)
for name in ("tokenizer.json", "tokenizer_config.json"):
    assert (root / name).is_file()
PY
}

ensure_model() {
  if valid_model; then
    echo "ModelScope checkpoint validated: $MODEL_PATH"
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    run_python_snippet "modelscope.snapshot_download" "$MODEL_ID" "$MODEL_PATH"
    return
  fi
  mkdir -p "$(dirname "$MODEL_PATH")"
  "$PY" - "$MODEL_ID" "$MODEL_PATH" <<'PY'
import sys
from modelscope import snapshot_download

model_id, local_dir = sys.argv[1:]
path = snapshot_download(model_id=model_id, local_dir=local_dir)
print(f"ModelScope checkpoint ready: {path}")
PY
  valid_model || die "downloaded checkpoint failed Llama 3 8B integrity checks"
}

stage_required_files() {
  case "$1" in
    figure1) printf '%s\n' "$RANKING" "$SELECTED_HEADS" "$HEAD_EMBEDDINGS" "$SELECTED_MLPS" "$MLP_MEANS" "$FIG1/figures/figure1.pdf" ;;
    figure2) printf '%s\n' "$FIG2/head_resume/sensitive/intervention_results_by_head_count.csv" "$FIG2/head_resume/random_seed_42/intervention_results_by_head_count_random.csv" "$FIG2/mlp_resume/per_sample.csv" "$FIG2/figures/figure2.pdf" ;;
    figure3) printf '%s\n' "$FIG3/exp20/mlp_mean_diff_p_yes.npy" "$FIG3/head_logit/mean_diff_p_yes.npy" "$FIG3/attention/fixed/qk_scores_full.json" "$FIG3/figures/figure3.pdf" ;;
    figure4) printf '%s\n' \
      "$FIG4/discrim/baseline.csv" \
      "$FIG4/discrim/head_all/sensitive/per_sample.csv" \
      "$FIG4/discrim/head_all/random_seed_42/per_sample.csv" \
      "$FIG4/discrim/head_count/sensitive/results.csv" \
      "$FIG4/discrim/head_count/random_seed_42/results.csv" \
      "$FIG4/compas/high_gap/evaluation/summary.csv" \
      "$FIG4/adult/high_gap/evaluation/summary.csv" \
      "$FIG4/figures/figure4.pdf" ;;
    figure5) printf '%s\n' \
      "$FIG5/adapters/global/final_model/adapter_model.safetensors" \
      "$FIG5/adapters/pfairft/final_model/adapter_model.safetensors" \
      "$FIG5/adapters/pfairft_kl/final_model/adapter_model.safetensors" \
      "$FIG5/adapters/pfairft_ce/final_model/adapter_model.safetensors" \
      "$FIG5/downstream/discrim_baseline.csv" \
      "$FIG5/downstream/discrim_debiased_prompt.csv" \
      "$FIG5/downstream/discrim_global.csv" \
      "$FIG5/downstream/discrim_pfairft.csv" \
      "$FIG5/downstream/discrim_pfairft_kl.csv" \
      "$FIG5/downstream/discrim_pfairft_ce.csv" \
      "$FIG5/context/context_results.json" \
      "$FIG5/inference_time/discrim_partial.csv" \
      "$FIG5/head_conditions/metadata.json" \
      "$FIG5/activation_geometry/metadata.json" \
      "$FIG5/snapshot/manifest.json" \
      "$FIG5/figures/figure5.pdf" ;;
  esac
}

stage_complete() {
  local stage="$1" file
  [[ -s "$STAGE_ROOT/$stage.json" ]] || return 1
  while IFS= read -r file; do
    [[ -s "$file" ]] || return 1
  done < <(stage_required_files "$stage")
}

require_previous_stages() {
  local stage="$1" number previous
  number="$(stage_number "$stage")"
  for ((previous=1; previous<number; previous++)); do
    if stage_complete "figure${previous}"; then
      continue
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      local selected
      for selected in "${SELECTED_STAGES[@]}"; do
        [[ "$selected" == "figure${previous}" ]] && continue 2
      done
    fi
    die "$stage requires a validated figure${previous} stage; run it first"
  done
}

archive_forced_stages() {
  [[ -n "$FORCE_STAGE" ]] || return 0
  local start timestamp stale number path
  start="$(stage_number "$FORCE_STAGE")"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  stale="$RESULT_ROOT/stale/${FORCE_STAGE}_and_downstream_$timestamp"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ archive figure${start}..figure5 under $stale"
    return
  fi
  mkdir -p "$stale/stages"
  for ((number=start; number<=5; number++)); do
    path="$RESULT_ROOT/figure${number}"
    [[ -e "$path" ]] && mv "$path" "$stale/"
    [[ -e "$STAGE_ROOT/figure${number}.json" ]] && mv "$STAGE_ROOT/figure${number}.json" "$stale/stages/"
  done
}

guard_partial_stage() {
  local stage="$1" directory="$RESULT_ROOT/$1"
  stage_complete "$stage" && return
  if [[ -d "$directory" && -n "$(find "$directory" -mindepth 1 -print -quit)" ]]; then
    die "$stage has incomplete output at $directory; rerun with --force-stage $stage"
  fi
}

mark_stage() {
  local stage="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ write validated stage manifest $STAGE_ROOT/$stage.json"
    return
  fi
  mkdir -p "$STAGE_ROOT"
  local required_files=()
  mapfile -t required_files < <(stage_required_files "$stage")
  "$PY" - "$stage" "$STAGE_ROOT/$stage.json" "$MODEL_ID" "$MODEL_PATH" "$RESULT_ROOT" "${required_files[@]}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

stage, output, model_id, model_path, result_root, *required = sys.argv[1:]
artifacts = []
for value in required:
    path = Path(value)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required stage artifact is missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    artifacts.append(
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    )
payload = {
    "stage": stage,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "model_path": str(Path(model_path).resolve()),
    "result_root": str(Path(result_root).resolve()),
    "seed": 42,
    "resume_prompt_mode": "summary_only",
    "method_mapping": {
        "PFairFT": "fairness_kl",
        "PFairFT-KL": "fairness_kl_ce",
        "PFairFT-CE": "fairness_ce",
    },
    "artifacts": artifacts,
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

compute_head_counts() {
  "$PY" - "$SELECTED_HEADS" <<'PY'
import json, sys
count = len(json.load(open(sys.argv[1], encoding="utf-8")))
values = [0] + [int(i * count / 5 + 0.5) for i in range(1, 5)] + [count]
print(" ".join(map(str, dict.fromkeys(values))))
PY
}

plot_figure() {
  local number="$1"
  run_cmd "$PY" nmi_plot/llama3_8b/plot_figures.py \
    --figure "$number" --result-root "$RESULT_ROOT" \
    --model-name "$MODEL_NAME" --output-dir "$RESULT_ROOT/figure${number}/figures"
}

run_figure1() {
  run_cmd mkdir -p "$FIG1/biased_samples" "$HEADS_DIR" "$MLP_IDENTIFICATION"
  run_cmd "$PY" 2_component_identification/evaluate_biased_sample.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --output_csv_path "$RANKING" \
    --resume_prompt_mode summary_only
  run_cmd "$PY" 2_component_identification/analyze_race_sensitive_heads_moefreeze.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --output_dir "$HEADS_DIR" \
    --sample_csv_path "$RANKING" --sample_size 100 --no-balanced \
    --resume_prompt_mode summary_only --batch_size 1
  run_cmd "$PY" 2_component_identification/analyze_race_sensitive_MLPs.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --output_dir "$MLP_IDENTIFICATION" \
    --sample_csv_path "$RANKING" --sample_size 100 --no-balanced --batch_size 1
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/select_race_sensitive_MLPs.py \
    --results_path "$MLP_IDENTIFICATION/results_mlp.pkl" \
    --output_dir "$(dirname "$SELECTED_MLPS")"
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/collect_race_mean_MLPs_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --output_path "$MLP_MEANS" \
    --sample_csv_path "$RANKING" --sample_size 100 --no-balanced \
    --resume_prompt_mode summary_only --batch_size 1 --seed 42
  plot_figure 1
}

run_figure2() {
  local counts_string="<derived-from-selected-heads>" counts=()
  if [[ "$DRY_RUN" == "0" ]]; then
    counts_string="$(compute_head_counts)"
    read -r -a counts <<< "$counts_string"
  else
    counts=(0 10 20 30 40 ALL_SELECTED)
  fi
  echo "Figure 2 head-count grid: $counts_string"
  run_cmd "$PY" 2_component_identification/evaluate_intervention_by_head_count.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --sample_csv_path "$RANKING" --sample_size 100 \
    --resume_prompt_mode summary_only --sensitive_heads_dir "$HEADS_DIR" \
    --batch_size 1 --seed 42 --output_dir "$FIG2/head_resume/sensitive" \
    --intervention_type negative --head_counts "${counts[@]}"
  run_cmd "$PY" 2_component_identification/evaluate_intervention_by_head_count_random.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --sample_csv_path "$RANKING" --sample_size 100 \
    --resume_prompt_mode summary_only --sensitive_heads_dir "$HEADS_DIR" \
    --batch_size 1 --seed 42 --output_dir "$FIG2/head_resume/random_seed_42" \
    --results_csv_name intervention_results_by_head_count_random.csv --head_counts "${counts[@]}"
  run_cmd "$PY" 4_intervention_ablation/mlp_intervention/evaluate_intervention_MLP_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --sample_csv_path "$RANKING" --sample_size 100 \
    --resume_prompt_mode summary_only --sensitive_mlp_path "$SELECTED_MLPS" \
    --mlp_embeddings_path "$MLP_MEANS" --seed 42 --output_dir "$FIG2/mlp_resume" \
    --csv_path "$FIG2/mlp_resume/per_sample.csv"
  plot_figure 2
}

run_figure3() {
  run_cmd "$PY" 3_pattern_analysis/head_logit_analysis/analyze_head_kl_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --biased_csv_path "$RANKING" \
    --sample_size 100 --batch_size 1 --output_dir "$FIG3/head_logit"
  run_cmd "$PY" 3_pattern_analysis/mlp_analysis/analyze_mlp_output_kl_resume.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --biased_csv_path "$RANKING" \
    --sample_size 100 --batch_size 1 --output_dir "$FIG3/exp20"
  run_cmd "$PY" 3_pattern_analysis/mlp_analysis/analyze_mlp_output_kl_resume_with_intervention.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --biased_csv_path "$RANKING" \
    --sensitive_heads_path "$SELECTED_HEADS" --embeddings_path "$HEAD_EMBEDDINGS" \
    --sample_size 100 --batch_size 1 --output_dir "$FIG3/exp20"
  run_cmd "$PY" 3_pattern_analysis/head_logit_analysis/analyze_head_kl_resume_mlp.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --dataset_json_path "$RESUME_JSON" --biased_csv_path "$RANKING" \
    --sample_size 100 --mlp_selected_path "$SELECTED_MLPS" --batch_size 1 \
    --output_dir "$FIG3/head_logit_after_mlp"
  run_cmd "$PY" 3_pattern_analysis/head_attention_pattern/analyze_qk_scores.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --sensitive_heads_path "$SELECTED_HEADS" --sample_csv_path "$RANKING" \
    --dataset_json_path "$RESUME_JSON" --prompt_source fixed --output_dir "$FIG3/attention/fixed"
  run_cmd "$PY" 3_pattern_analysis/head_attention_pattern/visualize_qk_scores.py \
    --qk_scores_json "$FIG3/attention/fixed/qk_scores_full.json" \
    --model_path "$MODEL_PATH" --output_dir "$FIG3/attention/fixed/plots"
  plot_figure 3
}

run_figure4() {
  local counts_string="<derived-from-selected-heads>" counts=()
  if [[ "$DRY_RUN" == "0" ]]; then
    counts_string="$(compute_head_counts)"; read -r -a counts <<< "$counts_string"
  else
    counts=(0 10 20 30 40 ALL_SELECTED)
  fi
  run_cmd "$PY" 6_downstream_evaluation/evaluate_models_discrim.py \
    --dataset_path "$DISCRIM_JSON" --base_model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --mode baseline --csv_path "$FIG4/discrim/baseline.csv" --model_name_suffix baseline
  local mode label
  for mode in negative negative_random; do
    [[ "$mode" == "negative" ]] && label=sensitive || label=random_seed_42
    run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_discrim_eval.py \
      --dataset_path "$DISCRIM_JSON" --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --sensitive_heads_dir "$HEADS_DIR" --intervention_mode "$mode" --seed 42 \
      --output_dir "$FIG4/discrim/head_all/$label" \
      --csv_path "$FIG4/discrim/head_all/$label/per_sample.csv"
    run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_discrim_eval_head_count.py \
      --dataset_path "$DISCRIM_JSON" --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
      --sensitive_heads_dir "$HEADS_DIR" --intervention_mode "$mode" --seed 42 --batch_size 1 \
      --output_dir "$FIG4/discrim/head_count/$label" --results_csv_name results.csv \
      --head_counts "${counts[@]}"
  done
  run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_compas_full.py \
    --dataset_path "$COMPAS_JSON" --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --sensitive_heads_dir "$HEADS_DIR" --selected_mlp_path "$SELECTED_MLPS" \
    --mlp_embeddings_path "$MLP_MEANS" --seed 42 --batch_size 1 --resume \
    --output_dir "$FIG4/compas/full"
  run_cmd "$PY" data/compas/select_high_gap_pairs.py \
    --dataset_path "$COMPAS_JSON" --full_result_dir "$FIG4/compas/full" \
    --output_dir "$FIG4/compas/high_gap" --top_k 100 --curve_k 50 100 200 500 1000
  run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_compas_full.py \
    --dataset_path "$FIG4/compas/high_gap/selected_top_100.json" \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --sensitive_heads_dir "$FIG4/compas/high_gap/component_snapshot/heads" \
    --selected_mlp_path "$FIG4/compas/high_gap/component_snapshot/mlps/selected_mlp_layers_elbow.json" \
    --mlp_embeddings_path "$FIG4/compas/high_gap/component_snapshot/mlps/mlp_means_resume.pkl" \
    --seed 42 --batch_size 1 --resume --output_dir "$FIG4/compas/high_gap/evaluation"
  run_cmd "$PY" data/adult_datasets/evaluate_fairness_score.py \
    --dataset_path "$ADULT_JSON" --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" --batch_size 1 --resume \
    --output_dir "$FIG4/adult/full"
  run_cmd "$PY" data/adult_datasets/select_high_gap_pairs.py \
    --dataset_path "$ADULT_JSON" --baseline_dir "$FIG4/adult/full" \
    --output_dir "$FIG4/adult/high_gap" --top_k 100 \
    --sensitive_heads_dir "$HEADS_DIR" --selected_mlp_path "$SELECTED_MLPS" \
    --mlp_embeddings_path "$MLP_MEANS"
  run_cmd "$PY" 4_intervention_ablation/head_intervention/evaluate_intervention_adult_race.py \
    --dataset_path "$FIG4/adult/high_gap/selected_top_100.json" \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --model_name "$MODEL_NAME" \
    --sensitive_heads_dir "$FIG4/adult/high_gap/component_snapshot/heads" \
    --selected_mlp_path "$FIG4/adult/high_gap/component_snapshot/mlps/selected_mlp_layers_elbow.json" \
    --mlp_embeddings_path "$FIG4/adult/high_gap/component_snapshot/mlps/mlp_means_resume.pkl" \
    --seed 42 --batch_size 1 --run_head_sweep --resume \
    --output_dir "$FIG4/adult/high_gap/evaluation"
  plot_figure 4
}

train_precision() {
  local loss_type="$1" output="$2"
  run_cmd "$PY" 5_finetuning/finetune_precision_fairness.py \
    --model_path "$MODEL_PATH" --dataset_json_path "$RESUME_JSON" \
    --heads_analysis_dir "$HEADS_DIR" --output_dir "$output" \
    --sample_csv_path "$RANKING" --sample_size 0 --max_samples 0 --no-balanced \
    --resume_prompt_mode summary_only --loss_type "$loss_type" --num_epochs 3 \
    --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5 --warmup_steps 0
}

eval_discrim() {
  local mode="$1" adapter="$2" output="$3" prompt_type="${4:-prompt}"
  local args=("$PY" 6_downstream_evaluation/evaluate_models_discrim.py
    --dataset_path "$DISCRIM_JSON" --base_model_path "$MODEL_PATH" --model_type "$MODEL_TYPE"
    --mode "$mode" --csv_path "$output" --model_name_suffix "$mode" --prompt_type "$prompt_type")
  [[ -n "$adapter" ]] && args+=(--adapter_path "$adapter")
  run_cmd "${args[@]}"
}

run_figure5() {
  local global="$FIG5/adapters/global"
  local pfairft="$FIG5/adapters/pfairft"
  local pfairft_kl="$FIG5/adapters/pfairft_kl"
  local pfairft_ce="$FIG5/adapters/pfairft_ce"
  run_cmd "$PY" 5_finetuning/finetune_global_lora.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_json_path "$RESUME_JSON" \
    --output_dir "$global" --sample_csv_path "$RANKING" --sample_size 0 --max_samples 0 \
    --no-balanced --resume_prompt_mode summary_only --num_epochs 3 --batch_size 1 \
    --gradient_accumulation_steps 16 --learning_rate 2e-5 --warmup_steps 0
  train_precision fairness_kl "$pfairft"
  train_precision fairness_kl_ce "$pfairft_kl"
  train_precision fairness_ce "$pfairft_ce"
  eval_discrim baseline "" "$FIG5/downstream/discrim_baseline.csv"
  eval_discrim debiased_prompt "" "$FIG5/downstream/discrim_debiased_prompt.csv" debiased_prompt
  eval_discrim global "$global/final_model" "$FIG5/downstream/discrim_global.csv"
  eval_discrim pfairft "$pfairft/final_model" "$FIG5/downstream/discrim_pfairft.csv"
  eval_discrim pfairft_kl "$pfairft_kl/final_model" "$FIG5/downstream/discrim_pfairft_kl.csv"
  eval_discrim pfairft_ce "$pfairft_ce/final_model" "$FIG5/downstream/discrim_pfairft_ce.csv"
  run_cmd "$PY" 1_bias_evaluation/evaluate_with_context_llama.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_path "$DISCRIM_JSON" \
    --output_path "$FIG5/context/context_results.json" --target_qids 40 12 94 --batch_size 1
  run_cmd "$PY" 4_intervention_ablation/projection_intervention/evaluate_intervention_all_heads_discrim.py \
    --model_path "$MODEL_PATH" --model_type "$MODEL_TYPE" --dataset_path "$DISCRIM_JSON" \
    --sensitive_heads_dir "$HEADS_DIR" --intervention_mode partial --intervention_strength 1.0 \
    --seed 42 --output_dir "$FIG5/inference_time" --csv_path "$FIG5/inference_time/discrim_partial.csv"
  run_cmd "$PY" 3_pattern_analysis/model_comparison/analyze_figure5_head_conditions.py \
    --base_model_path "$MODEL_PATH" --dataset_path "$DISCRIM_JSON" \
    --selected_heads_json "$SELECTED_HEADS" --baseline_csv "$FIG5/downstream/discrim_baseline.csv" \
    --global_csv "$FIG5/downstream/discrim_global.csv" --pfairft_csv "$FIG5/downstream/discrim_pfairft.csv" \
    --global_adapter "$global/final_model" --pfairft_adapter "$pfairft/final_model" \
    --candidate_count 70 --panel_b_qid 90 --batch_size 1 --output_dir "$FIG5/head_conditions"
  run_cmd "$PY" 3_pattern_analysis/model_comparison/analyze_figure5_activation_geometry.py \
    --base_model_path "$MODEL_PATH" --pfairft_adapter "$pfairft/final_model" \
    --heads_dir "$HEADS_DIR" --resume_dataset "$RESUME_JSON" --resume_ranking_csv "$RANKING" \
    --discrim_dataset "$DISCRIM_JSON" --resume_sample_size 100 --batch_size 1 \
    --output_dir "$FIG5/activation_geometry"
  run_cmd "$PY" nmi_plot/figure5/prepare_figure5_data.py \
    --project_root "$PROJECT_ROOT" --model_name "$MODEL_NAME" --result_root "$FIG5" \
    --output_dir "$FIG5/snapshot" --global_csv "$FIG5/downstream/discrim_global.csv" \
    --global_adapter "$global/final_model" --head_analysis_dir "$FIG5/head_conditions" \
    --activation_geometry_dir "$FIG5/activation_geometry" \
    --context_results "$FIG5/context/context_results.json" \
    --selected_heads "$SELECTED_HEADS" --heads_results "$HEAD_EMBEDDINGS"
  run_cmd "$PY" nmi_plot/figure5/plot_figure5.py \
    --data-dir "$FIG5/snapshot" --output-dir "$FIG5/figures" --single-model
}

cd "$PROJECT_ROOT"
echo "Model: $MODEL_ID"
echo "Checkpoint: $MODEL_PATH"
echo "Results: $RESULT_ROOT"
echo "Stages: ${SELECTED_STAGES[*]} | GPU=$GPU | dry_run=$DRY_RUN"
ensure_model
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
archive_forced_stages

for stage in "${SELECTED_STAGES[@]}"; do
  require_previous_stages "$stage"
  if stage_complete "$stage"; then
    echo "Reusing validated stage: $stage"
    continue
  fi
  guard_partial_stage "$stage"
  echo "===== Running $stage ====="
  case "$stage" in
    figure1) run_figure1 ;;
    figure2) run_figure2 ;;
    figure3) run_figure3 ;;
    figure4) run_figure4 ;;
    figure5) run_figure5 ;;
  esac
  mark_stage "$stage"
done

echo "Selected Llama 3 8B Figure workflow completed."
