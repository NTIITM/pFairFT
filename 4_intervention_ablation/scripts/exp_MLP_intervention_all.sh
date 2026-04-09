#!/usr/bin/env bash
set -euo pipefail

# 一键运行 exp15: 基于 MLP 的统一 mean ablation 干预实验
# 步骤（对每个模型）：
# 1) 若无现成 MLP 敏感度结果，则调用 exp2 的分析脚本生成 results_mlp.pkl
# 2) 用 exp15/select_race_sensitive_MLPs.py 做肘部点选择，得到 selected_mlp_layers_elbow.json
# 3) 用 exp15/collect_race_mean_MLPs_resume.py 在 Resume 上收集 MLP 均值
# 4) 用 exp15/evaluate_intervention_MLP_discrim_eval.py 在 discrim-eval 上做 MLP mean ablation 干预
# 5) 用 exp15/evaluate_intervention_MLP_resume.py 在 Resume 上做 MLP mean ablation 干预
# 6) （可选）统一画图：exp15/plot_intervention_qwen_llama_grid_with_mlp.py

PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
EXP2_DIR="${PROJECT_ROOT}/exp2_old"
EXP8_DIR="${PROJECT_ROOT}/exp8"
EXP15_DIR="${PROJECT_ROOT}/exp15"

# 模型根目录（与原有脚本保持一致）
LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

# 数据路径
RESUME_JSON="${PROJECT_ROOT}/data/resume/qwen_summaries_with_race.json"
DISCRIM_EVAL_JSON="${PROJECT_ROOT}/data/discrim-eval/dataset_paired.json"

# Python 脚本路径
PY_MLP_ANALYSIS="${EXP2_DIR}/analyze_race_sensitive_MLPs.py"
PY_MLP_ELBOW="${EXP15_DIR}/select_race_sensitive_MLPs.py"
PY_COLLECT_MLP_RESUME="${EXP15_DIR}/collect_race_mean_MLPs_resume.py"
PY_EVAL_MLP_DISCRIM="${EXP15_DIR}/evaluate_intervention_MLP_discrim_eval.py"
PY_EVAL_MLP_RESUME="${EXP15_DIR}/evaluate_intervention_MLP_resume.py"
PY_PLOT_MLP="${EXP15_DIR}/plot_intervention_qwen_llama_grid_with_mlp.py"

# 聚合所有模型的 MLP 干预结果（discrim-eval / resume）
CSV_MLP_DISCRIM="${EXP15_DIR}/per_sample_intervention_mlp_negative_discrim_all_models.csv"
CSV_MLP_RESUME="${EXP15_DIR}/per_sample_intervention_mlp_negative_resume_all_models.csv"

# 仅使用原始 prompt；如需 debiased_prompt 可自行扩展
PROMPT_TYPE="prompt"

# GPU configuration (edit as needed)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

TASKS=()

# 收集任务（以“每个模型完整跑完 Step1-5”为一个 task）
for MODEL_DIR in "${LLM_RESEARCH_DIR}"/* "${QWEN_DIR}"/*; do
  if [ ! -d "${MODEL_DIR}" ]; then
    continue
  fi

  MODEL_NAME="$(basename "${MODEL_DIR}")"

  # 每个模型单独输出 CSV，避免并行写冲突
  PER_MODEL_CSV_DISCRIM="${EXP15_DIR}/tmp_per_model/per_sample_intervention_mlp_negative_discrim_${MODEL_NAME}.csv"
  PER_MODEL_CSV_RESUME="${EXP15_DIR}/tmp_per_model/per_sample_intervention_mlp_negative_resume_${MODEL_NAME}.csv"

  # 使用 \$ 转义，防止主脚本在构造字符串时展开内部变量
  TASKS+=("bash -lc 'set -euo pipefail; \
PROJECT_ROOT=\"${PROJECT_ROOT}\"; \
EXP2_DIR=\"${EXP2_DIR}\"; \
EXP15_DIR=\"${EXP15_DIR}\"; \
LLM_RESEARCH_DIR=\"${LLM_RESEARCH_DIR}\"; \
QWEN_DIR=\"${QWEN_DIR}\"; \
RESUME_JSON=\"${RESUME_JSON}\"; \
DISCRIM_EVAL_JSON=\"${DISCRIM_EVAL_JSON}\"; \
PY_MLP_ANALYSIS=\"${PY_MLP_ANALYSIS}\"; \
PY_MLP_ELBOW=\"${PY_MLP_ELBOW}\"; \
PY_COLLECT_MLP_RESUME=\"${PY_COLLECT_MLP_RESUME}\"; \
PY_EVAL_MLP_DISCRIM=\"${PY_EVAL_MLP_DISCRIM}\"; \
PY_EVAL_MLP_RESUME=\"${PY_EVAL_MLP_RESUME}\"; \
PROMPT_TYPE=\"${PROMPT_TYPE}\"; \
MODEL_DIR=\"${MODEL_DIR}\"; \
MODEL_NAME=\"${MODEL_NAME}\"; \
PER_MODEL_CSV_DISCRIM=\"${PER_MODEL_CSV_DISCRIM}\"; \
PER_MODEL_CSV_RESUME=\"${PER_MODEL_CSV_RESUME}\"; \

mkdir -p \"${EXP15_DIR}/tmp_per_model\"; \
rm -f \"\$PER_MODEL_CSV_DISCRIM\" \"\$PER_MODEL_CSV_RESUME\"; \

echo \"============================================================\"; \
echo \"Processing model: \$MODEL_NAME\"; \
echo \"Model path: \$MODEL_DIR\"; \
echo \"============================================================\"; \

MLP_ANALYSIS_DIR=\"${EXP2_DIR}/sensitive_MLPs_\${MODEL_NAME}_top100\"; \
RESULTS_MLP_PKL=\"\$MLP_ANALYSIS_DIR/results_mlp.pkl\"; \

if [ ! -f \"\$RESULTS_MLP_PKL\" ]; then \
  echo \"[Step 1] MLP sensitivity results not found, running analysis for \$MODEL_NAME\"; \
  BIASED_DIR=\"${EXP2_DIR}/biased_samples_\${MODEL_NAME}\"; \
  CSV_PATH=\"\$BIASED_DIR/biased_samples_ranking.csv\"; \
  if [ ! -f \"\$CSV_PATH\" ]; then \
    echo \"  WARNING: biased_samples CSV not found: \$CSV_PATH\"; \
    echo \"  Skip all MLP steps for this model.\"; \
    exit 0; \
  fi; \
  mkdir -p \"\$MLP_ANALYSIS_DIR\"; \
  python \"\$PY_MLP_ANALYSIS\" \
    --model_path \"\$MODEL_DIR\" \
    --dataset_json_path \"\$RESUME_JSON\" \
    --sample_csv_path \"\$CSV_PATH\" \
    --sample_size 100 \
    --output_dir \"\$MLP_ANALYSIS_DIR\" \
    --device cuda \
    --model_type auto; \
else \
  echo \"[Step 1] MLP sensitivity results already exist: \$RESULTS_MLP_PKL\"; \
fi; \

MLP_ELBOW_DIR=\"${EXP15_DIR}/mlp_elbow_\${MODEL_NAME}\"; \
SELECTED_MLP_JSON=\"\$MLP_ELBOW_DIR/selected_mlp_layers_elbow.json\"; \
if [ ! -f \"\$SELECTED_MLP_JSON\" ]; then \
  echo \"[Step 2] Running elbow selection for MLP layers: \$MODEL_NAME\"; \
  mkdir -p \"\$MLP_ELBOW_DIR\"; \
  python \"\$PY_MLP_ELBOW\" \
    --results_path \"\$RESULTS_MLP_PKL\" \
    --output_dir \"\$MLP_ELBOW_DIR\"; \
else \
  echo \"[Step 2] Elbow selection already exists: \$SELECTED_MLP_JSON\"; \
fi; \

MLP_MEAN_RESUME_PKL=\"${EXP15_DIR}/mlp_means_resume_\${MODEL_NAME}.pkl\"; \
if [ ! -f \"\$MLP_MEAN_RESUME_PKL\" ]; then \
  echo \"[Step 3] Collecting Resume-based MLP race means for \$MODEL_NAME\"; \
  python \"\$PY_COLLECT_MLP_RESUME\" \
    --model_path \"\$MODEL_DIR\" \
    --dataset_json_path \"\$RESUME_JSON\" \
    --output_path \"\$MLP_MEAN_RESUME_PKL\" \
    --max_samples 500 \
    --batch_size 8 \
    --device cuda \
    --model_type auto; \
else \
  echo \"[Step 3] Resume MLP means already exist: \$MLP_MEAN_RESUME_PKL\"; \
fi; \

echo \"[Step 4] Running MLP negative intervention on discrim-eval for \$MODEL_NAME\"; \
python \"\$PY_EVAL_MLP_DISCRIM\" \
  --dataset_path \"\$DISCRIM_EVAL_JSON\" \
  --model_path \"\$MODEL_DIR\" \
  --model_type auto \
  --device cuda \
  --prompt_type \"\$PROMPT_TYPE\" \
  --output_dir \"${EXP15_DIR}/intervention_mlp_discrim_eval_\${MODEL_NAME}\" \
  --csv_path \"\$PER_MODEL_CSV_DISCRIM\" \
  --sensitive_mlp_path \"\$SELECTED_MLP_JSON\" \
  --mlp_embeddings_path \"\$MLP_MEAN_RESUME_PKL\" \
  --intervention_type mlp_negative; \

echo \"[Step 5] Running MLP negative intervention on Resume for \$MODEL_NAME\"; \
python \"\$PY_EVAL_MLP_RESUME\" \
  --dataset_json_path \"\$RESUME_JSON\" \
  --model_path \"\$MODEL_DIR\" \
  --model_type auto \
  --device cuda \
  --output_dir \"${EXP15_DIR}/intervention_mlp_resume_\${MODEL_NAME}\" \
  --csv_path \"\$PER_MODEL_CSV_RESUME\" \
  --sensitive_mlp_path \"\$SELECTED_MLP_JSON\" \
  --mlp_embeddings_path \"\$MLP_MEAN_RESUME_PKL\" \
  --intervention_type mlp_negative; \

echo \"Finished all MLP steps for model: \$MODEL_NAME\"'"
  )

done

echo "Total TASKS collected: ${#TASKS[@]}"

# 清理旧聚合 CSV（若存在）
rm -f "${CSV_MLP_DISCRIM}" "${CSV_MLP_RESUME}"

# Scheduler
declare -A GPU_TO_PID_MAP
idx=0
num_tasks=${#TASKS[@]}

while [ $idx -lt $num_tasks ]; do
  for gpu in "${GPUS[@]}"; do
    if [ $idx -ge $num_tasks ]; then
      break
    fi

    is_free=false
    if [[ ! -v GPU_TO_PID_MAP[$gpu] ]]; then
      is_free=true
    else
      pid="${GPU_TO_PID_MAP[$gpu]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        unset GPU_TO_PID_MAP[$gpu]
        is_free=true
      fi
    fi

    if [ "$is_free" = true ]; then
      cmd="${TASKS[$idx]}"
      echo "[Task $((idx+1))/$num_tasks] Running on GPU $gpu"
      CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" &
      GPU_TO_PID_MAP[$gpu]=$!
      idx=$((idx + 1))
      sleep 1
    fi
  done

  if [ $idx -lt $num_tasks ]; then
    sleep 5
  fi
done

# Wait for all remaining processes
echo "All tasks dispatched. Waiting for remaining processes to finish..."
for gpu in "${GPUS[@]}"; do
  if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
    pid="${GPU_TO_PID_MAP[$gpu]}"
    wait "$pid" 2>/dev/null || true
  fi
done

# 合并 CSV
mkdir -p "${EXP15_DIR}/tmp_per_model"
first=1
for f in "${EXP15_DIR}/tmp_per_model"/per_sample_intervention_mlp_negative_discrim_*.csv; do
  [ -f "$f" ] || continue
  if [ $first -eq 1 ]; then cat "$f" >> "${CSV_MLP_DISCRIM}"; first=0; else tail -n +2 "$f" >> "${CSV_MLP_DISCRIM}"; fi
done

first=1
for f in "${EXP15_DIR}/tmp_per_model"/per_sample_intervention_mlp_negative_resume_*.csv; do
  [ -f "$f" ] || continue
  if [ $first -eq 1 ]; then cat "$f" >> "${CSV_MLP_RESUME}"; first=0; else tail -n +2 "$f" >> "${CSV_MLP_RESUME}"; fi
done

echo "============================================================"
echo "All models processed."
echo "  - discrim-eval MLP intervention CSV: ${CSV_MLP_DISCRIM}"
echo "  - resume MLP intervention CSV:       ${CSV_MLP_RESUME}"
echo "============================================================"
