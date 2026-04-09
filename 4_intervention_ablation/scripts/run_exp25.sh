#!/bin/bash

# exp25 一键启动脚本
# 功能：参考 exp20 调度逻辑，针对所有模型并行运行 discrim-eval 和 resume 任务

PROJECT_ROOT="/home/common1/hwluo/project/pFairFT"
EXP_DIR="$PROJECT_ROOT/exp25"
EXP2_DIR="${PROJECT_ROOT}/exp2_old"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

LLM_RESEARCH_DIR="/mnt/nfs/huggingface/LLM-Research"
QWEN_DIR="/mnt/nfs/huggingface/Qwen"

RESUME_JSON="${PROJECT_ROOT}/data/resume/qwen_summaries_with_race.json"

MODELS=(
    "Qwen3-1.7B"
    "Qwen3-4B"
    "Qwen3-8B"
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Meta-Llama-3-8B-Instruct"
)

GPUS=(0 1 2 3 5 6 7)

TASKS=()

for MODEL_NAME in "${MODELS[@]}"; do
    if [[ "${MODEL_NAME}" == Qwen3-* ]]; then
        MODEL_PATH="${QWEN_DIR}/${MODEL_NAME}"
    else
        MODEL_PATH="${LLM_RESEARCH_DIR}/${MODEL_NAME}"
    fi

    if [ ! -d "${MODEL_PATH}" ]; then
        echo "[SKIP] Missing model dir: ${MODEL_PATH}"
        continue
    fi

    OUT_DIR="$EXP_DIR/results_$MODEL_NAME"
    mkdir -p "$OUT_DIR"

    # --- 任务 1: Discrim-Eval ---
    for MODE in "all" "partial"; do
        CSV_PATH="$OUT_DIR/per_sample_intervention_${MODE}_heads.csv"
        if [ -f "$CSV_PATH" ]; then
            echo "[SKIP] Discrim-Eval ($MODEL_NAME, $MODE) already exists"
        else
            TASKS+=("python $EXP_DIR/evaluate_intervention_all_heads_discrim.py --model_path $MODEL_PATH --intervention_mode $MODE --csv_path $CSV_PATH --output_dir $OUT_DIR")
        fi
    done

    # --- 任务 2: Resume Dataset ---
    BIASED_CSV="${EXP2_DIR}/biased_samples_${MODEL_NAME}/biased_samples_ranking.csv"
    for MODE in "all" "partial"; do
        CSV_PATH="$OUT_DIR/resume_top100_${MODE}.csv"
        if [ -f "$CSV_PATH" ]; then
            echo "[SKIP] Resume ($MODEL_NAME, $MODE) already exists"
        else
            if [ -f "$BIASED_CSV" ]; then
                TASKS+=("python $EXP_DIR/evaluate_intervention_all_heads_resume.py --base_model_path $MODEL_PATH --biased_csv_path $BIASED_CSV --intervention_mode $MODE --output_csv_path $CSV_PATH")
            else
                echo "[WARN] Missing biased CSV for $MODEL_NAME: $BIASED_CSV"
            fi
        fi
    done
done

echo "Total tasks: ${#TASKS[@]}"

# Scheduler logic from exp20
declare -A GPU_TO_PID_MAP
idx=0
num_tasks=${#TASKS[@]}

while [ $idx -lt $num_tasks ]; do
    for gpu in "${GPUS[@]}"; do
        if [ $idx -ge $num_tasks ]; then break; fi

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
            task_type=$(echo $cmd | awk '{print $2}' | xargs basename | cut -d'_' -f3)
            log_file="$LOG_DIR/${task_type}_${idx}.log"
            echo "[Task $((idx+1))/$num_tasks] GPU $gpu: $cmd"
            CUDA_VISIBLE_DEVICES="$gpu" $cmd > "$log_file" 2>&1 &
            GPU_TO_PID_MAP[$gpu]=$!
            idx=$((idx + 1))
            sleep 2
        fi
    done
    if [ $idx -lt $num_tasks ]; then sleep 5; fi
done

echo "Waiting for remaining tasks..."
for gpu in "${GPUS[@]}"; do
    if [[ -v GPU_TO_PID_MAP[$gpu] ]]; then
        pid="${GPU_TO_PID_MAP[$gpu]}"
        wait "$pid" 2>/dev/null || true
    fi
done

echo "All experiments finished. Running plot scripts..."
python "$EXP_DIR/plot_discrim_grid_exp25.py"
python "$EXP_DIR/plot_resume_fairness_exp25.py"

echo "Exp25 completed successfully."
