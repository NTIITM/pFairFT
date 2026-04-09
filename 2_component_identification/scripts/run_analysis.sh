#!/bin/bash
# 分析干预结果并生成折线图（敏感头 + 随机头 两条线）

OUTPUT_DIR="/home/common1/hwluo/project/pFairFT/exp9/intervention_results_qwen3-4b_top100"
CSV_SENSITIVE="${OUTPUT_DIR}/intervention_results_by_head_count.csv"
CSV_RANDOM="${OUTPUT_DIR}/intervention_results_by_head_count_random.csv"

# 若两个 CSV 都存在则画两条线，否则只画存在的那个
if [[ -f "$CSV_SENSITIVE" && -f "$CSV_RANDOM" ]]; then
  python /home/common1/hwluo/project/pFairFT/exp9/analyze_intervention_results.py \
    --csv_paths "$CSV_SENSITIVE" "$CSV_RANDOM" \
    --labels "Sensitive heads" "Random heads" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "mean_bias_by_head_count.png" \
    --title "Mean |fact_p_yes - cf_p_yes| by Head Count (Qwen3-4B)"
else
  # 单 CSV 兼容：只画存在的那一个
  if [[ -f "$CSV_SENSITIVE" ]]; then
    python /home/common1/hwluo/project/pFairFT/exp9/analyze_intervention_results.py \
      --csv_path "$CSV_SENSITIVE" \
      --output_dir "$OUTPUT_DIR" \
      --output_name "mean_bias_by_head_count.png" \
      --title "Mean |fact_p_yes - cf_p_yes| by Head Count (Qwen3-4B)"
  elif [[ -f "$CSV_RANDOM" ]]; then
    python /home/common1/hwluo/project/pFairFT/exp9/analyze_intervention_results.py \
      --csv_path "$CSV_RANDOM" \
      --output_dir "$OUTPUT_DIR" \
      --output_name "mean_bias_by_head_count.png" \
      --title "Mean |fact_p_yes - cf_p_yes| by Head Count (Qwen3-4B)"
  else
    echo "Error: No CSV found. Run run_evaluation.sh or run_evaluation_both.sh first."
    exit 1
  fi
fi
