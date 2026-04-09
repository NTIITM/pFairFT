#!/usr/bin/env bash
set -euo pipefail

# 实验根目录（exp5 目录）
EXP5_DIR="/home/common1/hwluo/project/pFairFT/exp5"

# 输出统计结果的文件
OUTPUT_FILE="${EXP5_DIR}/all_models_statistics.txt"
OUTPUT_CSV="${EXP5_DIR}/all_models_statistics.csv"

# 清空输出文件
> "$OUTPUT_FILE"
> "$OUTPUT_CSV"

# 写入 CSV 表头
echo "Model,Total_Samples,Black_Samples,White_Samples,Base_Mean_Bias,Base_Median_Bias,Base_Std_Bias,LoRA_Mean_Bias,LoRA_Median_Bias,LoRA_Std_Bias,Bias_Reduction,Bias_Reduction_Percent" >> "$OUTPUT_CSV"

echo "=========================================="
echo "统计所有模型的评估结果"
echo "=========================================="
echo ""

# 遍历所有评估结果目录
for EVAL_DIR in "${EXP5_DIR}"/eval_finetune_resume_results_*; do
  # 只处理目录
  if [ ! -d "$EVAL_DIR" ]; then
    continue
  fi

  # 提取模型名称
  MODEL_NAME="$(basename "$EVAL_DIR" | sed 's/^eval_finetune_resume_results_//')"
  
  # CSV 文件路径
  CSV_FILE="${EVAL_DIR}/finetune_discrim_results.csv"
  
  # 检查 CSV 文件是否存在
  if [ ! -f "$CSV_FILE" ]; then
    echo "Warning: CSV file not found for ${MODEL_NAME} at ${CSV_FILE}. Skipping."
    echo ""
    continue
  fi

  echo "=========================================="
  echo "模型: ${MODEL_NAME}"
  echo "=========================================="
  
  # 使用 Python 计算统计数据（更可靠）
  STATS=$(python3 -c "
import sys
import csv
import numpy as np

csv_file = '${CSV_FILE}'

base_bias_values = []
lora_bias_values = []
black_count = 0
white_count = 0

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        race = row['race']
        base_bias = float(row['base_model_bias_level'])
        lora_bias = float(row['lora_model_bias_level'])
        
        base_bias_values.append(base_bias)
        lora_bias_values.append(lora_bias)
        
        if race == 'Black':
            black_count += 1
        elif race == 'White':
            white_count += 1

total = len(base_bias_values)

if total == 0:
    print('0,0,0,0,0,0,0,0,0,0,0')
    sys.exit(0)

base_bias_array = np.array(base_bias_values)
lora_bias_array = np.array(lora_bias_values)

# 计算统计量
base_mean = np.mean(base_bias_array)
base_median = np.median(base_bias_array)
base_std = np.std(base_bias_array)

lora_mean = np.mean(lora_bias_array)
lora_median = np.median(lora_bias_array)
lora_std = np.std(lora_bias_array)

# 计算偏差减少
bias_reduction = base_mean - lora_mean
if base_mean != 0:
    bias_reduction_percent = (bias_reduction / base_mean) * 100
else:
    bias_reduction_percent = 0

print(f'{total},{black_count},{white_count},{base_mean:.6f},{base_median:.6f},{base_std:.6f},{lora_mean:.6f},{lora_median:.6f},{lora_std:.6f},{bias_reduction:.6f},{bias_reduction_percent:.2f}')
")

  # 解析统计结果
  IFS=',' read -r TOTAL BLACK WHITE BASE_MEAN BASE_MEDIAN BASE_STD LORA_MEAN LORA_MEDIAN LORA_STD BIAS_RED BIAS_RED_PCT <<< "$STATS"

  # 输出到终端和文件
  {
    echo "总样本数: ${TOTAL}"
    echo "  - Black 样本: ${BLACK}"
    echo "  - White 样本: ${WHITE}"
    echo ""
    echo "Base Model 偏差统计:"
    echo "  - 平均偏差: ${BASE_MEAN}"
    echo "  - 中位数偏差: ${BASE_MEDIAN}"
    echo "  - 标准差: ${BASE_STD}"
    echo ""
    echo "LoRA Fine-tuned Model 偏差统计:"
    echo "  - 平均偏差: ${LORA_MEAN}"
    echo "  - 中位数偏差: ${LORA_MEDIAN}"
    echo "  - 标准差: ${LORA_STD}"
    echo ""
    echo "偏差减少:"
    echo "  - 绝对减少: ${BIAS_RED}"
    echo "  - 相对减少: ${BIAS_RED_PCT}%"
    echo ""
  } | tee -a "$OUTPUT_FILE"

  # 写入 CSV
  echo "${MODEL_NAME},${TOTAL},${BLACK},${WHITE},${BASE_MEAN},${BASE_MEDIAN},${BASE_STD},${LORA_MEAN},${LORA_MEDIAN},${LORA_STD},${BIAS_RED},${BIAS_RED_PCT}" >> "$OUTPUT_CSV"

done

echo "=========================================="
echo "统计完成！"
echo "=========================================="
echo "详细结果已保存到: ${OUTPUT_FILE}"
echo "CSV 汇总已保存到: ${OUTPUT_CSV}"
echo ""

# 显示 CSV 汇总表格（使用 column 命令格式化）
if command -v column &> /dev/null; then
  echo "=========================================="
  echo "所有模型汇总表格:"
  echo "=========================================="
  column -t -s',' "$OUTPUT_CSV"
else
  echo "提示: 安装 'column' 命令可以查看格式化的表格"
  echo "CSV 文件: ${OUTPUT_CSV}"
fi
