#!/usr/bin/env bash
set -euo pipefail

# 8B 模型测试训练脚本
# 用于快速测试多GPU训练是否正常工作

PY_SCRIPT="/home/common1/hwluo/project/pFairFT/exp5/finetune_model.py"
EXP5_DIR="/home/common1/hwluo/project/pFairFT/exp5"
EXP2_DIR="/home/common1/hwluo/project/pFairFT/exp2"
DATASET_JSON="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json"

# 测试用的8B模型（选择一个）
# MODEL_PATH="/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct"
MODEL_PATH="/mnt/nfs/huggingface/Qwen/Qwen3-8B"

MODEL_NAME="$(basename "$MODEL_PATH")"
echo "=========================================="
echo "Testing 8B model fine-tuning: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "=========================================="

# 检查对应的 CSV（如果存在）
BIASED_DIR="${EXP2_DIR}/biased_samples_${MODEL_NAME}"
CSV_PATH="${BIASED_DIR}/biased_samples_ranking.csv"

# 测试输出目录
TEST_OUTPUT_DIR="${EXP5_DIR}/test_lora_${MODEL_NAME}"
mkdir -p "$TEST_OUTPUT_DIR"

# 测试参数（较小，用于快速验证）
# 如果显存充足，可以适当增加 batch_size 和 max_samples
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
NUM_EPOCHS=1
BATCH_SIZE=1  # 8B模型建议从1开始，如果显存充足可以增加到2或4
GRADIENT_ACCUMULATION_STEPS=4  # 通过梯度累积保持有效batch size
LEARNING_RATE=2e-5
MAX_SAMPLES=100  # 测试用，少量样本快速验证

# 构建训练命令
if [ -f "$CSV_PATH" ]; then
    echo "Using CSV-driven sampling from: $CSV_PATH"
    echo "Training command:"
    echo "python $PY_SCRIPT \\"
    echo "  --model_path \"$MODEL_PATH\" \\"
    echo "  --dataset_json_path \"$DATASET_JSON\" \\"
    echo "  --output_dir \"$TEST_OUTPUT_DIR\" \\"
    echo "  --sample_csv_path \"$CSV_PATH\" \\"
    echo "  --sample_size $MAX_SAMPLES \\"
    echo "  --train_type \"lora\" \\"
    echo "  --lora_rank $LORA_RANK \\"
    echo "  --lora_alpha $LORA_ALPHA \\"
    echo "  --lora_dropout $LORA_DROPOUT \\"
    echo "  --num_epochs $NUM_EPOCHS \\"
    echo "  --batch_size $BATCH_SIZE \\"
    echo "  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\"
    echo "  --learning_rate $LEARNING_RATE \\"
    echo "  --seed 42"
    echo ""
    
    python "$PY_SCRIPT" \
      --model_path "$MODEL_PATH" \
      --dataset_json_path "$DATASET_JSON" \
      --output_dir "$TEST_OUTPUT_DIR" \
      --sample_csv_path "$CSV_PATH" \
      --sample_size "$MAX_SAMPLES" \
      --train_type "lora" \
      --lora_rank "$LORA_RANK" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --num_epochs "$NUM_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate "$LEARNING_RATE" \
      --seed 42
else
    echo "CSV not found, using default balanced sampling (max_samples=$MAX_SAMPLES)"
    echo "Training command:"
    echo "python $PY_SCRIPT \\"
    echo "  --model_path \"$MODEL_PATH\" \\"
    echo "  --dataset_json_path \"$DATASET_JSON\" \\"
    echo "  --output_dir \"$TEST_OUTPUT_DIR\" \\"
    echo "  --max_samples $MAX_SAMPLES \\"
    echo "  --balanced \\"
    echo "  --train_type \"lora\" \\"
    echo "  --lora_rank $LORA_RANK \\"
    echo "  --lora_alpha $LORA_ALPHA \\"
    echo "  --lora_dropout $LORA_DROPOUT \\"
    echo "  --num_epochs $NUM_EPOCHS \\"
    echo "  --batch_size $BATCH_SIZE \\"
    echo "  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\"
    echo "  --learning_rate $LEARNING_RATE \\"
    echo "  --seed 42"
    echo ""
    
    python "$PY_SCRIPT" \
      --model_path "$MODEL_PATH" \
      --dataset_json_path "$DATASET_JSON" \
      --output_dir "$TEST_OUTPUT_DIR" \
      --max_samples "$MAX_SAMPLES" \
      --balanced \
      --train_type "lora" \
      --lora_rank "$LORA_RANK" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --num_epochs "$NUM_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate "$LEARNING_RATE" \
      --seed 42
fi

echo ""
echo "=========================================="
echo "Test training completed!"
echo "LoRA adapter saved to: $TEST_OUTPUT_DIR/final_model/"
echo "=========================================="
