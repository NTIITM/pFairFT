#!/usr/bin/env python3
"""
统计各模型 sensitive_heads_* 目录下 selected_heads_elbow.json 的元素个数，
并计算与对应模型总 head 数量的比例。
"""

import json
import os

# exp2 目录
EXP2_DIR = os.path.dirname(os.path.abspath(__file__))

# 各模型目录名 -> (num_layers, num_heads)，总 head 数 = num_layers * num_heads
MODEL_SPECS = {
    "sensitive_heads_Qwen3-1.7B_top100": (28, 16),           # 448
    "sensitive_heads_Qwen3-4B_top100": (36, 32),              # 1152
    "sensitive_heads_Qwen3-8B_top100": (36, 32),             # 1152
    "sensitive_heads_Llama-3.2-1B-Instruct_top100": (16, 16), # 256
    "sensitive_heads_Llama-3.2-3B-Instruct_top100": (28, 24), # 672
    "sensitive_heads_Meta-Llama-3-8B-Instruct_top100": (32, 32), # 1024
}

FILENAME = "selected_heads_elbow.json"


def main():
    print("selected_heads_elbow.json 元素个数及占总 head 比例\n")
    print("-" * 70)
    
    # DEBUG: 输出第一个要处理的文件信息
    first_dir = list(MODEL_SPECS.items())[0] if MODEL_SPECS else None
    if first_dir:
        dir_name, (num_layers, num_heads) = first_dir
        path = os.path.join(EXP2_DIR, dir_name, FILENAME)
        print("=" * 80)
        print("DEBUG: First file to be processed:")
        print(f"  Directory: {dir_name}")
        print(f"  File path: {path}")
        print(f"  Model layers: {num_layers}, heads: {num_heads}")
        print("=" * 80)
        print()

    for dir_name, (num_layers, num_heads) in MODEL_SPECS.items():
        path = os.path.join(EXP2_DIR, dir_name, FILENAME)
        if not os.path.isfile(path):
            print(f"{dir_name}: 文件不存在 {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = len(data) if isinstance(data, list) else 0
        total_heads = num_layers * num_heads
        ratio = count / total_heads if total_heads else 0.0

        model_label = dir_name.replace("sensitive_heads_", "").replace("_top100", "")
        print(f"模型: {model_label}")
        print(f"  元素个数: {count}")
        print(f"  总 head 数: {total_heads} ({num_layers} layers × {num_heads} heads)")
        print(f"  比例: {ratio:.4f} ({ratio * 100:.2f}%)")
        print()

    print("-" * 70)


if __name__ == "__main__":
    main()
