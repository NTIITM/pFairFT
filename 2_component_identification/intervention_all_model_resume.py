#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize intervention results for all models in exp9.

参照 exp9/plot_intervention_all_models.py 的数据处理逻辑：
- 结果目录：exp9/intervention_results_${MODEL_NAME}_top100/
    - intervention_results_by_head_count.csv         (敏感头 / Sensitive heads)
    - intervention_results_by_head_count_random.csv  (随机头 / Random heads)
- 每个 CSV 中的 y 轴 fairness violation = mean(bias_level) for each head_count。

本脚本不画图，只在终端中输出：
- 每个模型、每种干预类型（Sensitive / Random）
- 在 head_count = 0 和 最大 head_count 时的 fairness violation 数值。
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


# 模型显示名（与 exp9/plot_intervention_all_models.py 一致）
MODEL_DISPLAY = {
    "Qwen3-1.7B": "Qwen 1.7B",
    "Qwen3-4B": "Qwen 4B",
    "Qwen3-8B": "Qwen 8B",
    "Llama-3.2-1B-Instruct": "Llama 1B",
    "Llama-3.2-3B-Instruct": "Llama 3B",
    "Meta-Llama-3-8B-Instruct": "Llama 8B",
}

# Qwen 一组、Llama 一组，每组 3 个模型
QWEN_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
LLAMA_MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct",
]
MODEL_ORDER = QWEN_MODELS + LLAMA_MODELS


def load_csv_data(csv_path: str) -> Dict[int, List[float]]:
    """
    从 exp9 的 CSV 加载数据，按 head_count 分组，返回每个 head_count 对应的 bias_level 列表。

    与 exp9/plot_intervention_all_models.py 中的 load_csv_data 保持一致。

    CSV 列: head_count, sample_id, race, fact_p_yes, cf_p_yes, bias_level, intervention_type
    """
    if not os.path.exists(csv_path):
        return {}
    data_by_head_count: Dict[int, List[float]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            try:
                head_count = int(row["head_count"])
                bias_level = float(row["bias_level"])
                data_by_head_count[head_count].append(bias_level)
            except (ValueError, KeyError):
                continue
    return dict(data_by_head_count)


def compute_mean_bias_by_head_count(
    data_by_head_count: Dict[int, List[float]],
) -> Dict[int, float]:
    """
    每个 head_count 对应 mean(bias_level)。
    这就是图中的 y 轴 fairness violation。
    """
    mean_bias: Dict[int, float] = {}
    for head_count, bias_levels in sorted(data_by_head_count.items()):
        mean_bias[head_count] = float(np.mean(bias_levels)) if bias_levels else 0.0
    return mean_bias


def discover_result_dirs(results_root: str) -> List[Tuple[str, str, str]]:
    """
    扫描 results_root 下所有 intervention_results_*_top100 目录。

    Returns:
        [(model_name, path_sensitive_csv, path_random_csv), ...]
        仅当两个 CSV 都存在时才加入。
    """
    out: List[Tuple[str, str, str]] = []
    if not os.path.isdir(results_root):
        return out
    for name in os.listdir(results_root):
        if not name.startswith("intervention_results_") or not name.endswith("_top100"):
            continue
        # model_name: e.g. Qwen3-4B, Meta-Llama-3-8B-Instruct
        prefix = "intervention_results_"
        suffix = "_top100"
        model_name = name[len(prefix) : -len(suffix)]
        dir_path = os.path.join(results_root, name)
        if not os.path.isdir(dir_path):
            continue
        path_sensitive = os.path.join(
            dir_path, "intervention_results_by_head_count.csv"
        )
        path_random = os.path.join(
            dir_path, "intervention_results_by_head_count_random.csv"
        )
        if os.path.isfile(path_sensitive) and os.path.isfile(path_random):
            out.append((model_name, path_sensitive, path_random))
    return out


def summarize_head_counts(mean_by_head: Dict[int, float]) -> Tuple[int, float, int, float]:
    """
    从 mean_by_head 中提取：
    - head_count = 0 时的 fairness violation（如果不存在 0，就用最小 head_count）
    - 最大 head_count 时的 fairness violation
    """
    if not mean_by_head:
        return 0, float("nan"), 0, float("nan")

    head_counts = sorted(mean_by_head.keys())
    # 优先使用 0，如果没有 0，就用最小 head_count
    hc_min = 0 if 0 in mean_by_head else head_counts[0]
    hc_max = head_counts[-1]
    v_min = mean_by_head.get(hc_min, float("nan"))
    v_max = mean_by_head.get(hc_max, float("nan"))
    return hc_min, v_min, hc_max, v_max


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize fairness violation at head_count=0 and max head_count "
            "for each model (Sensitive vs Random heads)."
        )
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp9",
        help="Root directory containing intervention_results_*_top100 folders.",
    )
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    pairs = discover_result_dirs(results_root)
    if not pairs:
        print(
            f"No intervention_results_*_top100 dirs with both CSVs found under {results_root}"
        )
        return

    # 先按模型加载 mean bias per head_count
    sensitive_by_model: Dict[str, Dict[int, float]] = {}
    random_by_model: Dict[str, Dict[int, float]] = {}

    for model_name, path_sensitive, path_random in pairs:
        data_s = load_csv_data(path_sensitive)
        data_r = load_csv_data(path_random)
        if data_s:
            sensitive_by_model[model_name] = compute_mean_bias_by_head_count(data_s)
        if data_r:
            random_by_model[model_name] = compute_mean_bias_by_head_count(data_r)

    if not sensitive_by_model and not random_by_model:
        print("No valid data loaded.")
        return

    def print_group(group_name: str, models: List[str]):
        print(f"\n==== {group_name} ====")
        for model in models:
            disp = MODEL_DISPLAY.get(model, model)
            sens = sensitive_by_model.get(model)
            rand = random_by_model.get(model)
            print(f"\nModel: {model} ({disp})")

            if sens:
                hc0, v0, hc_max, v_max = summarize_head_counts(sens)
                print(
                    f"  Sensitive heads:"
                    f" head_count={hc0:<3d} -> fairness_violation={v0:.6f},"
                    f" head_count={hc_max:<3d} -> fairness_violation={v_max:.6f}"
                )
            else:
                print("  Sensitive heads: NO DATA")

            if rand:
                hc0, v0, hc_max, v_max = summarize_head_counts(rand)
                print(
                    f"  Random heads   :"
                    f" head_count={hc0:<3d} -> fairness_violation={v0:.6f},"
                    f" head_count={hc_max:<3d} -> fairness_violation={v_max:.6f}"
                )
            else:
                print("  Random heads   : NO DATA")

    # 按照 Qwen / Llama 两类分别输出
    print_group("Qwen models", QWEN_MODELS)
    print_group("Llama models", LLAMA_MODELS)


if __name__ == "__main__":
    main()

