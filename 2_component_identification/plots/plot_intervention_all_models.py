#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 run_evaluation_both.sh 产生的所有结果，按模型分子图：Qwen 一组（上排 3 个）、Llama 一组（下排 3 个）。
每个子图对应一个模型，子图内两条线：Sensitive heads 与 Random heads（Mean |fact_p_yes - cf_p_yes| vs head_count）。

结果目录：exp9/intervention_results_${MODEL_NAME}_top100/
  - intervention_results_by_head_count.csv         (敏感头)
  - intervention_results_by_head_count_random.csv  (随机头)
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 模型显示名（与 exp8 风格一致）
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
    """每个 head_count 对应 mean(bias_level)。"""
    mean_bias = {}
    for head_count, bias_levels in sorted(data_by_head_count.items()):
        mean_bias[head_count] = np.mean(bias_levels) if bias_levels else 0.0
    return mean_bias


def discover_result_dirs(results_root: str) -> List[Tuple[str, str, str]]:
    """
    扫描 results_root 下所有 intervention_results_*_top100 目录。
    Returns:
        [(model_name, path_sensitive_csv, path_random_csv), ...]
        仅当两个 CSV 都存在时才加入。
    """
    out = []
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


def plot_qwen_llama_groups(
    sensitive_by_model: Dict[str, Dict[int, float]],
    random_by_model: Dict[str, Dict[int, float]],
    output_path: str,
    xlabel: str = "Number of Intervened Heads",
    ylabel: str = "Fairness Violation↓",
):
    """
    Qwen 和 Llama 分别画成两个 3x1 图（各 3 个子图），
    每个子图内两条线：Sensitive heads 与 Random heads。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    # 放大整体字体，特别是坐标轴标签
    plt.rcParams["font.size"] = 14

    # 根据给定输出路径派生出 Qwen / Llama 两个文件名
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".pdf"
    out_qwen = f"{base}_qwen{ext}"
    out_llama = f"{base}_llama{ext}"

    def _plot_group(models, out_file, title_prefix: str):
        # 每组单独 3x1 图，缩小宽度让每个子图不那么宽
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

        for idx, model in enumerate(models):
            ax = axes[idx]
            disp = MODEL_DISPLAY.get(model, model)
            ax.set_title(disp, fontweight="bold")

            sens = sensitive_by_model.get(model)
            rand = random_by_model.get(model)
            if not sens and not rand:
                ax.set_xticks([])
                continue

            if sens:
                hc_s = sorted(sens.keys())
                ax.plot(
                    hc_s,
                    [sens[k] for k in hc_s],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label="Sensitive heads",
                    color="tab:blue",
                )
            if rand:
                hc_r = sorted(rand.keys())
                ax.plot(
                    hc_r,
                    [rand[k] for k in hc_r],
                    marker="s",
                    linewidth=2,
                    markersize=5,
                    label="Random heads",
                    color="tab:orange",
                )
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.grid(True, alpha=0.3)
            # 只在每个子图内部保留图例，字体略大一些
            ax.legend(loc="best", fontsize=11)

        # 只在最左侧子图设置 y 轴标签
        axes[0].set_ylabel(ylabel, fontweight="bold")

        # 不再添加整体标题，仅对子图做紧凑布局
        plt.tight_layout()
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")

    # Qwen: 3x1
    _plot_group(QWEN_MODELS, out_qwen, "Qwen models")
    # Llama: 3x1
    _plot_group(LLAMA_MODELS, out_llama, "Llama models")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Sensitive vs Random intervention per model: Qwen group (row 0) and Llama group (row 1)."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp9",
        help="Root directory containing intervention_results_*_top100 folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path. Default: <results_root>/intervention_all_models_qwen_llama_groups.pdf",
    )
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    pairs = discover_result_dirs(results_root)
    if not pairs:
        print(f"No intervention_results_*_top100 dirs with both CSVs found under {results_root}")
        return

    print("Discovered model result dirs:")
    for model_name, path_s, path_r in pairs:
        print(f"  {model_name}: sensitive={path_s}, random={path_r}")

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

    out_path = args.output
    if not out_path:
        out_path = os.path.join(
            results_root, "intervention_all_models_qwen_llama_groups.pdf"
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_qwen_llama_groups(sensitive_by_model, random_by_model, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
