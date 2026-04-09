#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 exp10 的 discrim-eval 偏见程度干预实验结果，针对 Llama-3-8B-Instruct 模型。

- 从 intervention_results_${MODEL_NAME}_discrim_eval/ 读取 results_sensitive_heads.csv 和 results_random_heads.csv。
- 选取 head_count = 0, 9, 27 的数据。
- 绘制 Mean |p_yes_a - p_yes_b|（按 question ID 平均后的结果，再总体平均）vs head_count 的折线图。
- 图中包含两条线：敏感头干预 和 随机头干预。
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 字体设置（来自 exp8）
def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"


def load_exp10_csv_data(csv_path: str) -> Dict[Tuple[int, int], float]:
    """
    从 exp10 的 CSV 文件加载数据。
    CSV 列: head_count, model_name, decision_question_id, mean_p_yes_gap, intervention_type
    返回: {(head_count, decision_question_id): mean_p_yes_gap}
    """
    data: Dict[Tuple[int, int], float] = {}
    if not os.path.exists(csv_path):
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        
        required_cols = {"head_count", "decision_question_id", "mean_p_yes_gap"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        for row in reader:
            try:
                head_count = int(row["head_count"])
                decision_question_id = int(row["decision_question_id"])
                mean_p_yes_gap = float(row["mean_p_yes_gap"])
                data[(head_count, decision_question_id)] = mean_p_yes_gap
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}, row: {row}")
                continue
    return data


def compute_overall_mean_gap_by_head_count(
    loaded_data: Dict[Tuple[int, int], float],
    target_head_counts: List[int],
) -> Dict[int, float]:
    """
    计算每个 head_count 对应的总体平均 mean_p_yes_gap（跨所有 question ID）。
    """
    mean_gaps_by_head_count: Dict[int, List[float]] = defaultdict(list)
    for (head_count, qid), mean_gap in loaded_data.items():
        if head_count in target_head_counts:
            mean_gaps_by_head_count[head_count].append(mean_gap)
    
    overall_mean_gaps: Dict[int, float] = {}
    for hc in target_head_counts:
        if mean_gaps_by_head_count[hc]:
            overall_mean_gaps[hc] = np.mean(mean_gaps_by_head_count[hc])
        else:
            overall_mean_gaps[hc] = np.nan # Use NaN if no data for this head_count
            
    return overall_mean_gaps


def plot_discrim_eval_results(
    model_name: str,
    sensitive_overall_mean_gaps: Dict[int, float],
    random_overall_mean_gaps: Dict[int, float],
    output_path: str,
    title: str = "Mean Bias Reduction by Head Count (Discrim-Eval)",
    xlabel: str = "Number of Intervened Heads",
    ylabel: str = "Mean |p_yes_a - p_yes_b| Gap",
):
    """
    绘制 discrim-eval 的结果：单模型，两条线（敏感头 vs 随机头）。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 敏感头干预线
    head_counts_s = sorted(sensitive_overall_mean_gaps.keys())
    mean_values_s = [sensitive_overall_mean_gaps[hc] for hc in head_counts_s]
    ax.plot(head_counts_s, mean_values_s, marker='o', linewidth=2, markersize=8,
            label="Sensitive heads", color='tab:blue')
    for hc, val in zip(head_counts_s, mean_values_s):
        ax.annotate(f'{val:.4f}', (hc, val), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)

    # 随机头干预线
    head_counts_r = sorted(random_overall_mean_gaps.keys())
    mean_values_r = [random_overall_mean_gaps[hc] for hc in head_counts_r]
    ax.plot(head_counts_r, mean_values_r, marker='s', linewidth=2, markersize=8,
            label="Random heads", color='tab:orange')
    for hc, val in zip(head_counts_r, mean_values_r):
        ax.annotate(f'{val:.4f}', (hc, val), textcoords="offset points",
                    xytext=(0, -12), ha='center', fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name}: {title}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    all_hc = sorted(list(set(head_counts_s) | set(head_counts_r)))
    ax.set_xticks(all_hc)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot discrim-eval intervention results for a single model."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing results_sensitive_heads.csv and results_random_heads.csv.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being plotted (e.g., Meta-Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--target_head_counts",
        type=int,
        nargs='+',
        default=[0, 9, 18, 27, 36, 45],
        help="Specific head counts to plot (e.g., 0 9 27).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path. Default: <results_dir>/mean_bias_discrim_eval.png",
    )
    args = parser.parse_args()

    sensitive_csv_path = os.path.join(args.results_dir, "results_sensitive_heads.csv")
    random_csv_path = os.path.join(args.results_dir, "results_random_heads.csv")

    sensitive_raw_data = load_exp10_csv_data(sensitive_csv_path)
    random_raw_data = load_exp10_csv_data(random_csv_path)
    
    if not sensitive_raw_data and not random_raw_data:
        print(f"No data found in {args.results_dir}. Exiting.")
        return

    sensitive_overall_mean_gaps = compute_overall_mean_gap_by_head_count(
        sensitive_raw_data, args.target_head_counts
    )
    random_overall_mean_gaps = compute_overall_mean_gap_by_head_count(
        random_raw_data, args.target_head_counts
    )

    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.results_dir, "mean_bias_discrim_eval.png")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plot_discrim_eval_results(
        model_name=args.model_name,
        sensitive_overall_mean_gaps=sensitive_overall_mean_gaps,
        random_overall_mean_gaps=random_overall_mean_gaps,
        output_path=output_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
