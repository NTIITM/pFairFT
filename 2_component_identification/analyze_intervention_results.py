#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析干预结果，统计fact_p_yes - cf_p_yes的绝对值的均值，并画折线图。

横坐标：干预头的数量
纵坐标：|fact_p_yes - cf_p_yes| 的均值
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_csv_data(csv_path: str) -> Dict[int, List[float]]:
    """
    从CSV文件加载数据，按head_count分组，返回每个head_count对应的bias_level列表。
    
    Returns:
        Dict[int, List[float]]: {head_count: [bias_level1, bias_level2, ...]}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data_by_head_count: Dict[int, List[float]] = defaultdict(list)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        
        required_cols = {"head_count", "fact_p_yes", "cf_p_yes"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        for row in reader:
            try:
                head_count = int(row["head_count"])
                fact_p_yes = float(row["fact_p_yes"])
                cf_p_yes = float(row["cf_p_yes"])
                
                # 计算 |fact_p_yes - cf_p_yes|
                bias_level = abs(fact_p_yes - cf_p_yes)
                data_by_head_count[head_count].append(bias_level)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    return data_by_head_count


def compute_mean_bias_by_head_count(data_by_head_count: Dict[int, List[float]]) -> Dict[int, float]:
    """
    计算每个head_count对应的bias_level的均值。
    
    Returns:
        Dict[int, float]: {head_count: mean_bias_level}
    """
    mean_bias = {}
    for head_count, bias_levels in sorted(data_by_head_count.items()):
        if bias_levels:
            mean_bias[head_count] = np.mean(bias_levels)
        else:
            mean_bias[head_count] = 0.0
    
    return mean_bias


def plot_results(
    mean_bias: Dict[int, float],
    output_path: str,
    title: str = "Mean |fact_p_yes - cf_p_yes| by Head Count",
    xlabel: str = "Number of Intervened Heads",
    ylabel: str = "Mean |fact_p_yes - cf_p_yes|",
):
    """
    绘制单条折线图。
    """
    plot_results_multi([(mean_bias, None)], output_path, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_results_multi(
    series: List[tuple],
    output_path: str,
    title: str = "Mean |fact_p_yes - cf_p_yes| by Head Count",
    xlabel: str = "Number of Intervened Heads",
    ylabel: str = "Mean |fact_p_yes - cf_p_yes|",
):
    """
    绘制多条折线图。
    series: [(mean_bias_dict, label), ...]，label 为 None 时用默认 "Series i"
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for idx, (mean_bias, label) in enumerate(series):
        head_counts = sorted(mean_bias.keys())
        mean_values = [mean_bias[hc] for hc in head_counts]
        line_label = label if label else f"Series {idx + 1}"
        m = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        plt.plot(head_counts, mean_values, marker=m, linewidth=2, markersize=8,
                 label=line_label, color=c)
        for hc, val in zip(head_counts, mean_values):
            plt.annotate(f'{val:.4f}', (hc, val), textcoords="offset points",
                        xytext=(0, 8 if idx % 2 == 0 else -12), ha='center', fontsize=8)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    if series:
        all_hc = sorted(set(hc for mb, _ in series for hc in mb.keys()))
        plt.xticks(all_hc)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def print_statistics(mean_bias: Dict[int, float], data_by_head_count: Dict[int, List[float]]):
    """打印统计信息。"""
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"{'Head Count':<12} {'Mean Bias':<15} {'Std Dev':<15} {'Sample Count':<15}")
    print("-" * 80)
    
    for head_count in sorted(mean_bias.keys()):
        bias_levels = data_by_head_count[head_count]
        mean_val = mean_bias[head_count]
        std_val = np.std(bias_levels) if len(bias_levels) > 1 else 0.0
        count = len(bias_levels)
        print(f"{head_count:<12} {mean_val:<15.6f} {std_val:<15.6f} {count:<15}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze intervention results and plot mean |fact_p_yes - cf_p_yes| by head count."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to a single CSV (for one-line plot). Mutually exclusive with --csv_paths.",
    )
    parser.add_argument(
        "--csv_paths",
        type=str,
        nargs="+",
        default=[],
        help="Paths to two (or more) CSVs for multi-line plot. E.g. sensitive.csv random.csv",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=[],
        help="Labels for each series when using --csv_paths. E.g. 'Sensitive heads' 'Random heads'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for the plot. If not specified, use the same directory as first CSV.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mean_bias_by_head_count.png",
        help="Output plot filename (default: mean_bias_by_head_count.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mean |fact_p_yes - cf_p_yes| by Head Count",
        help="Plot title.",
    )
    args = parser.parse_args()

    if args.csv_paths and args.csv_path:
        raise ValueError("Use either --csv_path (single) or --csv_paths (multiple), not both.")
    if not args.csv_paths and not args.csv_path:
        raise ValueError("Provide either --csv_path or --csv_paths.")

    if args.csv_paths:
        # 多条线：加载多个 CSV
        csv_list = args.csv_paths
        labels = args.labels if args.labels else [f"Series {i+1}" for i in range(len(csv_list))]
        if len(labels) < len(csv_list):
            labels = labels + [f"Series {i+1}" for i in range(len(labels), len(csv_list))]

        print("=" * 80)
        print("Loading data from multiple CSVs (two-line plot)")
        print("=" * 80)
        series = []
        all_data_by_head_count = []
        for i, path in enumerate(csv_list):
            print(f"  [{i+1}] {path}")
            data_by_head_count = load_csv_data(path)
            if not data_by_head_count:
                raise ValueError(f"No valid data in: {path}")
            mean_bias = compute_mean_bias_by_head_count(data_by_head_count)
            series.append((mean_bias, labels[i]))
            all_data_by_head_count.append((labels[i], data_by_head_count))
            for hc in sorted(data_by_head_count.keys()):
                print(f"      Head count {hc}: {len(data_by_head_count[hc])} samples")

        for label, data_by_head_count in all_data_by_head_count:
            mean_bias = compute_mean_bias_by_head_count(data_by_head_count)
            print(f"\n--- {label} ---")
            print_statistics(mean_bias, data_by_head_count)

        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(csv_list[0]) or "."
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output_name)
        print("\n" + "=" * 80)
        print("Generating plot (two lines)...")
        print("=" * 80)
        plot_results_multi(series, output_path, title=args.title)
    else:
        # 单条线：原有逻辑
        print("=" * 80)
        print(f"Loading data from: {args.csv_path}")
        print("=" * 80)
        data_by_head_count = load_csv_data(args.csv_path)
        if not data_by_head_count:
            raise ValueError("No valid data found in CSV file.")
        print(f"Loaded data for {len(data_by_head_count)} different head counts")
        for hc in sorted(data_by_head_count.keys()):
            print(f"  - Head count {hc}: {len(data_by_head_count[hc])} samples")
        mean_bias = compute_mean_bias_by_head_count(data_by_head_count)
        print_statistics(mean_bias, data_by_head_count)
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(args.csv_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output_name)
        print("\n" + "=" * 80)
        print("Generating plot...")
        print("=" * 80)
        plot_results(mean_bias, output_path, title=args.title)

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
