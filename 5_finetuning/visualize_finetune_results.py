#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化微调前后的偏差水平（bias_level）统计结果。

该脚本会：
1. 读取 finetune_discrim_results.csv 文件
2. 计算 base_model_bias_level 和 lora_model_bias_level 的统计指标（均值、中位数、标准差等）
3. 按种族分组统计
4. 生成可视化图表（箱线图、直方图、对比图等）
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def _race_to_group(race: str) -> Optional[int]:
    """Convert race string to group ID: 0 for White, 1 for Black."""
    if not isinstance(race, str):
        return None
    r = race.strip().lower()
    if r == "white":
        return 0
    if r == "black":
        return 1
    return None


def load_results(csv_path: str) -> Tuple[List[float], List[float], List[str]]:
    """
    加载评估结果 CSV 文件。
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        (base_bias_levels, lora_bias_levels, races) 元组
    """
    base_bias_levels = []
    lora_bias_levels = []
    races = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                base_bias = float(row["base_model_bias_level"])
                lora_bias = float(row["lora_model_bias_level"])
                race = row["race"]
                
                base_bias_levels.append(base_bias)
                lora_bias_levels.append(lora_bias)
                races.append(race)
            except (ValueError, KeyError) as e:
                continue
    
    return base_bias_levels, lora_bias_levels, races


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计指标。"""
    if not values:
        return {}
    
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "count": len(values),
    }


def print_statistics(base_stats: Dict[str, float], lora_stats: Dict[str, float], 
                     base_by_race: Dict[str, Dict[str, float]], 
                     lora_by_race: Dict[str, Dict[str, float]]):
    """打印统计结果。"""
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    
    print("\nOverall Statistics:")
    print(f"  Base Model Bias Level:")
    print(f"    - Count: {base_stats.get('count', 0)}")
    print(f"    - Mean: {base_stats.get('mean', 0):.6f}")
    print(f"    - Median: {base_stats.get('median', 0):.6f}")
    print(f"    - Std: {base_stats.get('std', 0):.6f}")
    print(f"    - Min: {base_stats.get('min', 0):.6f}")
    print(f"    - Max: {base_stats.get('max', 0):.6f}")
    print(f"    - Q25: {base_stats.get('q25', 0):.6f}")
    print(f"    - Q75: {base_stats.get('q75', 0):.6f}")
    
    print(f"\n  LoRA Model Bias Level:")
    print(f"    - Count: {lora_stats.get('count', 0)}")
    print(f"    - Mean: {lora_stats.get('mean', 0):.6f}")
    print(f"    - Median: {lora_stats.get('median', 0):.6f}")
    print(f"    - Std: {lora_stats.get('std', 0):.6f}")
    print(f"    - Min: {lora_stats.get('min', 0):.6f}")
    print(f"    - Max: {lora_stats.get('max', 0):.6f}")
    print(f"    - Q25: {lora_stats.get('q25', 0):.6f}")
    print(f"    - Q75: {lora_stats.get('q75', 0):.6f}")
    
    print(f"\n  Improvement (Base - LoRA):")
    mean_improvement = base_stats.get('mean', 0) - lora_stats.get('mean', 0)
    median_improvement = base_stats.get('median', 0) - lora_stats.get('median', 0)
    print(f"    - Mean reduction: {mean_improvement:.6f} ({mean_improvement/base_stats.get('mean', 1)*100:.2f}%)")
    print(f"    - Median reduction: {median_improvement:.6f} ({median_improvement/base_stats.get('median', 1)*100:.2f}%)")
    
    print("\nBy Race Group:")
    for race in ["White", "Black"]:
        base_race_stats = base_by_race.get(race, {})
        lora_race_stats = lora_by_race.get(race, {})
        if base_race_stats:
            print(f"\n  {race}:")
            print(f"    Base Model:")
            print(f"      - Count: {base_race_stats.get('count', 0)}")
            print(f"      - Mean: {base_race_stats.get('mean', 0):.6f}")
            print(f"      - Median: {base_race_stats.get('median', 0):.6f}")
            if lora_race_stats:
                print(f"    LoRA Model:")
                print(f"      - Count: {lora_race_stats.get('count', 0)}")
                print(f"      - Mean: {lora_race_stats.get('mean', 0):.6f}")
                print(f"      - Median: {lora_race_stats.get('median', 0):.6f}")
                improvement = base_race_stats.get('mean', 0) - lora_race_stats.get('mean', 0)
                print(f"    Improvement: {improvement:.6f} ({improvement/base_race_stats.get('mean', 1)*100:.2f}%)")
    
    print("=" * 80)


def plot_results(base_bias_levels: List[float], lora_bias_levels: List[float],
                 races: List[str], output_dir: str, model_name: str = ""):
    """
    绘制可视化图表。
    
    Args:
        base_bias_levels: Base model bias levels
        lora_bias_levels: LoRA model bias levels
        races: Race labels
        output_dir: 输出目录
        model_name: 模型名称（用于标题）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按种族分组
    white_base = [b for b, r in zip(base_bias_levels, races) if r.lower() == "white"]
    black_base = [b for b, r in zip(base_bias_levels, races) if r.lower() == "black"]
    white_lora = [b for b, r in zip(lora_bias_levels, races) if r.lower() == "white"]
    black_lora = [b for b, r in zip(lora_bias_levels, races) if r.lower() == "black"]
    
    # 1. 箱线图对比
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 4, 5]
    box_data = [base_bias_levels, lora_bias_levels, white_base + black_base, white_lora + black_lora]
    labels = ['Base\n(All)', 'LoRA\n(All)', 'Base\n(By Race)', 'LoRA\n(By Race)']
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Bias Level (|fact_p_yes - cf_p_yes|)', fontsize=12)
    ax.set_title(f'Bias Level Comparison: Base vs LoRA{f" ({model_name})" if model_name else ""}', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 按种族分组的箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 4, 5]
    box_data = [white_base, white_lora, black_base, black_lora]
    labels = ['White\nBase', 'White\nLoRA', 'Black\nBase', 'Black\nLoRA']
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Bias Level (|fact_p_yes - cf_p_yes|)', fontsize=12)
    ax.set_title(f'Bias Level by Race: Base vs LoRA{f" ({model_name})" if model_name else ""}', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_by_race_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 直方图对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(base_bias_levels, bins=30, alpha=0.7, color='lightblue', edgecolor='black', label='Base Model')
    axes[0].axvline(np.mean(base_bias_levels), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(base_bias_levels):.4f}')
    axes[0].set_xlabel('Bias Level', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Base Model Bias Level Distribution', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(lora_bias_levels, bins=30, alpha=0.7, color='lightgreen', edgecolor='black', label='LoRA Model')
    axes[1].axvline(np.mean(lora_bias_levels), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lora_bias_levels):.4f}')
    axes[1].set_xlabel('Bias Level', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('LoRA Model Bias Level Distribution', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Bias Level Distributions{f" ({model_name})" if model_name else ""}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 散点图：Base vs LoRA
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 按种族着色
    for i, race in enumerate(races):
        if race.lower() == "white":
            ax.scatter(base_bias_levels[i], lora_bias_levels[i], alpha=0.6, color='blue', s=50, label='White' if i == races.index('White') else '')
        else:
            ax.scatter(base_bias_levels[i], lora_bias_levels[i], alpha=0.6, color='red', s=50, label='Black' if i == races.index('Black') else '')
    
    # 对角线（y=x）
    max_val = max(max(base_bias_levels), max(lora_bias_levels))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x (no improvement)')
    
    ax.set_xlabel('Base Model Bias Level', fontsize=12)
    ax.set_ylabel('LoRA Model Bias Level', fontsize=12)
    ax.set_title(f'Base vs LoRA Bias Level{f" ({model_name})" if model_name else ""}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 条形图：均值对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Overall', 'White', 'Black']
    base_means = [
        np.mean(base_bias_levels),
        np.mean(white_base) if white_base else 0,
        np.mean(black_base) if black_base else 0,
    ]
    lora_means = [
        np.mean(lora_bias_levels),
        np.mean(white_lora) if white_lora else 0,
        np.mean(black_lora) if black_lora else 0,
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_means, width, label='Base Model', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, lora_means, width, label='LoRA Model', color='lightgreen', edgecolor='black')
    
    ax.set_ylabel('Mean Bias Level', fontsize=12)
    ax.set_title(f'Mean Bias Level Comparison{f" ({model_name})" if model_name else ""}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_mean_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fine-tuning evaluation results and compute statistics."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to finetune_discrim_results.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for plots. If not specified, will use the same directory as CSV.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Model name (for plot titles).",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    # 确定输出目录
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.csv_path)
    
    # 如果没有指定模型名，尝试从路径推断
    if not args.model_name:
        csv_dir = os.path.dirname(args.csv_path)
        if "eval_finetune_resume_results_" in csv_dir:
            args.model_name = csv_dir.split("eval_finetune_resume_results_")[-1]
    
    print("=" * 80)
    print(f"Loading results from: {args.csv_path}")
    print("=" * 80)
    
    base_bias_levels, lora_bias_levels, races = load_results(args.csv_path)
    
    if not base_bias_levels or not lora_bias_levels:
        raise ValueError("No valid data found in CSV file.")
    
    print(f"Loaded {len(base_bias_levels)} samples")
    
    # 计算统计指标
    base_stats = compute_statistics(base_bias_levels)
    lora_stats = compute_statistics(lora_bias_levels)
    
    # 按种族分组统计
    white_base = [b for b, r in zip(base_bias_levels, races) if r.lower() == "white"]
    black_base = [b for b, r in zip(base_bias_levels, races) if r.lower() == "black"]
    white_lora = [b for b, r in zip(lora_bias_levels, races) if r.lower() == "white"]
    black_lora = [b for b, r in zip(lora_bias_levels, races) if r.lower() == "black"]
    
    base_by_race = {
        "White": compute_statistics(white_base),
        "Black": compute_statistics(black_base),
    }
    lora_by_race = {
        "White": compute_statistics(white_lora),
        "Black": compute_statistics(black_lora),
    }
    
    # 打印统计结果
    print_statistics(base_stats, lora_stats, base_by_race, lora_by_race)
    
    # 绘制图表
    print("\nGenerating plots...")
    plot_results(base_bias_levels, lora_bias_levels, races, args.output_dir, args.model_name)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
