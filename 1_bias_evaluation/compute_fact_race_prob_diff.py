#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 exp_sample.sh 产生的全部结果，计算所有模型在 rank 前 100 的样本中：
事实样本中按照race分类，计算所得的均值做差
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def load_topk_samples(csv_path: str, top_k: int = 100) -> List[Dict]:
    """
    加载 biased_samples_ranking.csv 文件，提取 top-k 样本。
    
    Args:
        csv_path: CSV 文件路径
        top_k: 提取前 k 个样本（默认 100）
        
    Returns:
        样本列表
    """
    samples = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= top_k:
                break
            
            rank = int(row["rank"])
            index = int(row["index"])
            fact_p_yes = float(row["fact_p_yes"])
            fact_race = row["fact_race"]
            
            samples.append({
                "rank": rank,
                "index": index,
                "fact_p_yes": fact_p_yes,
                "fact_race": fact_race,
            })
    
    return samples


def compute_fact_race_prob_diff(samples: List[Dict]) -> Dict:
    """
    计算事实样本中不同race组的概率差。
    
    Args:
        samples: 样本列表
        
    Returns:
        统计信息字典
    """
    if not samples:
        return {}
    
    # 按 fact_race 分组
    fact_white_samples = [s for s in samples if _race_to_group(s["fact_race"]) == 0]
    fact_black_samples = [s for s in samples if _race_to_group(s["fact_race"]) == 1]
    
    # 计算各组的 fact_p_yes 均值
    fact_white_mean = float(np.mean([s["fact_p_yes"] for s in fact_white_samples])) if fact_white_samples else 0.0
    fact_black_mean = float(np.mean([s["fact_p_yes"] for s in fact_black_samples])) if fact_black_samples else 0.0
    
    # 计算概率差（Black - White）
    prob_diff = fact_black_mean - fact_white_mean
    
    return {
        "total_samples": len(samples),
        "white": {
            "count": len(fact_white_samples),
            "mean_p_yes": fact_white_mean,
        },
        "black": {
            "count": len(fact_black_samples),
            "mean_p_yes": fact_black_mean,
        },
        "prob_diff_black_minus_white": float(prob_diff),
    }


def find_all_model_results(results_dir: str) -> List[Tuple[str, str]]:
    """
    查找所有模型的结果文件。
    
    Args:
        results_dir: 结果目录路径（exp2 目录）
        
    Returns:
        (模型名, CSV 文件路径) 的列表
    """
    pattern = os.path.join(results_dir, "biased_samples_*", "biased_samples_ranking.csv")
    csv_files = glob.glob(pattern)
    
    model_results = []
    for csv_path in sorted(csv_files):
        # 从路径中提取模型名
        # 例如: /path/to/biased_samples_Qwen3-8B/biased_samples_ranking.csv
        dir_name = os.path.basename(os.path.dirname(csv_path))
        # 移除 "biased_samples_" 前缀
        if dir_name.startswith("biased_samples_"):
            model_name = dir_name[len("biased_samples_"):]
        else:
            model_name = dir_name
        
        model_results.append((model_name, csv_path))
    
    return model_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute fact race probability difference for all models from exp_sample.sh results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp2",
        help="Directory containing biased_samples_* subdirectories (default: exp2)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top samples to analyze (default: 100)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"分析 rank 前 {args.top_k} 的样本中，事实样本不同race组的概率差")
    print(f"结果目录: {args.results_dir}")
    print("=" * 80)
    
    # 查找所有模型的结果文件
    model_results = find_all_model_results(args.results_dir)
    
    if not model_results:
        print(f"警告: 在 {args.results_dir} 中未找到模型结果")
        return
    
    print(f"\n找到 {len(model_results)} 个模型:")
    for model_name, csv_path in model_results:
        print(f"  - {model_name}")
    
    # 处理每个模型
    all_results = {}
    for model_name, csv_path in model_results:
        print(f"\n{'=' * 80}")
        print(f"处理模型: {model_name}")
        print(f"  CSV: {csv_path}")
        print(f"{'=' * 80}")
        
        if not os.path.exists(csv_path):
            print(f"  警告: CSV 文件不存在，跳过: {csv_path}")
            continue
        
        try:
            # 加载样本
            samples = load_topk_samples(csv_path, args.top_k)
            
            if not samples:
                print(f"  警告: 在 {csv_path} 中未找到样本")
                continue
            
            # 计算统计信息
            stats = compute_fact_race_prob_diff(samples)
            all_results[model_name] = stats
            
            # 打印结果
            print(f"\n  {model_name} 的结果:")
            print(f"    - 总样本数: {stats['total_samples']}")
            print(f"\n    事实样本按race分类:")
            print(f"      White: 数量={stats['white']['count']}, "
                  f"mean_p_yes={stats['white']['mean_p_yes']:.6f}")
            print(f"      Black: 数量={stats['black']['count']}, "
                  f"mean_p_yes={stats['black']['mean_p_yes']:.6f}")
            print(f"    - 概率差 (Black - White): {stats['prob_diff_black_minus_white']:.6f}")
            
        except Exception as e:
            print(f"  处理 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总信息
    print(f"\n{'=' * 80}")
    print("所有模型汇总")
    print(f"{'=' * 80}")
    
    if all_results:
        print(f"\n{'模型名称':<35} {'White均值':<15} {'Black均值':<15} {'概率差(Black-White)':<20}")
        print("-" * 90)
        for model_name, stats in sorted(all_results.items()):
            white_mean = stats['white']['mean_p_yes']
            black_mean = stats['black']['mean_p_yes']
            prob_diff = stats['prob_diff_black_minus_white']
            print(f"{model_name:<35} {white_mean:>14.6f}  {black_mean:>14.6f}  {prob_diff:>19.6f}")
        
        # 计算所有模型的平均值
        all_white_means = [stats['white']['mean_p_yes'] for stats in all_results.values()]
        all_black_means = [stats['black']['mean_p_yes'] for stats in all_results.values()]
        all_prob_diffs = [stats['prob_diff_black_minus_white'] for stats in all_results.values()]
        
        print("-" * 90)
        print(f"{'平均值':<35} {np.mean(all_white_means):>14.6f}  {np.mean(all_black_means):>14.6f}  {np.mean(all_prob_diffs):>19.6f}")
    
    print("\n完成。")


if __name__ == "__main__":
    main()
