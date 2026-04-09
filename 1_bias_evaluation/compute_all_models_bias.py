#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 exp_sample.sh 产生的全部结果，计算所有模型在 rank 前 100 的样本中：
1. 事实与反事实的概率差均值
2. 对于事实与反事实样本，按照 race 分类，计算所得的均值做差
"""

import argparse
import csv
import glob
import json
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
            cf_p_yes = float(row["cf_p_yes"])
            fact_race = row["fact_race"]
            cf_race = row["cf_race"]
            
            samples.append({
                "rank": rank,
                "index": index,
                "fact_p_yes": fact_p_yes,
                "cf_p_yes": cf_p_yes,
                "fact_race": fact_race,
                "cf_race": cf_race,
            })
    
    return samples


def compute_model_statistics(samples: List[Dict]) -> Dict:
    """
    计算单个模型的统计信息。
    
    Args:
        samples: 样本列表
        
    Returns:
        统计信息字典
    """
    if not samples:
        return {}
    
    # 1. 计算所有样本的事实与反事实的概率差均值
    prob_diffs = [abs(s["fact_p_yes"] - s["cf_p_yes"]) for s in samples]
    mean_prob_diff = float(np.mean(prob_diffs))
    
    # 2. 按 fact_race 分类，分别计算事实和反事实的均值
    # 按 fact_race 分组，对于每个组，计算该组中 fact_p_yes 和 cf_p_yes 的均值
    fact_white_samples = [s for s in samples if _race_to_group(s["fact_race"]) == 0]
    fact_black_samples = [s for s in samples if _race_to_group(s["fact_race"]) == 1]
    
    # White 组：fact_race=White 的样本
    fact_white_mean = float(np.mean([s["fact_p_yes"] for s in fact_white_samples])) if fact_white_samples else 0.0
    cf_white_mean = float(np.mean([s["cf_p_yes"] for s in fact_white_samples])) if fact_white_samples else 0.0
    
    # Black 组：fact_race=Black 的样本
    fact_black_mean = float(np.mean([s["fact_p_yes"] for s in fact_black_samples])) if fact_black_samples else 0.0
    cf_black_mean = float(np.mean([s["cf_p_yes"] for s in fact_black_samples])) if fact_black_samples else 0.0
    
    # 3. 计算事实与反事实的均值差（按 race 分类）
    white_mean_diff = fact_white_mean - cf_white_mean
    black_mean_diff = fact_black_mean - cf_black_mean
    
    # 4. 不管数据来源是事实还是反事实，只考虑 race，按 race 分类计算 p_yes 概率均值做差
    # 收集所有 White 相关的 p_yes（包括 fact_race=White 的 fact_p_yes 和 cf_race=White 的 cf_p_yes）
    white_p_yes_list = []
    white_p_yes_list.extend([s["fact_p_yes"] for s in samples if _race_to_group(s["fact_race"]) == 0])
    white_p_yes_list.extend([s["cf_p_yes"] for s in samples if _race_to_group(s["cf_race"]) == 0])
    
    # 收集所有 Black 相关的 p_yes（包括 fact_race=Black 的 fact_p_yes 和 cf_race=Black 的 cf_p_yes）
    black_p_yes_list = []
    black_p_yes_list.extend([s["fact_p_yes"] for s in samples if _race_to_group(s["fact_race"]) == 1])
    black_p_yes_list.extend([s["cf_p_yes"] for s in samples if _race_to_group(s["cf_race"]) == 1])
    
    # 计算各组的总体均值
    overall_white_mean = float(np.mean(white_p_yes_list)) if white_p_yes_list else 0.0
    overall_black_mean = float(np.mean(black_p_yes_list)) if black_p_yes_list else 0.0
    
    # 计算 fairness gap (Black - White)
    fairness_gap = overall_black_mean - overall_white_mean
    
    return {
        "total_samples": len(samples),
        "mean_prob_diff": mean_prob_diff,
        "fact_by_race": {
            "white": {
                "count": len(fact_white_samples),
                "mean_p_yes": fact_white_mean,
            },
            "black": {
                "count": len(fact_black_samples),
                "mean_p_yes": fact_black_mean,
            },
        },
        "cf_by_race": {
            "white": {
                "count": len(fact_white_samples),  # 使用 fact_white_samples 的计数
                "mean_p_yes": cf_white_mean,
            },
            "black": {
                "count": len(fact_black_samples),  # 使用 fact_black_samples 的计数
                "mean_p_yes": cf_black_mean,
            },
        },
        "mean_diff_by_race": {
            "white": float(white_mean_diff),
            "black": float(black_mean_diff),
        },
        "overall_by_race": {
            "white": {
                "count": len(white_p_yes_list),
                "mean_p_yes": overall_white_mean,
            },
            "black": {
                "count": len(black_p_yes_list),
                "mean_p_yes": overall_black_mean,
            },
            "fairness_gap_black_minus_white": float(fairness_gap),
        },
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
        description="Compute bias statistics for all models from exp_sample.sh results"
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
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional output path to save analysis results (JSON format)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Analyzing top-{args.top_k} samples for all models")
    print(f"Results directory: {args.results_dir}")
    print("=" * 80)
    
    # 查找所有模型的结果文件
    model_results = find_all_model_results(args.results_dir)
    
    if not model_results:
        print(f"Warning: No model results found in {args.results_dir}")
        return
    
    print(f"\nFound {len(model_results)} models:")
    for model_name, csv_path in model_results:
        print(f"  - {model_name}")
    
    # 处理每个模型
    all_results = {}
    for model_name, csv_path in model_results:
        print(f"\n{'=' * 80}")
        print(f"Processing model: {model_name}")
        print(f"  CSV: {csv_path}")
        print(f"{'=' * 80}")
        
        if not os.path.exists(csv_path):
            print(f"  Warning: CSV file not found, skipping: {csv_path}")
            continue
        
        try:
            # 加载样本
            samples = load_topk_samples(csv_path, args.top_k)
            
            if not samples:
                print(f"  Warning: No samples found in {csv_path}")
                continue
            
            # 计算统计信息
            stats = compute_model_statistics(samples)
            all_results[model_name] = stats
            
            # 打印结果
            print(f"\n  Results for {model_name}:")
            print(f"    - Total samples: {stats['total_samples']}")
            print(f"    - Mean |fact_p_yes - cf_p_yes|: {stats['mean_prob_diff']:.6f}")
            
            print(f"\n    Fact scenario by race:")
            print(f"      White: count={stats['fact_by_race']['white']['count']}, "
                  f"mean_p_yes={stats['fact_by_race']['white']['mean_p_yes']:.6f}")
            print(f"      Black: count={stats['fact_by_race']['black']['count']}, "
                  f"mean_p_yes={stats['fact_by_race']['black']['mean_p_yes']:.6f}")
            
            print(f"\n    Counterfactual scenario by race:")
            print(f"      White: count={stats['cf_by_race']['white']['count']}, "
                  f"mean_p_yes={stats['cf_by_race']['white']['mean_p_yes']:.6f}")
            print(f"      Black: count={stats['cf_by_race']['black']['count']}, "
                  f"mean_p_yes={stats['cf_by_race']['black']['mean_p_yes']:.6f}")
            
            print(f"\n    Mean difference (Fact - CF) by race:")
            print(f"      White: {stats['mean_diff_by_race']['white']:.6f}")
            print(f"      Black: {stats['mean_diff_by_race']['black']:.6f}")
            
            print(f"\n    Overall p_yes by race (regardless of fact/cf source):")
            print(f"      White: count={stats['overall_by_race']['white']['count']}, "
                  f"mean_p_yes={stats['overall_by_race']['white']['mean_p_yes']:.6f}")
            print(f"      Black: count={stats['overall_by_race']['black']['count']}, "
                  f"mean_p_yes={stats['overall_by_race']['black']['mean_p_yes']:.6f}")
            print(f"      Fairness gap (Black - White): {stats['overall_by_race']['fairness_gap_black_minus_white']:.6f}")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总信息
    print(f"\n{'=' * 80}")
    print("SUMMARY FOR ALL MODELS")
    print(f"{'=' * 80}")
    
    if all_results:
        print(f"\n{'Model Name':<35} {'Mean Prob Diff':<18} {'White Diff':<15} {'Black Diff':<15} {'Fairness Gap':<15}")
        print("-" * 113)
        for model_name, stats in sorted(all_results.items()):
            mean_diff = stats['mean_prob_diff']
            white_diff = stats['mean_diff_by_race']['white']
            black_diff = stats['mean_diff_by_race']['black']
            fairness_gap = stats['overall_by_race']['fairness_gap_black_minus_white']
            print(f"{model_name:<35} {mean_diff:>17.6f}  {white_diff:>14.6f}  {black_diff:>14.6f}  {fairness_gap:>14.6f}")
        
        # 计算所有模型的平均值
        all_mean_diffs = [stats['mean_prob_diff'] for stats in all_results.values()]
        all_white_diffs = [stats['mean_diff_by_race']['white'] for stats in all_results.values()]
        all_black_diffs = [stats['mean_diff_by_race']['black'] for stats in all_results.values()]
        all_fairness_gaps = [stats['overall_by_race']['fairness_gap_black_minus_white'] for stats in all_results.values()]
        
        print("-" * 113)
        print(f"{'Average':<35} {np.mean(all_mean_diffs):>17.6f}  {np.mean(all_white_diffs):>14.6f}  {np.mean(all_black_diffs):>14.6f}  {np.mean(all_fairness_gaps):>14.6f}")
    
    # 保存结果（如果指定了输出路径）
    if args.output_path:
        output_data = {
            "top_k": args.top_k,
            "results_dir": args.results_dir,
            "models": all_results,
        }
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_path}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
