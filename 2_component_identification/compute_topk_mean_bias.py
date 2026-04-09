#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 top-k 样本的概率差绝对值的平均值
对于每个样本，计算 |fact_p_yes - cf_p_yes| 的平均值
"""

import argparse
import csv
import os
from typing import Dict, List, Optional

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


def load_and_compute_topk_bias(csv_path: str, top_k: int = 100) -> Dict:
    """
    加载 biased_samples_ranking.csv 文件，计算 top-k 样本的概率差绝对值平均值。
    
    Args:
        csv_path: CSV 文件路径
        top_k: 提取前 k 个样本（默认 100）
        
    Returns:
        包含统计信息的字典
    """
    samples = []
    bias_levels = []
    
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
            
            # 计算概率差的绝对值
            bias_level = abs(fact_p_yes - cf_p_yes)
            bias_levels.append(bias_level)
            
            # 转换种族为组 ID
            race_group = _race_to_group(fact_race)
            
            samples.append({
                "rank": rank,
                "index": index,
                "fact_p_yes": fact_p_yes,
                "cf_p_yes": cf_p_yes,
                "fact_race": fact_race,
                "cf_race": cf_race,
                "race_group": race_group,
                "bias_level": bias_level,
            })
    
    if not samples:
        raise ValueError(f"No valid samples found in {csv_path}")
    
    # 计算整体统计
    bias_array = np.array(bias_levels)
    mean_bias = float(np.mean(bias_array))
    median_bias = float(np.median(bias_array))
    std_bias = float(np.std(bias_array))
    min_bias = float(np.min(bias_array))
    max_bias = float(np.max(bias_array))
    
    # 按种族分组统计
    white_samples = [s for s in samples if s["race_group"] == 0]
    black_samples = [s for s in samples if s["race_group"] == 1]
    
    white_bias = [s["bias_level"] for s in white_samples]
    black_bias = [s["bias_level"] for s in black_samples]
    
    white_mean_bias = float(np.mean(white_bias)) if white_bias else 0.0
    black_mean_bias = float(np.mean(black_bias)) if black_bias else 0.0
    
    return {
        "top_k": top_k,
        "total_samples": len(samples),
        "overall_statistics": {
            "mean_bias": mean_bias,
            "median_bias": median_bias,
            "std_bias": std_bias,
            "min_bias": min_bias,
            "max_bias": max_bias,
        },
        "by_race": {
            "white": {
                "count": len(white_samples),
                "mean_bias": white_mean_bias,
            },
            "black": {
                "count": len(black_samples),
                "mean_bias": black_mean_bias,
            },
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean absolute probability difference for top-k biased samples"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to biased_samples_ranking.csv file",
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
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    print("=" * 80)
    print(f"Computing mean absolute bias for top-{args.top_k} samples from:")
    print(f"  {args.csv_path}")
    print("=" * 80)
    
    # DEBUG: 输出第一个样本用于调试
    print("=" * 80)
    print("DEBUG: First sample to be analyzed:")
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row:
            print(f"  Rank: {first_row.get('rank', 'N/A')}")
            print(f"  Index: {first_row.get('index', 'N/A')}")
            print(f"  Fact p_yes: {first_row.get('fact_p_yes', 'N/A')}")
            print(f"  CF p_yes: {first_row.get('cf_p_yes', 'N/A')}")
            print(f"  Fact race: {first_row.get('fact_race', 'N/A')}")
            print(f"  CF race: {first_row.get('cf_race', 'N/A')}")
    print("=" * 80)
    
    # 计算
    results = load_and_compute_topk_bias(args.csv_path, args.top_k)
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"TOP-{args.top_k} SAMPLES: MEAN ABSOLUTE PROBABILITY DIFFERENCE")
    print("=" * 60)
    
    overall = results["overall_statistics"]
    by_race = results["by_race"]
    
    print(f"\nOverall Statistics:")
    print(f"  - Total samples: {results['total_samples']}")
    print(f"  - Mean |fact_p_yes - cf_p_yes|: {overall['mean_bias']:.6f}")
    print(f"  - Median |fact_p_yes - cf_p_yes|: {overall['median_bias']:.6f}")
    print(f"  - Std |fact_p_yes - cf_p_yes|: {overall['std_bias']:.6f}")
    print(f"  - Min |fact_p_yes - cf_p_yes|: {overall['min_bias']:.6f}")
    print(f"  - Max |fact_p_yes - cf_p_yes|: {overall['max_bias']:.6f}")
    
    print(f"\nBy Race Group:")
    print(f"  - White samples: {by_race['white']['count']}")
    print(f"    - Mean |fact_p_yes - cf_p_yes|: {by_race['white']['mean_bias']:.6f}")
    print(f"  - Black samples: {by_race['black']['count']}")
    print(f"    - Mean |fact_p_yes - cf_p_yes|: {by_race['black']['mean_bias']:.6f}")
    
    # 显示前 10 个样本
    print(f"\nTop 10 Samples with Highest Bias:")
    for i, sample in enumerate(results["samples"][:10], 1):
        print(f"  {i:2d}. Rank {sample['rank']:4d}, Index {sample['index']:4d}, "
              f"|Δp|={sample['bias_level']:.6f}, "
              f"fact={sample['fact_p_yes']:.4f} ({sample['fact_race']}), "
              f"cf={sample['cf_p_yes']:.4f} ({sample['cf_race']})")
    
    print("=" * 60)
    
    # 保存结果（如果指定了输出路径）
    if args.output_path:
        import json
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_path}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
