#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 top-k 最具偏见样本的概率差
参考 evaluate_intervention.py 中的 baseline 场景，计算不同种族的概率差
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


def summarize(p_yes: List[float], races: List[int]) -> dict:
    """Summarize p(yes) probabilities by race group."""
    arr = np.asarray(p_yes, dtype=np.float64)
    overall_mean = float(np.mean(arr)) if len(arr) else 0.0

    white_idx = [i for i, r in enumerate(races) if r == 0]
    black_idx = [i for i, r in enumerate(races) if r == 1]

    white_mean = float(np.mean(arr[white_idx])) if white_idx else 0.0
    black_mean = float(np.mean(arr[black_idx])) if black_idx else 0.0
    fairness_gap = black_mean - white_mean

    return {
        "n": int(len(arr)),
        "white_n": int(len(white_idx)),
        "black_n": int(len(black_idx)),
        "p_yes_mean": overall_mean,
        "p_yes_white_mean": white_mean,
        "p_yes_black_mean": black_mean,
        "fairness_gap_black_minus_white": float(fairness_gap),
    }


def load_biased_samples_csv(csv_path: str, top_k: int = 100) -> Dict:
    """
    加载 biased_samples_ranking.csv 文件，提取 top-k 样本的事实概率和种族信息。
    
    Args:
        csv_path: CSV 文件路径
        top_k: 提取前 k 个样本（默认 100）
        
    Returns:
        包含样本信息的字典
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
            
            # 转换种族为组 ID
            race_group = _race_to_group(fact_race)
            if race_group is None:
                continue
            
            samples.append({
                "rank": rank,
                "index": index,
                "fact_p_yes": fact_p_yes,
                "cf_p_yes": cf_p_yes,
                "fact_race": fact_race,
                "cf_race": cf_race,
                "race_group": race_group,
                "bias_level": abs(fact_p_yes - cf_p_yes),
            })
    
    return {
        "samples": samples,
        "total_count": len(samples),
    }


def analyze_topk_bias(csv_path: str, top_k: int = 100) -> Dict:
    """
    分析 top-k 样本的偏见情况，只考虑事实场景，区分不同种族的概率差。
    
    Args:
        csv_path: CSV 文件路径
        top_k: 分析前 k 个样本（默认 100）
        
    Returns:
        分析结果字典
    """
    # 加载数据
    data = load_biased_samples_csv(csv_path, top_k)
    samples = data["samples"]
    
    if not samples:
        raise ValueError(f"No valid samples found in {csv_path}")
    
    # 提取事实概率和种族组
    fact_p_yes_list = [s["fact_p_yes"] for s in samples]
    race_groups = [s["race_group"] for s in samples]
    
    # 使用 summarize 函数计算统计信息
    summary = summarize(fact_p_yes_list, race_groups)
    
    # 额外统计信息
    white_samples = [s for s in samples if s["race_group"] == 0]
    black_samples = [s for s in samples if s["race_group"] == 1]
    
    # 计算平均偏见程度
    avg_bias_level = np.mean([s["bias_level"] for s in samples])
    white_avg_bias = np.mean([s["bias_level"] for s in white_samples]) if white_samples else 0.0
    black_avg_bias = np.mean([s["bias_level"] for s in black_samples]) if black_samples else 0.0
    
    return {
        "top_k": top_k,
        "total_samples": len(samples),
        "summary": summary,
        "bias_statistics": {
            "avg_bias_level": float(avg_bias_level),
            "white_avg_bias": float(white_avg_bias),
            "black_avg_bias": float(black_avg_bias),
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze top-k biased samples from biased_samples_ranking.csv"
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
    print(f"Analyzing top-{args.top_k} biased samples from:")
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
    
    # 分析
    results = analyze_topk_bias(args.csv_path, args.top_k)
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"TOP-{args.top_k} BIASED SAMPLES ANALYSIS (FACT SCENARIO ONLY)")
    print("=" * 60)
    
    summary = results["summary"]
    bias_stats = results["bias_statistics"]
    
    print(f"\nOverall Statistics:")
    print(f"  - Total samples analyzed: {results['total_samples']}")
    print(f"  - Mean p(yes): {summary['p_yes_mean']:.6f}")
    print(f"  - Average bias level: {bias_stats['avg_bias_level']:.6f}")
    
    print(f"\nBy Race Group:")
    print(f"  - White samples: {summary['white_n']}")
    print(f"    - Mean p(yes): {summary['p_yes_white_mean']:.6f}")
    print(f"    - Average bias level: {bias_stats['white_avg_bias']:.6f}")
    print(f"  - Black samples: {summary['black_n']}")
    print(f"    - Mean p(yes): {summary['p_yes_black_mean']:.6f}")
    print(f"    - Average bias level: {bias_stats['black_avg_bias']:.6f}")
    
    print(f"\nFairness Gap (Fact Scenario):")
    print(f"  - Black - White: {summary['fairness_gap_black_minus_white']:.6f}")
    
    # 显示前 10 个样本
    print(f"\nTop 10 Most Biased Samples:")
    for i, sample in enumerate(results["samples"][:10], 1):
        print(f"  {i:2d}. Rank {sample['rank']:4d}, Index {sample['index']:4d}, "
              f"Bias={sample['bias_level']:.6f}, "
              f"Fact p(yes)={sample['fact_p_yes']:.4f} ({sample['fact_race']})")
    
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
