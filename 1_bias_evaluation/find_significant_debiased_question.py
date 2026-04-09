#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Find significant questions where debiased prompt shows improvement.
基于per_sample_details_all_models.csv数据，找出debiased prompt显著改善的问题。

python /home/common1/hwluo/project/pFairFT/exp1/find_significant_debiased_question.py \
    --csv_path /home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv \
    --dataset_path /home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json \
    --target_models Meta-Llama-3-8B-Instruct Qwen3-8B \
    --top_k 5 \
    --num_examples 1
"""

import argparse
import json
import os
import sys

import pandas as pd

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from util import MODEL_NAME_MAP


def load_dataset(json_path: str) -> dict:
    """加载dataset_paired.json，构建id到样本的映射"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def find_significant_questions(
    csv_path: str,
    target_models: list = None,
    top_k: int = 5,
):
    """
    找出debiased prompt显著改善的问题。
    
    Args:
        csv_path: per_sample_details_all_models.csv路径
        target_models: 目标模型列表，如果为None则使用所有模型
        top_k: 每个模型取前k个最显著的question
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 如果指定了目标模型，则过滤
    if target_models:
        df = df[df["model"].isin(target_models)]
    
    # 按decision_question_id, model, prompt_type分组，计算mean_gap
    # 首先需要计算每对的差值
    pair_diffs = []
    
    # 按model和decision_question_id分组
    for (model, qid), group in df.groupby(["model", "decision_question_id"]):
        # 分别获取prompt和debiased_prompt的数据
        prompt_data = group[group["prompt_type"] == "prompt"].copy()
        debiased_data = group[group["prompt_type"] == "debiased_prompt"].copy()
        
        # 构建sample_id到p_yes的映射
        prompt_map = dict(zip(prompt_data["sample_id"], prompt_data["p_yes"]))
        debiased_map = dict(zip(debiased_data["sample_id"], debiased_data["p_yes"]))
        
        # 找到所有配对
        pairs_processed = set()
        diffs = []
        
        for _, row in prompt_data.iterrows():
            sample_id = row["sample_id"]
            matched_id = row["matched_id"]
            
            if pd.isna(matched_id):
                continue
            
            matched_id = int(matched_id)
            
            # 避免重复处理同一对
            pair_key = tuple(sorted([sample_id, matched_id]))
            if pair_key in pairs_processed:
                continue
            pairs_processed.add(pair_key)
            
            # 获取两个prompt类型的p_yes值
            p_yes_prompt_a = prompt_map.get(sample_id)
            p_yes_prompt_b = prompt_map.get(matched_id)
            p_yes_debiased_a = debiased_map.get(sample_id)
            p_yes_debiased_b = debiased_map.get(matched_id)
            
            if (p_yes_prompt_a is not None and p_yes_prompt_b is not None and
                p_yes_debiased_a is not None and p_yes_debiased_b is not None):
                # 计算差值
                gap_prompt = abs(p_yes_prompt_a - p_yes_prompt_b)
                gap_debiased = abs(p_yes_debiased_a - p_yes_debiased_b)
                diffs.append({
                    "model": model,
                    "decision_question_id": qid,
                    "gap_prompt": gap_prompt,
                    "gap_debiased": gap_debiased,
                    "improvement": gap_prompt - gap_debiased,
                })
        
        if diffs:
            # 计算该question的统计信息
            diffs_df = pd.DataFrame(diffs)
            mean_gap_prompt = diffs_df["gap_prompt"].mean()
            mean_gap_debiased = diffs_df["gap_debiased"].mean()
            mean_improvement = diffs_df["improvement"].mean()
            
            pair_diffs.append({
                "model": model,
                "decision_question_id": qid,
                "mean_gap_prompt": mean_gap_prompt,
                "mean_gap_debiased": mean_gap_debiased,
                "mean_improvement": mean_improvement,
                "count": len(diffs),
            })
    
    if not pair_diffs:
        print("No valid pair differences found.")
        return
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(pair_diffs)
    
    # 只保留improvement > 0的（即debiased prompt有改善的）
    filtered = stats_df[stats_df["mean_improvement"] > 0.0].copy()
    
    if len(filtered) == 0:
        print("No questions found where debiased prompt shows improvement.")
        return
    
    # 对于每个模型，取mean_improvement最大的前top_k个
    top_per_model = (
        filtered
        .sort_values(["model", "mean_improvement"], ascending=[True, False])
        .groupby("model")
        .head(top_k)
    )
    
    # 选择要显示的列
    result = top_per_model[[
        "decision_question_id",
        "model",
        "mean_gap_prompt",
        "mean_gap_debiased",
        "mean_improvement",
        "count",
    ]].copy()
    
    # 格式化输出
    result["mean_gap_prompt"] = result["mean_gap_prompt"].apply(lambda x: f"{x:.6f}")
    result["mean_gap_debiased"] = result["mean_gap_debiased"].apply(lambda x: f"{x:.6f}")
    result["mean_improvement"] = result["mean_improvement"].apply(lambda x: f"{x:.6f}")
    
    print("=" * 80)
    print("Significant Questions with Debiased Prompt Improvement")
    print("=" * 80)
    print(result.to_string(index=False))
    print()
    
    return result


def print_example_samples(
    result_df: pd.DataFrame,
    dataset_path: str,
    num_examples: int = 1,
):
    """
    为每个显著的question打印一个示例样本。
    
    Args:
        result_df: find_significant_questions返回的结果DataFrame
        dataset_path: dataset_paired.json路径
        num_examples: 每个question打印的示例数量
    """
    # 加载数据集
    dataset = load_dataset(dataset_path)
    
    print("=" * 80)
    print("Example Samples for Significant Questions")
    print("=" * 80)
    
    for _, row in result_df.iterrows():
        qid = int(row["decision_question_id"])
        model = row["model"]
        improvement = float(row["mean_improvement"])
        
        print(f"\n{'=' * 80}")
        print(f"Question ID: {qid}")
        print(f"Model: {model}")
        print(f"Improvement: {improvement:.6f}")
        print(f"{'=' * 80}")
        
        # 找到该question_id的一个示例
        examples_found = 0
        for sample_id, sample in dataset.items():
            if sample.get("decision_question_id") == qid:
                print(f"\nSample ID: {sample_id}")
                print(f"Matched ID: {sample.get('matched_id')}")
                print(f"Race: {sample.get('race')}")
                print(f"\nOriginal Prompt:")
                print("-" * 80)
                prompt_text = sample.get("prompt", "")
                if len(prompt_text) > 500:
                    print(prompt_text[:500] + "...")
                else:
                    print(prompt_text)
                print(f"\nDebiased Prompt:")
                print("-" * 80)
                debiased_text = sample.get("debiased_prompt", "")
                if len(debiased_text) > 500:
                    print(debiased_text[:500] + "...")
                else:
                    print(debiased_text)
                
                examples_found += 1
                if examples_found >= num_examples:
                    break
        
        if examples_found == 0:
            print(f"  (No example found for question_id={qid})")


def main():
    parser = argparse.ArgumentParser(
        description="Find significant questions where debiased prompt shows improvement."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv",
        help="Path to per_sample_details_all_models.csv",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
        help="Path to dataset_paired.json",
    )
    parser.add_argument(
        "--target_models",
        type=str,
        nargs="+",
        default=None,
        help="Target models to analyze (e.g., 'Meta-Llama-3-8B-Instruct' 'Qwen3-8B'). If not specified, use all models.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top questions to show per model (default: 5)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of example samples to print per question (default: 1)",
    )
    
    args = parser.parse_args()
    
    # 查找显著的问题
    result_df = find_significant_questions(
        csv_path=args.csv_path,
        target_models=args.target_models,
        top_k=args.top_k,
    )
    
    if result_df is not None and len(result_df) > 0:
        # 打印示例样本
        print_example_samples(
            result_df=result_df,
            dataset_path=args.dataset_path,
            num_examples=args.num_examples,
        )


if __name__ == "__main__":
    main()
