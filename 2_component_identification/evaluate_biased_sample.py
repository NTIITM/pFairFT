#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估样本的歧视程度（概率差的绝对值）
构建事实和反事实数据，计算每个样本的歧视程度，并按歧视程度排序输出
"""

import csv
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from prompt import (
    add_yes_no_instruction, build_resume_prompt, resolve_model_type
)
from probability import (
    get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES
)
from util import (
    get_input_device, extract_race_from_query,
    create_counterfactual_by_race, compute_p_yes_for_prompt
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate bias level (probability difference) for each sample"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture for prompt formatting. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="biased_samples_ranking.csv",
        help="Output CSV file path for biased samples ranking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--resume_prompt_mode",
        type=str,
        default="summary_only",
        choices=["summary_only", "category", "no_job_description"],
        help="Resume prompt body before the strict Yes/No instruction.",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    input_device = get_input_device(model, args.device)
    print(f"Input tensors will be on device: {input_device}")
    
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    print(f"Using model_type: {model_type}")
    
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")
    
    print("=" * 80)
    print(f"Loading dataset from {args.dataset_json_path} ...")
    print("=" * 80)
    
    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of records.")

    print(f"Processing {len(dataset)} samples...")
    
    # 准备数据：从 summary 和 category 构建 query（不带 yes/no 指令）
    samples = []
    for idx, item in enumerate(dataset):
        summary = item.get("summary", "")
        category = item.get("category", "")
        race_str = item.get("race", "")
        
        
        # 从 query 中提取种族（用于验证）
        extracted_race = extract_race_from_query(summary)
        if extracted_race is None:
            # 如果从 query 中提取不到，使用原始 race 字段
            extracted_race = race_str
        
        if not extracted_race:
            continue
        
        query = build_resume_prompt(
            summary=summary,
            category=category,
            mode=args.resume_prompt_mode,
        )

        samples.append({
            "index": idx,
            "ID": item.get("ID", idx),
            "race": extracted_race,
            "query": query,  # 不带 yes/no 指令
            "summary": summary,
            "category": category,
        })
    
    if not samples:
        raise ValueError("No valid samples found.")
    
    print(f"Valid samples: {len(samples)}")
    
    # 创建反事实数据
    print("=" * 80)
    print("Creating counterfactual data (flipping race)...")
    print("=" * 80)
    
    fact_data = []
    cf_data = []
    
    for sample in samples:
        fact_item = {
            "query": sample["query"],  # 不带 yes/no 指令
            "summary": sample["summary"],
            "category": sample["category"],
            "race": sample["race"],
            "ID": sample["ID"],
            "index": sample["index"],
        }
        
        # 创建反事实数据（基于不带 yes/no 指令的 query）
        cf_item = create_counterfactual_by_race(fact_item)
        
        # 为事实和反事实数据都添加 yes/no 指令
        fact_query_with_instruction = add_yes_no_instruction(fact_item["query"])
        cf_query_with_instruction = add_yes_no_instruction(cf_item["query"])
        
        fact_data.append({
            "query": fact_query_with_instruction,
            "summary": fact_item["summary"],
            "category": fact_item["category"],
            "race": fact_item["race"],
            "ID": fact_item["ID"],
            "index": fact_item["index"],
        })
        
        cf_data.append({
            "query": cf_query_with_instruction,
            "summary": cf_item.get("summary", fact_item["summary"]),
            "category": cf_item.get("category", fact_item["category"]),
            "race": cf_item.get("race", ""),
            "ID": cf_item.get("ID", fact_item["ID"]),
            "index": fact_item["index"],
        })
    
    print(f"Created {len(fact_data)} fact-counterfactual pairs")
    
    # 计算每个样本的原概率和反事实概率
    print("=" * 80)
    print("Computing probabilities for fact and counterfactual data...")
    print("=" * 80)
    
    results = []
    
    # DEBUG: 输出第一个样本用于调试
    if len(fact_data) > 0:
        print("=" * 80)
        print("DEBUG: First sample to be processed by model:")
        print(f"  Fact query: {fact_data[0]['query']}")
        print(f"  CF query: {cf_data[0]['query']}")
        print(f"  Fact race: {fact_data[0].get('race', 'Unknown')}")
        print(f"  CF race: {cf_data[0].get('race', 'Unknown')}")
        print(f"  Index: {fact_data[0].get('index', 'Unknown')}")
        print("=" * 80)
    
    for i, (fact_item, cf_item) in enumerate(tqdm(zip(fact_data, cf_data), total=len(fact_data), desc="Computing probabilities")):
        # 计算原概率（事实）
        fact_p_yes = compute_p_yes_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=fact_item["query"],
            device=input_device,
            model_type=model_type,
            yes_ids=yes_ids,
            no_ids=no_ids,
            sample_idx=fact_item["index"],
            show_warnings=True,
            prefix="Fact",
        )
        
        # 计算反事实概率
        cf_p_yes = compute_p_yes_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=cf_item["query"],
            device=input_device,
            model_type=model_type,
            yes_ids=yes_ids,
            no_ids=no_ids,
            sample_idx=fact_item["index"],
            show_warnings=True,
            prefix="CF",
        )
        
        # 计算概率差的绝对值（歧视程度）
        bias_level = abs(fact_p_yes - cf_p_yes)
        
        results.append({
            "index": fact_item["index"],
            "fact_p_yes": fact_p_yes,
            "cf_p_yes": cf_p_yes,
            "bias_level": bias_level,
            "fact_race": fact_item["race"],
            "cf_race": cf_item.get("race", ""),
        })
    
    # 按歧视程度（概率差的绝对值）排序，从高到低
    results.sort(key=lambda x: x["bias_level"], reverse=True)
    
    # 添加排名
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    
    # 输出CSV
    print("=" * 80)
    print(f"Saving results to {args.output_csv_path}...")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)
    
    with open(args.output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            "rank", "index", "fact_p_yes", "cf_p_yes", "fact_race", "cf_race"
        ])
        
        # 写入数据
        for result in results:
            writer.writerow([
                result["rank"],
                result["index"],
                f"{result['fact_p_yes']:.6f}",
                f"{result['cf_p_yes']:.6f}",
                result["fact_race"],
                result["cf_race"],
            ])
    
    print(f"Saved {len(results)} samples to {args.output_csv_path}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {len(results)}")
    print(f"Mean bias level: {sum(r['bias_level'] for r in results) / len(results):.6f}")
    print(f"Max bias level: {max(r['bias_level'] for r in results):.6f}")
    print(f"Min bias level: {min(r['bias_level'] for r in results):.6f}")
    
    print("\nTop 10 most biased samples:")
    for i, result in enumerate(results[:10], 1):
        print(f"  {i:2d}. Rank {result['rank']:4d}, Index {result['index']:4d}, "
              f"Bias={result['bias_level']:.6f}, "
              f"Fact p(yes)={result['fact_p_yes']:.4f} ({result['fact_race']}), "
              f"CF p(yes)={result['cf_p_yes']:.4f} ({result['cf_race']})")
    
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
