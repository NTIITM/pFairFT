#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估微调前后模型在 Resume 数据集上的表现，保存每个样本在事实 / 反事实上的 p(yes) 概率。

该脚本会：
1. 加载原始模型和微调后的模型（LoRA adapter）
2. 从 Resume 数据集（qwen_summaries_with_race.json）中采样样本（支持 CSV 驱动的 top-k 偏见样本）
3. 对每个样本，构造 fact / counterfactual 两个输入，分别计算原始模型和微调后模型的 p(yes)
4. 将结果保存为 CSV 文件，后续可用 exp2/compute_topk_mean_bias.py 等脚本做 top-k 偏差统计
"""

import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple
import csv

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type, build_category_prompt
from probability import (
    get_target_token_ids,
    YES_CANDIDATES,
    NO_CANDIDATES,
    compute_p_yes_batch,
)
from util import (
    get_input_device,
    extract_race_from_query,
    get_model_config,
    compute_p_yes_from_logits_with_warning,
    create_counterfactual_by_race,
    load_intervention_results,
    get_sensitive_heads_sorted_by_heatmap,
)
from sampling import sample_resume_data_by_race, load_samples_by_csv_indices



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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate original and LoRA-finetuned models on Resume fact/counterfactual data "
            "and save per-sample p(yes) probabilities."
        )
    )
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the original base model directory (HuggingFace / ModelScope).")
    parser.add_argument("--lora_model_path", type=str, required=True,
                        help="Path to the fine-tuned LoRA model adapter directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture for prompt formatting.",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to the Resume dataset JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_discrim_results",
        help="Output directory for results.",
    )
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing.")
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling (balance by race).",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    parser.add_argument("--random_sampling", action="store_true", default=False,
                        help="Use random sampling instead of sequential sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu).")
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        default="",
        help="If set, DO NOT sample from dataset_json_path; instead, follow the order of the CSV's `index` "
             "column and take the first --sample_size indices. This overrides --max_samples/--balanced/"
             "--random_sampling.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="When --sample_csv_path is set, number of rows (indices) to take from the CSV. "
             "If <=0, use all indices in the CSV.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Loading base model and tokenizer...")
    print("=" * 80)

    # 检查可用 GPU 数量，支持多卡推理
    num_gpus = torch.cuda.device_count() if args.device == "cuda" and torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Using device_map='auto' to distribute model across GPUs.")
    elif num_gpus == 1:
        print("Using device_map='auto' to optimize memory usage on single GPU.")
    
    # 使用 device_map="auto" 自动分布模型到多GPU，节省显存
    device_map = "auto" if args.device == "cuda" and torch.cuda.is_available() else None
    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model.eval()

    print("=" * 80)
    print("Loading LoRA model...")
    print("=" * 80)

    # LoRA 模型加载时也需要使用相同的 device_map
    lora_model = PeftModel.from_pretrained(
        base_model, 
        args.lora_model_path,
        device_map=device_map,  # 保持多GPU分布,
        trust_remote_code=True
    )
    lora_model = lora_model.merge_and_unload()  # Merge LoRA weights into the base model
    
    # merge_and_unload 后可能需要重新分布到多GPU
    if device_map == "auto" and hasattr(lora_model, "hf_device_map") and lora_model.hf_device_map is None:
        # 如果 merge 后失去了 device_map，重新应用
        lora_model = lora_model.to("cuda")
        # 对于大模型，可以手动指定 device_map 或使用 accelerate
        print("Warning: After merging LoRA, model may need manual device placement for multi-GPU.")
    
    lora_model.eval()

    input_device = get_input_device(base_model, args.device)
    print(f"Input tensors will be on device: {input_device}")
    
    # 如果使用 device_map，打印模型分布信息
    if hasattr(base_model, "hf_device_map") and base_model.hf_device_map is not None:
        print(f"Base model device map: {base_model.hf_device_map}")
    if hasattr(lora_model, "hf_device_map") and lora_model.hf_device_map is not None:
        print(f"LoRA model device map: {lora_model.hf_device_map}")

    model_type = resolve_model_type(args.model_type, model=base_model, tokenizer=tokenizer, model_path=args.base_model_path)
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

    # 采样数据，并从 CSV 读取 baseline 的 fact_p_yes 和 cf_p_yes（如果使用 CSV）
    csv_fact_p_yes: Optional[List[Optional[float]]] = None
    csv_cf_p_yes: Optional[List[Optional[float]]] = None
    csv_baseline_list: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    used_indices: Optional[List[int]] = None
    
    if args.sample_csv_path:
        print("=" * 80)
        print(f"Sampling by CSV order: {args.sample_csv_path}")
        print("=" * 80)
        sampled_data, used_indices, csv_fact_p_yes = load_samples_by_csv_indices(
            dataset=dataset,
            csv_path=args.sample_csv_path,
            sample_size=args.sample_size,
        )
        
        # 直接从 CSV 读取 fact_p_yes 和 cf_p_yes，按照 CSV 行的顺序建立列表
        csv_baseline_list = []
        if os.path.exists(args.sample_csv_path):
            with open(args.sample_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV file has no headers: {args.sample_csv_path}")
                
                has_fact_p_yes = "fact_p_yes" in reader.fieldnames
                has_cf_p_yes = "cf_p_yes" in reader.fieldnames
                
                if not has_fact_p_yes or not has_cf_p_yes:
                    print(f"Warning: CSV missing fact_p_yes or cf_p_yes columns. Will compute baseline instead.")
                    csv_baseline_list = None
                else:
                    row_count = 0
                    for row in reader:
                        if args.sample_size > 0 and row_count >= args.sample_size:
                            break
                        try:
                            fact_val = row.get("fact_p_yes", "")
                            cf_val = row.get("cf_p_yes", "")
                            
                            fact_p_yes = float(fact_val) if fact_val and fact_val.strip() else None
                            cf_p_yes = float(cf_val) if cf_val and cf_val.strip() else None
                            
                            csv_baseline_list.append((fact_p_yes, cf_p_yes))
                            row_count += 1
                        except (ValueError, KeyError) as e:
                            csv_baseline_list.append((None, None))
                            row_count += 1
                    
                    print(f"Loaded baseline p(yes) values for {len(csv_baseline_list)} samples from CSV")
    else:
        sampled_data = sample_resume_data_by_race(
            data_records=dataset,
            max_samples=args.max_samples,
            balanced=args.balanced,
            random_sampling=args.random_sampling,
            seed=args.seed,
        )

    # 打印采样统计
    white_count = sum(1 for item in sampled_data if item.get("race", "").lower() == "white")
    black_count = sum(1 for item in sampled_data if item.get("race", "").lower() == "black")
    print(f"Sampled {len(sampled_data)} samples")
    if (not args.sample_csv_path) and args.balanced:
        print(f"  - White: {white_count}, Black: {black_count}")
    if args.sample_csv_path:
        print(f"  - White: {white_count}, Black: {black_count} (CSV-driven sampling)")

    # 准备数据：从 summary 和 category 构建 query，并提取种族
    # 如果使用 CSV 采样，需要保存每个 sample 在 sampled_data 中的位置（用于匹配 CSV baseline 值）
    samples = []
    for idx, item in enumerate(sampled_data):
        summary = item.get("summary", "")
        category = item.get("category", "")
        race_str = item.get("race", "")

        if not summary or not category:
            continue

        query = build_category_prompt(summary, category)

        extracted_race = extract_race_from_query(query)
        if extracted_race is None:
            extracted_race = race_str

        race_group = _race_to_group(extracted_race)
        if race_group is None:
            continue

        sample_dict = {
            "id": item.get("ID", idx),
            "race": race_group,
            "race_str": extracted_race,
            "query": query,  # 不带 yes/no 指令，用于创建反事实
            "summary": summary,
            "category": category,
            "sampled_data_idx": idx,  # 在 sampled_data 中的位置（用于匹配 CSV baseline）
        }
        
        samples.append(sample_dict)

    if not samples:
        raise ValueError("No valid samples found (need race in {White, Black}).")

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
            "race": sample["race_str"],
            "id": sample["id"],
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
            "id": fact_item["id"],
        })
        
        cf_data.append({
            "query": cf_query_with_instruction,
            "summary": cf_item.get("summary", fact_item["summary"]),
            "category": cf_item.get("category", fact_item["category"]),
            "race": cf_item.get("race", ""),
            "id": cf_item.get("id", fact_item["id"]),
        })

    print(f"Created {len(fact_data)} fact-counterfactual pairs")

    # 第一次输入样本前打印事实与反事实数据（第一个样本）
    if fact_data and cf_data:
        print("=" * 80)
        print("第一个样本 - 事实与反事实数据")
        print("=" * 80)
        print("事实 (fact_data[0]):")
        for k, v in fact_data[0].items():
            print(f"  {k}: {v}")
        print("反事实 (cf_data[0]):")
        for k, v in cf_data[0].items():
            print(f"  {k}: {v}")
        print("=" * 80)

    # 准备CSV输出
    csv_path = os.path.join(args.output_dir, "finetune_discrim_results.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sample_id",
        "race",
        "base_model_fact_p_yes",
        "base_model_cf_p_yes",
        "base_model_bias_level",
        "lora_model_fact_p_yes",
        "lora_model_cf_p_yes",
        "lora_model_bias_level",
    ])

    # 对每个样本计算原始模型和微调后模型的 p(yes) 概率
    # 如果使用 CSV 且 CSV 中包含 baseline 值，直接从 CSV 读取；否则重新计算
    base_model_fact_prompts = [item["query"] for item in fact_data]
    base_model_cf_prompts = [item["query"] for item in cf_data]

    if csv_baseline_list is not None:
        # 从 CSV 读取 baseline 值（按照 CSV 行的顺序，与 sampled_data 的顺序一致）
        print("=" * 80)
        print("Using baseline p(yes) values from CSV (skipping base model forward pass)")
        print("=" * 80)
        base_model_fact_p_yes_list = []
        base_model_cf_p_yes_list = []
        
        # csv_baseline_list 的顺序与 sampled_data 的顺序一致（都是按 CSV 行的顺序）
        # 使用 sample["sampled_data_idx"] 来从 csv_baseline_list 中获取对应的 baseline 值
        for sample in samples:
            sampled_idx = sample.get("sampled_data_idx", None)
            if sampled_idx is not None and sampled_idx < len(csv_baseline_list):
                fact_p_yes, cf_p_yes = csv_baseline_list[sampled_idx]
                base_model_fact_p_yes_list.append(fact_p_yes)
                base_model_cf_p_yes_list.append(cf_p_yes)
            else:
                # 如果索引超出范围，使用 None（后续会重新计算）
                base_model_fact_p_yes_list.append(None)
                base_model_cf_p_yes_list.append(None)
        
        # 检查是否有缺失值
        missing_count = sum(1 for v in base_model_fact_p_yes_list if v is None)
        if missing_count > 0:
            print(f"Warning: {missing_count} samples missing baseline values in CSV, will compute for those")
    else:
        # 重新计算 baseline
        print("=" * 80)
        print("Computing baseline p(yes) values with base model")
        print("=" * 80)
        base_model_fact_p_yes_list = compute_p_yes_batch(
            model=base_model,
            tokenizer=tokenizer,
            prompts=base_model_fact_prompts,
            device=str(input_device),
            yes_ids=yes_ids,
            no_ids=no_ids,
            model_type=model_type,
            desc=f"Computing base model fact p(yes)",
            show_warnings=False,
        )

        base_model_cf_p_yes_list = compute_p_yes_batch(
            model=base_model,
            tokenizer=tokenizer,
            prompts=base_model_cf_prompts,
            device=str(input_device),
            yes_ids=yes_ids,
            no_ids=no_ids,
            model_type=model_type,
            desc=f"Computing base model cf p(yes)",
            show_warnings=False,
        )

    lora_model_fact_prompts = [item["query"] for item in fact_data]
    lora_model_cf_prompts = [item["query"] for item in cf_data]

    lora_model_fact_p_yes_list = compute_p_yes_batch(
        model=lora_model,
        tokenizer=tokenizer,
        prompts=lora_model_fact_prompts,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing LoRA model fact p(yes)",
        show_warnings=False,
    )

    lora_model_cf_p_yes_list = compute_p_yes_batch(
        model=lora_model,
        tokenizer=tokenizer,
        prompts=lora_model_cf_prompts,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing LoRA model cf p(yes)",
        show_warnings=False,
    )

    for i, (sample, fact_item, cf_item) in enumerate(tqdm(
        zip(samples, fact_data, cf_data),
        total=len(samples),
        desc=f"Processing results"
    )):
        race_group = sample["race"]

        base_model_fact_p_yes = base_model_fact_p_yes_list[i]
        base_model_cf_p_yes = base_model_cf_p_yes_list[i]
        
        # 如果 baseline 值缺失（None），尝试重新计算（这种情况应该很少）
        if base_model_fact_p_yes is None or base_model_cf_p_yes is None:
            print(f"Warning: Sample {sample['id']} missing baseline values, computing...")
            # 这里可以添加重新计算的逻辑，但通常不应该发生
            if base_model_fact_p_yes is None:
                base_model_fact_p_yes = compute_p_yes_batch(
                    model=base_model,
                    tokenizer=tokenizer,
                    prompts=[fact_item["query"]],
                    device=str(input_device),
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    model_type=model_type,
                    desc="",
                    show_warnings=False,
                )[0]
            if base_model_cf_p_yes is None:
                base_model_cf_p_yes = compute_p_yes_batch(
                    model=base_model,
                    tokenizer=tokenizer,
                    prompts=[cf_item["query"]],
                    device=str(input_device),
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    model_type=model_type,
                    desc="",
                    show_warnings=False,
                )[0]
        
        base_model_bias_level = abs(base_model_fact_p_yes - base_model_cf_p_yes)

        lora_model_fact_p_yes = lora_model_fact_p_yes_list[i]
        lora_model_cf_p_yes = lora_model_cf_p_yes_list[i]
        lora_model_bias_level = abs(lora_model_fact_p_yes - lora_model_cf_p_yes)

        csv_writer.writerow([
            sample["id"],
            "Black" if race_group == 1 else "White",
            f"{base_model_fact_p_yes:.6f}" if base_model_fact_p_yes is not None else "NaN",
            f"{base_model_cf_p_yes:.6f}" if base_model_cf_p_yes is not None else "NaN",
            f"{base_model_bias_level:.6f}",
            f"{lora_model_fact_p_yes:.6f}",
            f"{lora_model_cf_p_yes:.6f}",
            f"{lora_model_bias_level:.6f}",
        ])

    csv_file.close()
    print(f"\nSaved results to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
