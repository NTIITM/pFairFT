#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在exp8的干预基础上，以5为步长干预不同数量的头，计算事实与反事实数据，保存为CSV。

仅支持：随机头采样（从非敏感头中随机选取，同 exp8 evaluate_intervention_discrim-eval.py 的 negative_random）
以及负向干预（mean ablation）。

该脚本会：
1. 加载敏感头数据与 embedding，构建非敏感头集合
2. 循环不同的头数量（5, 10, 15, 20...），每次从非敏感头中随机选取 N 个头进行负向干预
3. 对每个样本计算事实和反事实概率
4. 将结果保存为CSV文件
"""

import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    get_non_sensitive_heads_from_results,
)
from sampling import sample_resume_data_by_race, load_samples_by_csv_indices
from hook import (
    get_last_token_indices_safe,
    make_intervention_hook_mean_replacement,
    remove_intervention_hooks,
)


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


def compute_p_yes_with_intervention(
    model,
    tokenizer,
    prompt: str,
    model_type: str,
    sensitive_heads: List[Tuple[int, int]],
    white_emb: Dict[Tuple[int, int], np.ndarray],
    black_emb: Dict[Tuple[int, int], np.ndarray],
    race_group: int,
    input_device: torch.device,
    yes_ids: List[int],
    no_ids: List[int],
    num_heads: int,
    head_dim: int,
) -> float:
    """计算带负向干预（mean ablation）的 p(yes) 概率。"""
    formatted_prompt = format_prompt_for_model(prompt, model_type)
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(input_device)
    attention_mask = torch.ones_like(input_ids).to(input_device)

    last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
    output_pos = int(last_token_indices[0].item())

    prompt_hooks = []
    for l, h in sensitive_heads:
        if (l, h) not in white_emb or (l, h) not in black_emb:
            continue

        target_module = model.model.layers[l].self_attn.o_proj
        # 负向干预：mean ablation
        mean_emb_np = (white_emb[(l, h)] + black_emb[(l, h)]) / 2.0
        mean_emb = (
            torch.from_numpy(mean_emb_np).float()
            if isinstance(mean_emb_np, np.ndarray)
            else mean_emb_np
        )
        hook_fn = make_intervention_hook_mean_replacement(
            l, h, mean_emb, output_pos, num_heads, head_dim
        )
        hook = target_module.register_forward_pre_hook(hook_fn)
        prompt_hooks.append(hook)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_row = outputs.logits[0, output_pos, :].float()

            p_yes = compute_p_yes_from_logits_with_warning(
                logits_row=logits_row,
                tokenizer=tokenizer,
                yes_ids=yes_ids,
                no_ids=no_ids,
                sample_idx=0,
                show_warnings=False,
                prefix="Intervention-negative",
            )
            return float(p_yes)
    finally:
        remove_intervention_hooks(prompt_hooks)

    return 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate p(yes) with different numbers of intervened heads (step=5), "
            "computing factual and counterfactual probabilities."
        )
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture for prompt formatting.",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_by_head_count_results",
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
    parser.add_argument(
        "--sensitive_heads_dir",
        type=str,
        default="",
        help="Directory containing results.pkl (heatmap and embeddings).",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="",
        help="Path to results.pkl file (from analyze_race_sensitive_heads.py).",
    )
    parser.add_argument(
        "--max_head_count",
        type=int,
        default=100,
        help="Maximum number of heads to test (will test 5, 10, 15, ..., max_head_count).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size for head count (default: 5).",
    )
    parser.add_argument(
        "--results_csv_name",
        type=str,
        default="intervention_results_by_head_count_random.csv",
        help="Output CSV filename under output_dir (default: intervention_results_by_head_count_random.csv).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

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

    # 采样数据：同 evaluate_intervention_heads.py
    csv_fact_p_yes: Optional[List[Optional[float]]] = None
    if args.sample_csv_path:
        print("=" * 80)
        print(f"Sampling by CSV order: {args.sample_csv_path}")
        print("=" * 80)
        sampled_data, _, csv_fact_p_yes = load_samples_by_csv_indices(
            dataset=dataset,
            csv_path=args.sample_csv_path,
            sample_size=args.sample_size,
        )
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

        samples.append(
            {
                "id": item.get("ID", idx),
                "race": race_group,
                "race_str": extracted_race,
                "query": query,  # 不带 yes/no 指令，用于创建反事实
                "summary": summary,
                "category": category,
            }
        )

    if not samples:
        raise ValueError("No valid samples found (need race in {White, Black}).")

    print(f"Valid samples: {len(samples)}")

    # 确定 embeddings_path（results.pkl，含 heatmap 与 white_emb/black_emb）
    embeddings_path = args.embeddings_path
    if not embeddings_path:
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            exp2_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp2")
            exp2_dir = os.path.abspath(exp2_dir)
            heads_dir = os.path.join(exp2_dir, f"sensitive_heads_{model_name}_top100")
        candidate_path = os.path.join(heads_dir, "results.pkl")
        if os.path.exists(candidate_path):
            embeddings_path = candidate_path
            print(f"Auto-detected embeddings_path: {embeddings_path}")

    if not embeddings_path or not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            "results.pkl not found. Please run analyze_race_sensitive_heads.py first, "
            "or specify --embeddings_path or --sensitive_heads_dir."
        )

    print("=" * 80)
    print(f"Loading intervention data from {embeddings_path} (heatmap-based non-sensitive head set)")
    print("=" * 80)

    results_data = load_intervention_results(embeddings_path)
    non_sensitive_heads = get_non_sensitive_heads_from_results(results_data)
    print(f"Loaded {len(non_sensitive_heads)} non-sensitive heads (heatmap < elbow_score, for random sampling)")

    white_embeddings = results_data.get("white_emb", {})
    black_embeddings = results_data.get("black_emb", {})
    white_emb = {
        (int(k[0]), int(k[1])): v for k, v in white_embeddings.items() if isinstance(k, (tuple, list))
    }
    black_emb = {
        (int(k[0]), int(k[1])): v for k, v in black_embeddings.items() if isinstance(k, (tuple, list))
    }

    # 获取模型配置
    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]

    print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")

    if len(non_sensitive_heads) < args.max_head_count:
        raise ValueError(
            f"Not enough non-sensitive heads with embeddings for random intervention: "
            f"have {len(non_sensitive_heads)}, need at least {args.max_head_count}."
        )
    print(
        f"Random head sampling: {len(non_sensitive_heads)} non-sensitive heads available "
        f"(intervention: negative / mean ablation only)."
    )

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

    # 生成要测试的头数量列表（0, 5, 10, 15, ..., 不超过 non_sensitive_heads 数量）
    max_n = min(args.max_head_count, len(non_sensitive_heads))
    head_counts = list(range(0, max_n + 1, args.step))
    if max_n > 0 and (max_n < args.step or 0 not in head_counts):
        head_counts = [0] + [c for c in head_counts if c > 0]
    head_counts = sorted(set(head_counts))
    print(f"Will test head counts (random heads, negative intervention): {head_counts}")

    # 准备CSV输出
    csv_path = os.path.join(args.output_dir, args.results_csv_name)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "head_count",
        "sample_id",
        "race",
        "fact_p_yes",
        "cf_p_yes",
        "bias_level",
        "intervention_type",
    ])

    # 对每个头数量进行测试（从非敏感头中随机选取 head_count 个头，负向干预）
    for head_count in head_counts:
        print("=" * 80)
        print(f"Testing with {head_count} heads (negative intervention on random non-sensitive heads)")
        print("=" * 80)

        # 从非敏感头中随机选取 head_count 个头（与 exp8 negative_random 一致）
        current_heads = random.sample(non_sensitive_heads, head_count) if head_count > 0 else []
        if head_count == 0:
            print("Using baseline (no intervention)")
        else:
            print(f"Using random heads: {current_heads[:5]}..." if len(current_heads) > 5 else f"Using random heads: {current_heads}")

        # 对每个样本计算事实和反事实概率
        if head_count == 0:
            # Baseline: 无干预，使用批量计算
            fact_prompts = [item["query"] for item in fact_data]
            cf_prompts = [item["query"] for item in cf_data]
            
            fact_p_yes_list = compute_p_yes_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=fact_prompts,
                device=str(input_device),
                yes_ids=yes_ids,
                no_ids=no_ids,
                model_type=model_type,
                desc=f"Computing baseline fact p(yes) with {head_count} heads",
                show_warnings=False,
            )
            
            cf_p_yes_list = compute_p_yes_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=cf_prompts,
                device=str(input_device),
                yes_ids=yes_ids,
                no_ids=no_ids,
                model_type=model_type,
                desc=f"Computing baseline cf p(yes) with {head_count} heads",
                show_warnings=False,
            )
            
            for i, (fact_item, cf_item) in enumerate(tqdm(
                zip(fact_data, cf_data),
                total=len(fact_data),
                desc=f"Processing baseline results"
            )):
                sample = samples[i]
                race_group = sample["race"]
                fact_p_yes = fact_p_yes_list[i]
                cf_p_yes = cf_p_yes_list[i]
                
                # 计算歧视程度
                bias_level = abs(fact_p_yes - cf_p_yes)
                
                # 写入CSV
                csv_writer.writerow([
                    head_count,
                    sample["id"],
                    "Black" if race_group == 1 else "White",
                    f"{fact_p_yes:.6f}",
                    f"{cf_p_yes:.6f}",
                    f"{bias_level:.6f}",
                    "baseline",
                ])
        else:
            # 有干预的情况
            for i, (fact_item, cf_item) in enumerate(tqdm(
                zip(fact_data, cf_data),
                total=len(fact_data),
                desc=f"Computing with {head_count} heads"
            )):
                sample = samples[i]
                race_group = sample["race"]

                # 计算事实概率（带负向干预）
                fact_p_yes = compute_p_yes_with_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=fact_item["query"],
                    model_type=model_type,
                    sensitive_heads=current_heads,
                    white_emb=white_emb,
                    black_emb=black_emb,
                    race_group=race_group,
                    input_device=input_device,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )

                # 计算反事实概率（带干预，但使用反事实的race）
                cf_race_str = cf_item.get("race", "")
                if not cf_race_str:
                    # 如果反事实中没有race字段，从query中提取
                    cf_race_str = extract_race_from_query(cf_item["query"])
                cf_race_group = _race_to_group(cf_race_str)
                if cf_race_group is None:
                    # 如果提取失败，使用翻转的race
                    cf_race_group = 1 - race_group
                
                cf_p_yes = compute_p_yes_with_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=cf_item["query"],
                    model_type=model_type,
                    sensitive_heads=current_heads,
                    white_emb=white_emb,
                    black_emb=black_emb,
                    race_group=cf_race_group,
                    input_device=input_device,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )

                # 计算歧视程度
                bias_level = abs(fact_p_yes - cf_p_yes)

                # 写入CSV
                csv_writer.writerow([
                    head_count,
                    sample["id"],
                    "Black" if race_group == 1 else "White",
                    f"{fact_p_yes:.6f}",
                    f"{cf_p_yes:.6f}",
                    f"{bias_level:.6f}",
                    "negative",
                ])


    csv_file.close()
    print(f"\nSaved results to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
