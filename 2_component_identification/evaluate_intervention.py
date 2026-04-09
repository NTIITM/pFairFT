#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate baseline p(yes) probability differences across race groups for Resume dataset
比较不同种族（White vs Black）的 p(yes) 概率差异
"""

import csv
import json
import os
import pickle
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type
from probability import (
    get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES,
    compute_p_yes_batch
)
from util import get_input_device, extract_race_from_query, get_model_config, compute_p_yes_from_logits_with_warning
from sampling import sample_resume_data_by_race
from hook import (
    register_intervention_hooks,
    remove_intervention_hooks
)
from tqdm import tqdm


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


def _load_samples_by_csv_indices(
    dataset: List[dict],
    csv_path: str,
    sample_size: int,
) -> List[dict]:
    """
    根据 CSV 中的 index 列顺序，从原始 dataset 中选取样本。
    CSV 一般为 biased_samples_*/biased_samples_ranking.csv，包含列：index, fact_p_yes, cf_p_yes, fact_race, cf_race 等。

    Args:
        dataset: 完整 JSON 数据集（list of records）
        csv_path: CSV 路径
        sample_size: 取前 N 行的 index；若 <=0，则使用 CSV 中所有 index

    Returns:
        sampled_data: 按 CSV 顺序排列的样本列表
    """
    if not csv_path:
        raise ValueError("csv_path must be non-empty when using CSV-driven sampling.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    indices: List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "index" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'index' column. Got: {reader.fieldnames}")
        for row in reader:
            try:
                indices.append(int(row["index"]))
            except Exception:
                continue

    if sample_size and sample_size > 0:
        indices = indices[:sample_size]

    sampled: List[dict] = []
    for idx in indices:
        if 0 <= idx < len(dataset):
            sampled.append(dataset[idx])
        else:
            # 跳过越界 index
            continue
    return sampled

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate p(yes) probability differences across race groups for Resume dataset with intervention"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture for prompt formatting. Use 'auto' to infer from model/tokenizer.",
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
        default="intervention_results",
        help="Output directory for results.",
    )
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (not used in current implementation, kept for consistency).")
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling (balance by race). (Default: True)"
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling (use --no-balanced to disable)."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (only used when --random_sampling is enabled).")
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
        help="Directory containing selected_heads_elbow.json and results.pkl from analyze_race_sensitive_heads.py. "
             "If not provided, will try to infer from model_path. If empty and cannot infer, will run baseline.",
    )
    parser.add_argument(
        "--sensitive_heads_path",
        type=str,
        default="",
        help="Path to selected_heads_elbow.json file. Overrides --sensitive_heads_dir if provided.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="",
        help="Path to results.pkl file. Overrides --sensitive_heads_dir if provided.",
    )
    parser.add_argument(
        "--intervention_mode",
        type=str,
        default="mean_replacement",
        choices=["mean_replacement", "debias_projection", "zero_value"],
        help="Intervention mode to use.",
    )
    parser.add_argument(
        "--intervention_strength",
        type=float,
        default=1.0,
        help="Intervention strength (only used for debias_projection mode).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="If set, append results to this CSV file (for multi-model aggregation).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run baseline evaluation without intervention (overrides default intervention behavior).",
    )
    args = parser.parse_args()

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

    # # Filter: 只保留 category 为 "INFORMATION-TECHNOLOGY" 的样本
    # target_category = "INFORMATION-TECHNOLOGY"
    # dataset = [item for item in dataset if item.get("category") == target_category]
    # print(f"Filtered to {len(dataset)} samples with category '{target_category}'")

    # 采样数据：
    # 1) 若提供 --sample_csv_path，则按 CSV 的 index 顺序取样（覆盖其它采样参数）
    # 2) 否则使用原有的按种族平衡/随机采样逻辑
    if args.sample_csv_path:
        print("=" * 80)
        print(f"Sampling by CSV order: {args.sample_csv_path}")
        print("=" * 80)
        sampled_data = _load_samples_by_csv_indices(
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
    
    # 准备数据：从 summary 和 category 构建 query
    samples = []
    for idx, item in enumerate(sampled_data):
        summary = item.get("summary", "")
        category = item.get("category", "")
        race_str = item.get("race", "")
        
        if not summary or not category:
            continue
        
        # 构建完整的 query
        query = add_yes_no_instruction(summary)
        
        # 从 query 中提取种族（用于验证）
        extracted_race = extract_race_from_query(query)
        if extracted_race is None:
            # 如果从 query 中提取不到，使用原始 race 字段
            extracted_race = race_str
        
        race_group = _race_to_group(extracted_race)
        if race_group is None:
            continue
        
        samples.append({
            "id": item.get("ID", idx),
            "race": race_group,
            "race_str": extracted_race,
            "query": query,
            "summary": summary,
            "category": category,
        })

    if not samples:
        raise ValueError(
            "No valid samples found (need race in {White, Black})."
        )

    # Build prompts
    prompts = [s["query"] for s in samples]
    races = [s["race"] for s in samples]

    # Determine intervention paths
    # Priority: explicit paths > sensitive_heads_dir > auto-infer from model_path
    sensitive_heads_path = args.sensitive_heads_path
    embeddings_path = args.embeddings_path
    
    if not sensitive_heads_path or not embeddings_path:
        # Try to use sensitive_heads_dir
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            # Auto-infer: assume sensitive_heads_{model_name} directory exists in exp2
            model_name = os.path.basename(os.path.normpath(args.model_path))
            # Get exp2 directory (parent of current script's directory)
            exp2_dir = os.path.dirname(os.path.abspath(__file__))
            heads_dir = os.path.join(exp2_dir, f"sensitive_heads_{model_name}")
        
        if not sensitive_heads_path:
            candidate_path = os.path.join(heads_dir, "selected_heads_elbow.json")
            if os.path.exists(candidate_path):
                sensitive_heads_path = candidate_path
                print(f"Auto-detected sensitive_heads_path: {sensitive_heads_path}")
        
        if not embeddings_path:
            candidate_path = os.path.join(heads_dir, "results.pkl")
            if os.path.exists(candidate_path):
                embeddings_path = candidate_path
                print(f"Auto-detected embeddings_path: {embeddings_path}")
    
    # Check if intervention should be used
    # Default: use intervention (unless --baseline is specified)
    use_intervention = not args.baseline
    
    # Initialize sensitive_heads for later use in results
    sensitive_heads = []
    
    if use_intervention and (not sensitive_heads_path or not embeddings_path):
        # If intervention is requested but files are missing, raise error
        missing = []
        if not sensitive_heads_path:
            missing.append("selected_heads_elbow.json")
        if not embeddings_path:
            missing.append("results.pkl")
        raise FileNotFoundError(
            f"Intervention files not found. Missing: {', '.join(missing)}\n"
            f"Please run analyze_race_sensitive_heads.py first, or specify --sensitive_heads_path and --embeddings_path, "
            f"or use --baseline to run without intervention."
        )
    
    if use_intervention:
        print("=" * 80)
        print(f"Loading intervention data from {sensitive_heads_path} and {embeddings_path}")
        print("=" * 80)
        
        # Load selected heads
        with open(sensitive_heads_path, "r", encoding="utf-8") as f:
            selected_heads_data = json.load(f)
        sensitive_heads = [(h["layer"], h["head"]) for h in selected_heads_data]
        print(f"Loaded {len(sensitive_heads)} sensitive heads")
        
        # Load embeddings
        with open(embeddings_path, "rb") as f:
            embeddings_data = pickle.load(f)
        
        white_embeddings = embeddings_data.get("white_emb", {})
        black_embeddings = embeddings_data.get("black_emb", {})
        combined_stds = embeddings_data.get("combined_std", {})
        
        # Convert to tuple keys
        white_emb = {(int(k[0]), int(k[1])): v for k, v in white_embeddings.items() if isinstance(k, (tuple, list))}
        black_emb = {(int(k[0]), int(k[1])): v for k, v in black_embeddings.items() if isinstance(k, (tuple, list))}
        combined_std = {(int(k[0]), int(k[1])): v for k, v in combined_stds.items() if isinstance(k, (tuple, list))}
        
        # Get model config
        config = get_model_config(model)
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")
        print(f"Intervention mode: {args.intervention_mode}")
        
        # Filter sensitive heads to only those with embeddings
        valid_heads = []
        for l, h in sensitive_heads:
            if (l, h) in white_emb and (l, h) in black_emb:
                valid_heads.append((l, h))
        sensitive_heads = valid_heads
        print(f"Using {len(sensitive_heads)} heads with valid embeddings")
        
        if not sensitive_heads:
            raise ValueError("No valid sensitive heads found with embeddings.")
    else:
        print("=" * 80)
        print("Evaluating baseline p(yes) probabilities (no intervention)")
        print("=" * 80)

    # Compute p(yes) with or without intervention
    if use_intervention:
        # Compute with intervention - register hooks per prompt (safer for variable prompt lengths)
        print(f"Computing p(yes) with intervention mode: {args.intervention_mode}")
        intervention_results = []
        
        # DEBUG: 输出第一个样本用于调试
        if len(prompts) > 0:
            print("=" * 80)
            print("DEBUG: First sample to be processed by model (with intervention):")
            print(f"  Prompt: {prompts[0]}")
            print(f"  Race: {samples[0].get('race_str', 'Unknown')}")
            print(f"  Category: {samples[0].get('category', 'Unknown')}")
            print(f"  Intervention mode: {args.intervention_mode}")
            print("=" * 80)
        
        for idx, prompt in enumerate(tqdm(prompts, desc=f"Computing p(yes) with {args.intervention_mode}")):
            # Format prompt and get output position (needed for hooks)
            formatted_prompt = format_prompt_for_model(prompt, model_type)
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
            attention_mask = torch.ones_like(input_ids).to(input_device)
            
            # Get last token index (needed for hooks)
            from hook import get_last_token_indices_safe
            last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
            output_pos = int(last_token_indices[0].item())
            
            # Register hooks for this prompt
            prompt_hooks = []
            for l, h in sensitive_heads:
                if (l, h) not in white_emb or (l, h) not in black_emb:
                    continue
                    
                target_module = model.model.layers[l].self_attn.o_proj
                
                if args.intervention_mode == "mean_replacement":
                    from hook import make_intervention_hook_mean_replacement
                    mean_emb = (white_emb[(l, h)] + black_emb[(l, h)]) / 2.0
                    if isinstance(mean_emb, np.ndarray):
                        mean_emb = torch.from_numpy(mean_emb).float()
                    hook_fn = make_intervention_hook_mean_replacement(
                        l, h, mean_emb, output_pos, num_heads, head_dim
                    )
                    hook = target_module.register_forward_pre_hook(hook_fn)
                    prompt_hooks.append(hook)
                    
                elif args.intervention_mode == "debias_projection":
                    from hook import make_intervention_hook_debias_projection
                    white_emb_t = torch.from_numpy(white_emb[(l, h)]).float()
                    black_emb_t = torch.from_numpy(black_emb[(l, h)]).float()
                    combined_std_t = None
                    if (l, h) in combined_std:
                        combined_std_t = torch.from_numpy(combined_std[(l, h)]).float()
                    hook_fn = make_intervention_hook_debias_projection(
                        l, h, white_emb_t, black_emb_t, combined_std_t,
                        output_pos, args.intervention_strength, num_heads, head_dim
                    )
                    hook = target_module.register_forward_pre_hook(hook_fn)
                    prompt_hooks.append(hook)
                    
                elif args.intervention_mode == "zero_value":
                    from hook import make_intervention_hook_zero_value
                    hook_fn = make_intervention_hook_zero_value(l, h, output_pos, num_heads, head_dim)
                    hook = target_module.register_forward_pre_hook(hook_fn)
                    prompt_hooks.append(hook)
            
            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_row = outputs.logits[0, output_pos, :].float()
                    
                    # Use the utility function to compute p(yes) with warnings
                    p_yes = compute_p_yes_from_logits_with_warning(
                        logits_row=logits_row,
                        tokenizer=tokenizer,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                        sample_idx=idx,
                        show_warnings=True,
                        prefix=f"Intervention-{args.intervention_mode}",
                    )
                    intervention_results.append(float(p_yes))
            finally:
                # Remove hooks for this prompt
                remove_intervention_hooks(prompt_hooks)
            
            del input_ids, outputs, logits_row
        
        results_list = intervention_results
        intervention_suffix = f"_{args.intervention_mode}"
    else:
        # DEBUG: 输出第一个样本用于调试
        if len(prompts) > 0:
            print("=" * 80)
            print("DEBUG: First sample to be processed by model (baseline, no intervention):")
            print(f"  Prompt: {prompts[0]}")
            print(f"  Race: {samples[0].get('race_str', 'Unknown')}")
            print(f"  Category: {samples[0].get('category', 'Unknown')}")
            print("=" * 80)
        
        baseline_results = compute_p_yes_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=str(input_device),
            yes_ids=yes_ids,
            no_ids=no_ids,
            model_type=model_type,
            desc="Computing p(yes)",
            show_warnings=True,
        )
        results_list = baseline_results
        intervention_suffix = "_baseline"

    summary = summarize(results_list, races)

    results = {
        "model_info": {
            "model_path": args.model_path,
            "model_type": model_type,
            "input_device": str(input_device),
        },
        "dataset_info": {
            "dataset_json_path": args.dataset_json_path,
            "balanced": bool(args.balanced),
            "seed": int(args.seed),
            "max_samples": int(args.max_samples),
            "total_samples": int(len(samples)),
            "white_samples": int(sum(1 for r in races if r == 0)),
            "black_samples": int(sum(1 for r in races if r == 1)),
        },
        "intervention_info": {
            "use_intervention": use_intervention,
            "intervention_mode": args.intervention_mode if use_intervention else None,
            "intervention_strength": args.intervention_strength if use_intervention else None,
            "num_sensitive_heads": len(sensitive_heads) if use_intervention else 0,
        },
        "results": {
            **summary,
        },
    }

    results_path = os.path.join(args.output_dir, f"results{intervention_suffix}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {results_path}")

    per_sample = []
    for i, s in enumerate(samples):
        per_sample.append(
            {
                "id": s["id"],
                "race": "Black" if s["race"] == 1 else "White",
                "p_yes": float(results_list[i]),
                "category": s.get("category", ""),
            }
        )
    per_sample_path = os.path.join(args.output_dir, f"per_sample_results{intervention_suffix}.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    print(f"Saved: {per_sample_path}")

    # Save to CSV if path provided
    if args.csv_path:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        file_exists = os.path.exists(args.csv_path)
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "model", "intervention_mode", "intervention_strength",
                    "p_yes_mean", "p_yes_white_mean", "p_yes_black_mean",
                    "fairness_gap", "white_n", "black_n", "total_samples"
                ])
            
            writer.writerow([
                model_name,
                args.intervention_mode if use_intervention else "baseline",
                args.intervention_strength if use_intervention else 0.0,
                summary["p_yes_mean"],
                summary["p_yes_white_mean"],
                summary["p_yes_black_mean"],
                summary["fairness_gap_black_minus_white"],
                summary["white_n"],
                summary["black_n"],
                summary["n"],
            ])
        print(f"Appended to CSV: {args.csv_path}")

    print("\n" + "=" * 60)
    if use_intervention:
        print(f"P(YES) PROBABILITY DIFFERENCES BY RACE (Intervention: {args.intervention_mode})")
    else:
        print("BASELINE P(YES) PROBABILITY DIFFERENCES BY RACE")
    print("=" * 60)
    print(f"Overall:")
    print(f"  - Mean p(yes): {summary['p_yes_mean']:.6f} (n={summary['n']})")
    print(f"\nBy Race Group:")
    print(
        f"  - White mean p(yes): {summary['p_yes_white_mean']:.6f} (n={summary['white_n']})"
    )
    print(
        f"  - Black mean p(yes): {summary['p_yes_black_mean']:.6f} (n={summary['black_n']})"
    )
    print(f"\nFairness Gap:")
    print(
        f"  - Black - White: {summary['fairness_gap_black_minus_white']:.6f}"
    )
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
