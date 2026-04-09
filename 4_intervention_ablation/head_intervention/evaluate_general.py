#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate baseline p(yes) probability differences across race groups for Discrim-Eval dataset
比较不同种族（White vs Black）的 p(yes) 概率差异

参考 evaluate_intervention.py 的结构，但使用 discrim-eval 数据集
"""

import csv
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import resolve_model_type, add_yes_no_instruction
from probability import (
    get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES,
    compute_p_yes_batch, p_yes_from_logits_stable
)
from util import get_input_device, get_model_config
from sampling import load_discrim_eval_pairs
from hook import (
    get_last_token_indices_safe,
    register_intervention_hooks,
    remove_intervention_hooks
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate baseline p(yes) probability differences across race groups for Discrim-Eval dataset"
    )
    parser.add_argument("--model_path", default="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct/", type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture for prompt formatting in compute_p_yes_batch. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp3/general_results",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="prompt",
        choices=["prompt", "debiased_prompt"],
        help="Which prompt to use: 'prompt' (original) or 'debiased_prompt' (debiased).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sensitive_heads_dir",
        type=str,
        default="",
        help="Directory containing selected_heads_elbow.json and results.pkl from analyze_race_sensitive_heads.py. "
             "If not provided, will try to infer from model_path.",
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

    # Load dataset (we only need the data, not the pairs for this analysis)
    data, _ = load_discrim_eval_pairs(args.dataset_json_path)
    print(f"Loaded {len(data)} samples.")

    # Prepare samples with race information
    samples = []
    for item in data:
        race_str = item.get("race", "")
        race_group = _race_to_group(race_str)
        if race_group is None:
            continue
        
        prompt_text = item.get(args.prompt_type, "")
        if not prompt_text:
            continue
        
        samples.append({
            "id": item.get("id"),
            "race": race_group,
            "race_str": race_str,
            "prompt": prompt_text,
            "decision_question_id": item.get("decision_question_id"),
        })

    if not samples:
        raise ValueError(
            "No valid samples found (need race in {white, black})."
        )

    # Print sample statistics
    white_count = sum(1 for s in samples if s["race"] == 0)
    black_count = sum(1 for s in samples if s["race"] == 1)
    print(f"Total samples: {len(samples)}")
    print(f"  - White: {white_count}, Black: {black_count}")

    # Build prompts
    prompts = [add_yes_no_instruction(s["prompt"]) for s in samples]
    races = [s["race"] for s in samples]

    # Determine intervention paths
    sensitive_heads_path = args.sensitive_heads_path
    embeddings_path = args.embeddings_path
    
    if not sensitive_heads_path or not embeddings_path:
        # Try to use sensitive_heads_dir
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            # Auto-infer: assume sensitive_heads_{model_name} directory exists in exp2
            model_name = os.path.basename(os.path.normpath(args.model_path))
            # Get exp2 directory
            exp2_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp2")
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
            f"Please run exp_2_1.sh first, or specify --sensitive_heads_path and --embeddings_path, "
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
        print(f"Evaluating baseline p(yes) probabilities for {args.prompt_type} (no intervention)")
        print("=" * 80)

    # Compute p(yes) with or without intervention
    if use_intervention:
        # Compute with intervention - register hooks per prompt (safer for variable prompt lengths)
        print(f"Computing p(yes) with intervention mode: {args.intervention_mode}")
        intervention_results = []
        
        for idx, prompt in enumerate(tqdm(prompts, desc=f"Computing p(yes) with {args.intervention_mode}")):
            # Format prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
            attention_mask = torch.ones_like(input_ids).to(input_device)
            
            # Get last token index
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
                    p_yes = p_yes_from_logits_stable(logits_row, yes_ids=yes_ids, no_ids=no_ids)
                    intervention_results.append(float(p_yes))
            finally:
                # Remove hooks for this prompt
                remove_intervention_hooks(prompt_hooks)
            
            del input_ids, outputs, logits_row
        
        results_list = intervention_results
        intervention_suffix = f"_{args.intervention_mode}"
    else:
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
            "prompt_type": args.prompt_type,
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

    results_path = os.path.join(args.output_dir, f"results_{args.prompt_type}{intervention_suffix}.json")
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
                "decision_question_id": s.get("decision_question_id"),
            }
        )
    per_sample_path = os.path.join(args.output_dir, f"per_sample_results_{args.prompt_type}{intervention_suffix}.json")
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
                    "model", "prompt_type", "intervention_mode", "intervention_strength",
                    "p_yes_mean", "p_yes_white_mean", "p_yes_black_mean",
                    "fairness_gap", "white_n", "black_n", "total_samples"
                ])
            
            writer.writerow([
                model_name,
                args.prompt_type,
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
        print(f"P(YES) PROBABILITY DIFFERENCES BY RACE ({args.prompt_type.upper()}, Intervention: {args.intervention_mode})")
    else:
        print(f"BASELINE P(YES) PROBABILITY DIFFERENCES BY RACE ({args.prompt_type.upper()})")
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
