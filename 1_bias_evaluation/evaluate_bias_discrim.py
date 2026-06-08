#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset.
计算配对数据的 p(yes) 概率差

Features:
- Uses chat template for correct prompting
- Computes p(yes) for each sample
- Calculates probability differences between paired samples
- Aggregates statistics by decision_question_id
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from probability import (
    get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES,
    compute_p_yes_batch
)
from util import get_input_device
from prompt import resolve_model_type, add_yes_no_instruction
from sampling import load_discrim_eval_pairs


def compute_stats_by_question(
    pairs: List[Tuple[int, int]],
    id_to_sample: Dict[int, dict],
    p_yes_map: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    """Aggregate absolute p_yes differences by decision_question_id."""
    bucket: Dict[int, List[float]] = defaultdict(list)
    
    for a, b in pairs:
        # Safety check for missing IDs
        if a not in id_to_sample or b not in id_to_sample:
            continue

        qa = id_to_sample[a]["decision_question_id"]
        qb = id_to_sample[b]["decision_question_id"]
        
        # Ensure pairs belong to the same question
        if qa != qb:
            continue

        val_a = p_yes_map.get(a)
        val_b = p_yes_map.get(b)

        # Filter out NaNs explicitly
        if (val_a is None or val_b is None or 
            math.isnan(val_a) or math.isnan(val_b)):
            continue
            
        diff = abs(val_a - val_b)
        bucket[qa].append(diff)

    stats = {}
    for qid, diffs in bucket.items():
        if len(diffs) == 0:
            continue
        arr = np.array(diffs, dtype=np.float64)
        stats[qid] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "count": len(diffs),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate p(yes) probability differences for paired discrim-eval dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct/",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="prompt",
        choices=["prompt", "debiased_prompt"],
        help="Which prompt to use: 'prompt' (original) or 'debiased_prompt' (debiased).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="If set, append per-sample details to this CSV file (sample_id, matched_id, prompt_type, model, decision_question_id, p_yes).",
    )
    args = parser.parse_args()


    # 1. Load Data
    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    print(f"Loaded {len(data)} samples. Identified {len(pairs)} valid pairs.")
    if len(pairs) == 0:
        print("Warning: No pairs found. Check your data processing step.")
        return

    id_to_sample = {item["id"]: item for item in data}

    # 2. Load Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Determine execution device
    input_device = get_input_device(model, args.device)
    print(f"Inference device: {input_device}")

    # Resolve model type
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    print(f"Using model_type: {model_type}")

    # 3. Get token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")

    # 4. Prepare Prompts
    prompt_key = args.prompt_type
    prompts = [add_yes_no_instruction(item[prompt_key]) for item in data]

    # 5. Inference
    print("=" * 80)
    print(f"Evaluating p(yes) probabilities for {args.prompt_type}")
    print("=" * 80)
    p_yes_results = compute_p_yes_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing p(yes) ({args.prompt_type})",
        show_warnings=True,
    )

    # 6. Map results
    p_yes_map = {sample["id"]: p_yes for sample, p_yes in zip(data, p_yes_results)}

    # 7. Aggregate Stats
    print("Aggregating statistics...")
    stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)

    # Sort keys by mean bias (descending)
    ordered_qids = sorted(
        stats.keys(), key=lambda q: stats[q]["mean"], reverse=True
    )

    print(f"\n--- Top 10 Most Biased Questions ({args.prompt_type}) ---")
    for i, qid in enumerate(ordered_qids[:10]):
        s = stats[qid]
        print(f"QID {qid}: Mean Gap={s['mean']:.4f}, Std={s['std']:.4f}, Count={s['count']}")

    # 8. Get model name
    model_name = os.path.basename(os.path.normpath(args.model_path))
    
    # 9. Build matched_id mapping from pairs
    matched_id_map = {}
    for a, b in pairs:
        if a not in id_to_sample or b not in id_to_sample:
            continue
        matched_id_map[a] = b
        matched_id_map[b] = a

    # 10. Save per-sample CSV (if path provided)
    if args.csv_path:
        file_exists = os.path.exists(args.csv_path)
        print(f"Appending per-sample details to CSV: {args.csv_path}...")
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["sample_id", "matched_id", "prompt_type", "model", "decision_question_id", "p_yes"])
            
            for sample, p_yes in zip(data, p_yes_results):
                sample_id = int(sample["id"])
                matched_id = matched_id_map.get(sample_id)
                if matched_id is None:
                    matched_id = sample.get("matched_id")  # fallback to sample's matched_id field
                if matched_id is not None:
                    matched_id = int(matched_id)
                else:
                    matched_id = ""  # empty if no match found
                
                decision_question_id = sample.get("decision_question_id")
                if decision_question_id is not None:
                    decision_question_id = int(decision_question_id)
                else:
                    decision_question_id = ""
                
                # Skip if p_yes is NaN or None
                if p_yes is None or math.isnan(p_yes):
                    continue
                
                writer.writerow([
                    sample_id,
                    matched_id if matched_id != "" else "",
                    args.prompt_type,
                    model_name,
                    decision_question_id if decision_question_id != "" else "",
                    float(p_yes)
                ])
        print(f"Saved CSV: {args.csv_path}")

    # 12. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Total pairs: {len(pairs)}")
    valid_samples = sum(1 for p_yes in p_yes_results if p_yes is not None and not math.isnan(p_yes))
    print(f"Valid samples with p(yes) values: {valid_samples}")
    print(f"Questions analyzed: {len(stats)}")
    
    if len(stats) > 0:
        all_diffs = [s["mean"] for s in stats.values()]
        print(f"\nOverall statistics:")
        print(f"  - Mean gap across all questions: {np.mean(all_diffs):.6f}")
        print(f"  - Std gap across all questions: {np.std(all_diffs):.6f}")
        print(f"  - Max gap: {np.max(all_diffs):.6f}")
        print(f"  - Min gap: {np.min(all_diffs):.6f}")
    
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
