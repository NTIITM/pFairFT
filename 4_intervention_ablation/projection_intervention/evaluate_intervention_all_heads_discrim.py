#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
under Global (All Heads) or Partial (Sensitive Heads) Projection Intervention.
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from probability import (
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
)
from prompt import (
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
    build_category_prompt
)
from sampling import load_discrim_eval_pairs
from util import (
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
)
from hook import (
    get_last_token_indices_safe,
    make_intervention_hook_debias_projection,
    remove_intervention_hooks,
    create_config_detection_hook,
)

def compute_stats_by_question(
    pairs: List[Tuple[int, int]],
    id_to_sample: Dict[int, dict],
    p_yes_map: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    bucket: Dict[int, List[float]] = defaultdict(list)
    for a, b in pairs:
        if a not in id_to_sample or b not in id_to_sample:
            continue
        qa = id_to_sample[a]["decision_question_id"]
        qb = id_to_sample[b]["decision_question_id"]
        if qa != qb: continue
        val_a, val_b = p_yes_map.get(a), p_yes_map.get(b)
        if val_a is None or val_b is None or math.isnan(val_a) or math.isnan(val_b):
            continue
        bucket[qa].append(abs(val_a - val_b))
    stats = {}
    for qid, diffs in bucket.items():
        arr = np.array(diffs, dtype=np.float64)
        stats[qid] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "count": len(diffs)}
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="exp25_results")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--sensitive_heads_dir", type=str, default="")
    parser.add_argument("--intervention_mode", type=str, choices=["all", "partial"], default="all")
    parser.add_argument("--intervention_strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    id_to_sample = {int(item["id"]): item for item in data}

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, "cuda")
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    yes_ids, no_ids = get_target_token_ids(tokenizer, YES_CANDIDATES), get_target_token_ids(tokenizer, NO_CANDIDATES)

    if not args.sensitive_heads_dir:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.sensitive_heads_dir = f"/home/common1/hwluo/project/pFairFT/exp2/sensitive_heads_{model_name}_top100"

    with open(os.path.join(args.sensitive_heads_dir, "results.pkl"), "rb") as f:
        embeddings_data = pickle.load(f)
    
    white_emb = {(int(k[0]), int(k[1])): v for k, v in embeddings_data.get("white_emb", {}).items() if isinstance(k, (tuple, list))}
    black_emb = {(int(k[0]), int(k[1])): v for k, v in embeddings_data.get("black_emb", {}).items() if isinstance(k, (tuple, list))}
    
    if args.intervention_mode == "partial":
        with open(os.path.join(args.sensitive_heads_dir, "selected_heads_elbow.json"), "r") as f:
            target_heads = [(h["layer"], h["head"]) for h in json.load(f)]
    else: # all heads that have embeddings
        target_heads = list(white_emb.keys())

    config = get_model_config(model)
    num_heads, head_dim = config["num_heads"], config["head_dim"]
    
    # Config detection
    temp_buffer = {}
    detect_hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(create_config_detection_hook(temp_buffer))
    try:
        test_in = tokenizer([format_prompt_for_model(add_yes_no_instruction(build_category_prompt(data[0]["prompt"],"")), model_type)], return_tensors="pt", add_special_tokens=False).to(input_device)
        with torch.no_grad(): _ = model(**test_in)
    finally: detect_hook.remove()
    num_heads, head_dim = temp_buffer.get("num_heads", num_heads), temp_buffer.get("head_dim", head_dim)

    p_yes_results = []
    prompts = [add_yes_no_instruction(build_category_prompt(item["prompt"],"")) for item in data]

    for item, prompt in tqdm(zip(data, prompts), total=len(data), desc=f"Eval {args.intervention_mode}"):
        f_prompt = format_prompt_for_model(prompt, model_type)
        ids = tokenizer.encode(f_prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
        mask = torch.ones_like(ids).to(input_device)
        pos = int(get_last_token_indices_safe(ids, mask, tokenizer)[0].item())

        hooks = []
        for l, h in target_heads:
            if (l, h) in white_emb and (l, h) in black_emb:
                hook_fn = make_intervention_hook_debias_projection(l, h, torch.from_numpy(white_emb[(l, h)]).float(), torch.from_numpy(black_emb[(l, h)]).float(), None, pos, args.intervention_strength, num_heads, head_dim)
                hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_fn))
        try:
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
                p_yes = compute_p_yes_from_logits_with_warning(out.logits[0, pos, :].float(), tokenizer, yes_ids, no_ids)
                p_yes_results.append(float(p_yes))
        finally: remove_intervention_hooks(hooks)

    if args.csv_path:
        file_exists = os.path.exists(args.csv_path)
        m_name = os.path.basename(os.path.normpath(args.model_path))
        matched_map = {a: b for a, b in pairs}; matched_map.update({b: a for a, b in pairs})
        with open(args.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["sample_id", "matched_id", "model", "decision_question_id", "p_yes", "intervention_type"])
            for s, p in zip(data, p_yes_results):
                writer.writerow([s["id"], matched_map.get(int(s["id"]), ""), m_name, s.get("decision_question_id", ""), p, f"debias_{args.intervention_mode}"])

if __name__ == "__main__":
    main()
