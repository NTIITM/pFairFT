#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
UNDER IGBP PROJECTION INTERVENTION.
"""

import argparse
import csv
import json
import math
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from probability import YES_CANDIDATES, NO_CANDIDATES, get_target_token_ids
from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type, build_category_prompt
from sampling import load_discrim_eval_pairs
from util import compute_p_yes_from_logits_with_warning, get_input_device
from hook import get_last_token_indices_safe, remove_intervention_hooks
from igbp_hook import make_intervention_hook_igbp

import torch.nn as nn
class Probe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

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
        if qa != qb:
            continue
        val_a = p_yes_map.get(a)
        val_b = p_yes_map.get(b)

        if val_a is None or val_b is None or math.isnan(val_a) or math.isnan(val_b):
            continue
        bucket[qa].append(abs(val_a - val_b))

    stats: Dict[int, Dict[str, float]] = {}
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
    parser = argparse.ArgumentParser(description="Evaluate p(yes) differences under IGBP intervention.")
    parser.add_argument("--dataset_path", type=str, default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "deepseek"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt_type", type=str, default="prompt")
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="intervention_results")
    parser.add_argument("--csv_path", type=str, default="", help="Append per-sample results to this CSV.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    id_to_sample: Dict[int, dict] = {int(item["id"]): item for item in data}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)

    print("Loading IGBP Probes...")
    probe_path = args.probe_path
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probes not found at {probe_path}! Did you run train_igbp_probes.py first?")
    probes = torch.load(probe_path, map_location=args.device, weights_only=False)

    p_yes_results = []
    prompts = [add_yes_no_instruction(build_category_prompt(item[args.prompt_type],"")) for item in data]

    target_module = model.model.norm

    for idx, (sample, prompt) in enumerate(tqdm(zip(data, prompts), total=len(data), desc="Evaluating")):
        formatted_prompt = format_prompt_for_model(prompt, model_type)
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
        attention_mask = torch.ones_like(input_ids).to(input_device)

        last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
        output_pos = int(last_token_indices[0].item())

        prompt_hooks = []
        hook_fn = make_intervention_hook_igbp(probes)
        hook = target_module.register_forward_hook(hook_fn)
        prompt_hooks.append(hook)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_row = outputs.logits[0, output_pos, :].float()
                p_yes = compute_p_yes_from_logits_with_warning(logits_row, tokenizer, yes_ids, no_ids)
                p_yes_results.append(float(p_yes))
        finally:
            remove_intervention_hooks(prompt_hooks)

    p_yes_map = {int(s["id"]): p for s, p in zip(data, p_yes_results)}
    stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)
    
    if args.csv_path:
        file_exists = os.path.exists(args.csv_path)
        model_name = os.path.basename(os.path.normpath(args.model_path))
        matched_map = {}
        for a, b in pairs:
            matched_map[a], matched_map[b] = b, a
        
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["sample_id", "matched_id", "model", "decision_question_id", "p_yes", "intervention_type"])
            for s, p in zip(data, p_yes_results):
                sid = int(s["id"])
                writer.writerow([sid, matched_map.get(sid, ""), model_name, s.get("decision_question_id", ""), p, "igbp"])

    print(f"Overall Mean Gap: {np.mean([s['mean'] for s in stats.values()]):.6f}")

if __name__ == "__main__":
    main()
