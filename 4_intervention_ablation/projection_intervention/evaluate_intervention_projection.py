#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
UNDER DEBIAS PROJECTION INTERVENTION.

功能：
- 复用 exp8 的数据处理逻辑（discrim-eval 配对样本，计算 p(yes) 差值）。
- 复用 exp2 的 debias_projection 逻辑（通过投影消除敏感方向的偏见）。
- 按 decision_question_id 统计结果。
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

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from probability import (  # noqa: E402
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
)
from prompt import (  # noqa: E402
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
    build_category_prompt
)
from sampling import load_discrim_eval_pairs  # noqa: E402
from util import (  # noqa: E402
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
)
from hook import (  # noqa: E402
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
    """Aggregate absolute p_yes differences by decision_question_id."""
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

        if (
            val_a is None
            or val_b is None
            or math.isnan(val_a)
            or math.isnan(val_b)
        ):
            continue

        diff = abs(val_a - val_b)
        bucket[qa].append(diff)

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
    parser = argparse.ArgumentParser(
        description="Evaluate p(yes) differences under debias_projection intervention."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_results",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Append per-sample results to this CSV.",
    )
    parser.add_argument(
        "--sensitive_heads_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--intervention_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    id_to_sample: Dict[int, dict] = {int(item["id"]): item for item in data}

    # 2. Load Model & Tokenizer
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

    # 3. Load Intervention Data
    if not args.sensitive_heads_dir:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.sensitive_heads_dir = os.path.join("/home/common1/hwluo/project/pFairFT/exp2", f"sensitive_heads_{model_name}_top100")

    sensitive_heads_path = os.path.join(args.sensitive_heads_dir, "selected_heads_elbow.json")
    embeddings_path = os.path.join(args.sensitive_heads_dir, "results.pkl")

    with open(sensitive_heads_path, "r", encoding="utf-8") as f:
        selected_heads_data = json.load(f)
    sensitive_heads = [(h["layer"], h["head"]) for h in selected_heads_data]

    with open(embeddings_path, "rb") as f:
        embeddings_data = pickle.load(f)

    white_emb = {
        (int(k[0]), int(k[1])): v
        for k, v in embeddings_data.get("white_emb", {}).items()
        if isinstance(k, (tuple, list))
    }
    black_emb = {
        (int(k[0]), int(k[1])): v
        for k, v in embeddings_data.get("black_emb", {}).items()
        if isinstance(k, (tuple, list))
    }
    combined_std = {
        (int(k[0]), int(k[1])): v
        for k, v in embeddings_data.get("combined_std", {}).items()
        if isinstance(k, (tuple, list))
    }

    config = get_model_config(model)
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]

    # 4. Compute p(yes)
    p_yes_results = []
    prompts = [add_yes_no_instruction(build_category_prompt(item[args.prompt_type],"")) for item in data]

    # --- Runtime config detection (same idea as exp2/analyze_race_sensitive_heads.py) ---
    # Some models may have incorrect / inconsistent config fields; detect actual head config
    # from a real forward pass through o_proj.
    temp_buffer: Dict[str, object] = {}
    detect_hook_fn = create_config_detection_hook(temp_buffer)
    detect_hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(detect_hook_fn)
    try:
        if len(data) == 0:
            raise ValueError("Empty dataset; cannot run runtime config detection.")
        test_prompt_raw = prompts[0]
        test_prompt = format_prompt_for_model(test_prompt_raw, model_type)
        test_inputs = tokenizer(
            [test_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        with torch.no_grad():
            _ = model(**test_inputs)
    finally:
        detect_hook.remove()

    detected_num_heads = temp_buffer.get("num_heads")
    detected_head_dim = temp_buffer.get("head_dim")
    if detected_num_heads is None or detected_head_dim is None:
        raise RuntimeError(
            "Could not detect model head configuration at runtime (num_heads/head_dim missing)."
        )

    if int(detected_num_heads) != int(num_heads) or int(detected_head_dim) != int(head_dim):
        print("Detected configuration mismatch!")
        print(f"  Initial: num_heads={num_heads}, head_dim={head_dim}")
        print(f"  Actual:  num_heads={detected_num_heads}, head_dim={detected_head_dim}")
        print("Updating configuration to detected values")
        num_heads = int(detected_num_heads)
        head_dim = int(detected_head_dim)

    for idx, (sample, prompt) in enumerate(tqdm(zip(data, prompts), total=len(data), desc="Evaluating")):
        formatted_prompt = format_prompt_for_model(prompt, model_type)
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
        attention_mask = torch.ones_like(input_ids).to(input_device)

        last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
        output_pos = int(last_token_indices[0].item())

        prompt_hooks = []
        for l, h in sensitive_heads:
            if (l, h) not in white_emb or (l, h) not in black_emb:
                continue
            
            target_module = model.model.layers[l].self_attn.o_proj
            
            w_emb = torch.from_numpy(white_emb[(l, h)]).float()
            b_emb = torch.from_numpy(black_emb[(l, h)]).float()
            c_std = torch.from_numpy(combined_std[(l, h)]).float() if (l, h) in combined_std else None

            hook_fn = make_intervention_hook_debias_projection(
                l, h, w_emb, b_emb, c_std, output_pos, args.intervention_strength, num_heads, head_dim, use_std=False
            )
            hook = target_module.register_forward_pre_hook(hook_fn)
            prompt_hooks.append(hook)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_row = outputs.logits[0, output_pos, :].float()
                p_yes = compute_p_yes_from_logits_with_warning(logits_row, tokenizer, yes_ids, no_ids)
                p_yes_results.append(float(p_yes))
        finally:
            remove_intervention_hooks(prompt_hooks)

    # 5. Save Results
    p_yes_map = {int(s["id"]): p for s, p in zip(data, p_yes_results)}
    stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)
    
    # Save CSV
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
                writer.writerow([sid, matched_map.get(sid, ""), model_name, s.get("decision_question_id", ""), p, "debias_projection"])

    print(f"Overall Mean Gap: {np.mean([s['mean'] for s in stats.values()]):.6f}")

if __name__ == "__main__":
    main()
