#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
UNDER NEGATIVE / RANDOM NEGATIVE HEAD-LEVEL INTERVENTION, with varying head counts.

该脚本会：
1. 加载 discrim-eval 数据集与配对信息
2. 从 results.pkl 加载 heatmap 及 embedding，选取敏感头 / 非敏感头
3. 循环不同数量的头（0, step, 2*step...），每次选择前N个头进行干预（敏感头模式），或随机选择N个非敏感头（随机模式）
4. 对每个样本计算干预下的 p(yes) 概率
5. 对每个 decision_question_id，计算配对样本的 p(yes) 绝对差的均值
6. 将结果保存为 CSV 文件：head_count, model_name, decision_question_id, mean_p_yes_gap, intervention_type
"""

import argparse
import csv
import math
import os
import pickle
import sys
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from probability import (
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
    compute_p_yes_batch, # For baseline (head_count=0)
)
from prompt import (
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
    build_category_prompt,
)
from sampling import load_discrim_eval_pairs
from util import (
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
    load_intervention_results,
    get_sensitive_heads_sorted_by_heatmap,
    get_non_sensitive_heads_from_results,
)
from hook import (
    get_last_token_indices_safe,
    remove_intervention_hooks,
)
from model_adapter import get_model_adapter


def compute_p_yes_with_intervention(
    model,
    adapter,
    tokenizer,
    prompt: str,
    model_type: str,
    heads_for_intervention: List[Tuple[int, int]],
    white_emb: Dict[Tuple[int, int], np.ndarray],
    black_emb: Dict[Tuple[int, int], np.ndarray],
    input_device: torch.device,
    yes_ids: List[int],
    no_ids: List[int],
    num_heads: int,
    head_dim: int,
) -> float:
    """计算带负向干预（mean ablation）的p(yes)概率。"""
    formatted_prompt = format_prompt_for_model(prompt, model_type)
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(input_device)
    attention_mask = torch.ones_like(input_ids).to(input_device)

    last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
    output_pos = int(last_token_indices[0].item())

    prompt_hooks = []
    for l, h in heads_for_intervention:
        if (l, h) not in white_emb or (l, h) not in black_emb:
            continue

        # 负向干预：mean ablation
        mean_emb_np = (white_emb[(l, h)] + black_emb[(l, h)]) / 2.0
        mean_emb = (
            torch.from_numpy(mean_emb_np).float()
            if isinstance(mean_emb_np, np.ndarray)
            else mean_emb_np
        )
        hook = adapter.register_head_mean_replacement_hook(
            l, h, mean_emb, output_pos, num_heads, head_dim
        )
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
                prefix=f"Intervention-negative",
            )
            return float(p_yes)
    finally:
        remove_intervention_hooks(prompt_hooks)

    return 0.0


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
        description=(
            "Evaluate p(yes) probability differences for paired discrim-eval "
            "dataset under NEGATIVE / RANDOM NEGATIVE head-level intervention, "
            "with varying head counts."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
        help="Path to the discrim-eval paired dataset JSON file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_discrim_eval_head_count_results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="",
        help="Path to results.pkl (from analyze_race_sensitive_heads.py, containing heatmap and embeddings).",
    )
    parser.add_argument(
        "--sensitive_heads_dir",
        type=str,
        default="",
        help="Directory containing results.pkl. If not provided, will try to infer from model_path.",
    )
    parser.add_argument(
        "--intervention_mode",
        type=str,
        default="negative",
        choices=["negative", "negative_random"],
        help=(
            "Intervention strategy: "
            "'negative' uses sorted race-sensitive heads; "
            "'negative_random' randomly selects the same number of non-sensitive heads "
            "for mean ablation."
        ),
    )
    parser.add_argument(
        "--max_head_count",
        type=int,
        default=100,
        help="Maximum number of heads to test (e.g. 5*step for 6 points from 0 to max_head_count).",
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
        default="intervention_results_discrim_eval_by_head_count.csv",
        help="Output CSV filename under output_dir.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible head sampling.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    print(f"Loaded {len(data)} samples. Identified {len(pairs)} valid pairs.")
    if len(pairs) == 0:
        print("Warning: No pairs found. Check your data processing step.")
        return

    id_to_sample: Dict[int, dict] = {int(item["id"]): item for item in data}

    # 2. Load Model & Tokenizer
    print("=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto"
        if args.device == "cuda" and torch.cuda.is_available()
        else None,
        torch_dtype=(
            torch.float16
            if args.device == "cuda" and torch.cuda.is_available()
            else torch.float32
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, args.device)
    print(f"Inference device: {input_device}")

    model_type = resolve_model_type(
        args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Using model_type: {model_type}")
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")

    # 3. Load sensitive heads & embeddings (from exp2_old results.pkl)
    embeddings_path = args.embeddings_path

    if not embeddings_path:
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            # 假设 exp2_old 结果与此脚本在同一父目录下
            exp2_old_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp2_old"
            )
            heads_dir = os.path.join(exp2_old_dir, f"sensitive_heads_{model_name}_top100")
        candidate_path = os.path.join(heads_dir, "results.pkl")
        if os.path.exists(candidate_path):
            embeddings_path = candidate_path
            print(f"Auto-detected embeddings_path: {embeddings_path}")

    if not embeddings_path or not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"results.pkl not found. Please run analyze_race_sensitive_heads.py first, "
            f"or specify --embeddings_path or --sensitive_heads_dir. Tried: {embeddings_path}"
        )

    print("=" * 80)
    print(f"Loading intervention data from {embeddings_path} (heatmap-based head selection)")
    print("=" * 80)

    results_data = load_intervention_results(embeddings_path)
    all_sensitive_heads = get_sensitive_heads_sorted_by_heatmap(results_data)
    print(f"Loaded {len(all_sensitive_heads)} sensitive heads (sorted by heatmap KL)")

    white_embeddings = results_data.get("white_emb", {})
    black_embeddings = results_data.get("black_emb", {})
    white_emb: Dict[Tuple[int, int], np.ndarray] = {
        (int(k[0]), int(k[1])): v for k, v in white_embeddings.items() if isinstance(k, (tuple, list))
    }
    black_emb: Dict[Tuple[int, int], np.ndarray] = {
        (int(k[0]), int(k[1])): v for k, v in black_embeddings.items() if isinstance(k, (tuple, list))
    }

    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")

    if not all_sensitive_heads:
        raise ValueError("No valid sensitive heads found with embeddings.")
    print(f"Using {len(all_sensitive_heads)} heads for sensitive intervention candidates.")

    non_sensitive_heads: List[Tuple[int, int]] = []
    if args.intervention_mode == "negative_random":
        non_sensitive_heads = get_non_sensitive_heads_from_results(results_data)
        if not non_sensitive_heads:
            print("Warning: No non-sensitive heads found. Random intervention not possible.")

    # 4. Main loop: iterate head_count
    # head_counts = [0, step, 2*step, ..., max_head_count]
    max_n = min(args.max_head_count, len(all_sensitive_heads))
    if args.intervention_mode == "negative_random":
        max_n = min(max_n, len(non_sensitive_heads))
    head_counts = list(range(0, max_n + 1, args.step))
    if max_n not in head_counts:
        head_counts.append(max_n)
    head_counts = sorted(set(head_counts))

    print(f"Will test head counts: {head_counts}")
    selected_heads_by_count: Dict[str, List[List[int]]] = {}

    # 准备CSV输出
    csv_path = os.path.join(args.output_dir, args.results_csv_name)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "head_count",
        "model_name",
        "decision_question_id",
        "mean_p_yes_gap",
        "intervention_type",
    ])

    model_name = os.path.basename(os.path.normpath(args.model_path))
    intervention_type_str = args.intervention_mode

    for head_count in head_counts:
        print("=" * 80)
        print(f"Testing with {head_count} heads (intervention type: {intervention_type_str})")
        print("=" * 80)

        current_heads: List[Tuple[int, int]] = []
        if head_count > 0:
            if intervention_type_str == "negative":
                current_heads = all_sensitive_heads[:head_count]
            elif intervention_type_str == "negative_random":
                if len(non_sensitive_heads) < head_count:
                    print(
                        f"Warning: Not enough non-sensitive heads ({len(non_sensitive_heads)}) "
                        f"to sample {head_count} for random intervention. Using all available."
                    )
                    current_heads = random.sample(non_sensitive_heads, len(non_sensitive_heads))
                else:
                    current_heads = random.sample(non_sensitive_heads, head_count)
            
            if not current_heads:
                print("No heads selected for intervention. Skipping.")
                continue
            print(f"Using {len(current_heads)} heads: {current_heads[:5]}..." if len(current_heads) > 5 else f"Using heads: {current_heads}")
        else:
            print("Using baseline (no intervention)")
        selected_heads_by_count[str(head_count)] = [list(head) for head in current_heads]

        # 计算每个样本的 p(yes) - 无论是否干预
        p_yes_map: Dict[int, float] = {}
        # Baseline (head_count=0) 需要先计算原始 prompt 的 p_yes
        # 然后再计算干预后的 p_yes。但这里我们是计算干预下的 bias，
        # 所以对于 head_count=0，我们计算 baseline 的 discrim-eval bias。
        # 对于 head_count > 0，我们计算干预后的 discrim-eval bias。

        # 0. 准备 prompts
        prompts: List[str] = [
            add_yes_no_instruction(item["prompt"]) for item in data
            # add_yes_no_instruction(build_category_prompt(item["prompt"], "")) for item in data
        ]
        
        # 1. 计算所有样本的 p(yes) （在当前干预头数量和模式下）
        p_yes_results_intervened: List[float] = []
        if head_count == 0:
            # Baseline: 无干预，使用批量计算
            p_yes_results_intervened = compute_p_yes_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=str(input_device),
                yes_ids=yes_ids,
                no_ids=no_ids,
                model_type=model_type,
                desc=f"Computing baseline p(yes)",
                show_warnings=False,
            )
        else:
            for idx, prompt in enumerate(tqdm(
                prompts,
                desc=f"Computing p(yes) with {len(current_heads)} heads"
            )):
                p_yes = compute_p_yes_with_intervention(
                    model=model,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    model_type=model_type,
                    heads_for_intervention=current_heads,
                    white_emb=white_emb,
                    black_emb=black_emb,
                    input_device=input_device,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                p_yes_results_intervened.append(p_yes)

        # 2. Map results by sample_id
        p_yes_map = {int(sample["id"]): p_yes for sample, p_yes in zip(data, p_yes_results_intervened)}

        # 3. Compute stats by question
        stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)

        # 4. Write results to CSV
        for qid, stat_data in stats.items():
            csv_writer.writerow([
                head_count,
                model_name,
                qid,
                f"{stat_data['mean']:.6f}",
                intervention_type_str,
            ])
    
    csv_file.close()
    metadata = {
        "experiment": "discrim_head_count",
        "model_path": args.model_path,
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "dataset_path": args.dataset_path,
        "embeddings_path": embeddings_path,
        "intervention_mode": args.intervention_mode,
        "head_counts": head_counts,
        "selected_heads_by_count": selected_heads_by_count,
        "seed": args.seed,
        "rows": len(data),
        "pairs": len(pairs),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved results to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
