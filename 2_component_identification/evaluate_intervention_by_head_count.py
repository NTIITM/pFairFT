#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在exp8的干预基础上，以5为步长干预不同数量的头，计算事实与反事实数据，保存为CSV。

该脚本会：
1. 加载敏感头数据
2. 循环不同的头数量（5, 10, 15, 20...），每次选择前N个头进行干预
3. 对每个样本计算事实和反事实概率
4. 将结果保存为CSV文件
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
from hook import (
    get_last_token_indices_safe,
    remove_intervention_hooks,
    make_positive_direction_hook,
)
from model_adapter import get_model_adapter


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
    adapter,
    tokenizer,
    prompt: str,
    model_type: str,
    sensitive_heads: List[Tuple[int, int]],
    white_emb: Dict[Tuple[int, int], np.ndarray],
    black_emb: Dict[Tuple[int, int], np.ndarray],
    head_directions: Optional[Dict[Tuple[int, int], np.ndarray]],
    head_std: Optional[Dict[Tuple[int, int], np.ndarray]],
    intervention_type: str,
    positive_strength: float,
    race_group: int,
    input_device: torch.device,
    yes_ids: List[int],
    no_ids: List[int],
    num_heads: int,
    head_dim: int,
) -> float:
    """计算带干预的p(yes)概率。"""
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

        if intervention_type == "negative":
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

        elif intervention_type == "positive":
            target_module = adapter.get_head_activation_module(l)
            if adapter.head_activation_kind != "o_proj_input":
                raise NotImplementedError(
                    "Positive direction intervention currently requires an o_proj-input head surface."
                )
            # 正向干预：方向增强
            if head_directions is None or (l, h) not in head_directions:
                continue
            dir_np = head_directions[(l, h)]
            std_np = head_std.get((l, h)) if head_std else None
            if std_np is None:
                std_np = np.ones_like(dir_np, dtype=np.float32)
            direction = torch.from_numpy(dir_np * std_np).float()

            sign = 1.0 if race_group == 0 else -1.0

            hook_fn = make_positive_direction_hook(
                l,
                h,
                direction,
                output_pos,
                num_heads,
                head_dim,
                strength=positive_strength * sign,
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
                prefix=f"Intervention-{intervention_type}",
            )
            return float(p_yes)
    finally:
        remove_intervention_hooks(prompt_hooks)

    return 0.0


def compute_head_directions(
    model,
    adapter,
    tokenizer,
    samples: List[Dict],
    sensitive_heads: List[Tuple[int, int]],
    model_type: str,
    input_device: torch.device,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    cache_path: Optional[str] = None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """计算头方向向量（用于positive干预）。"""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading head directions from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cached_dirs = cached.get("head_directions", {})
        cached_std = cached.get("head_std", {})
        head_directions = {}
        head_std = {}
        for key in sensitive_heads:
            if key in cached_dirs:
                head_directions[key] = np.asarray(cached_dirs[key], dtype=np.float32)
                head_std[key] = np.asarray(cached_std.get(key, np.ones_like(head_directions[key])), dtype=np.float32)
        print(f"Loaded {len(head_directions)} head directions from cache.")
        return head_directions, head_std

    print("=" * 80)
    print("Computing head directions v_dir from factual vs. counterfactual activations...")
    print("=" * 80)

    # 构建 factual / counterfactual prompts
    fact_prompts_dir: List[str] = []
    cf_prompts_dir: List[str] = []
    races: List[int] = []
    for s in samples:
        fact_query = s["query"]
        fact_prompts_dir.append(format_prompt_for_model(fact_query, model_type))

        data_item = {
            "query": fact_query,
            "summary": s.get("summary", ""),
            "category": s.get("category", ""),
            "race": s.get("race_str", ""),
        }
        cf_item = create_counterfactual_by_race(data_item)
        cf_query = cf_item["query"]
        cf_prompts_dir.append(format_prompt_for_model(cf_query, model_type))
        races.append(s["race"])

    # 为每个 (layer, head) 初始化累计和
    dir_sum: Dict[Tuple[int, int], np.ndarray] = {
        (l, h): np.zeros(head_dim, dtype=np.float64) for (l, h) in sensitive_heads
    }
    dir_sq_sum: Dict[Tuple[int, int], np.ndarray] = {
        (l, h): np.zeros(head_dim, dtype=np.float64) for (l, h) in sensitive_heads
    }
    sample_count = 0

    # 预计算每层有哪些敏感头
    heads_by_layer: Dict[int, List[int]] = {}
    for l, h in sensitive_heads:
        heads_by_layer.setdefault(l, []).append(h)

    def _collect_last_token_activations(batch_prompts: List[str]) -> Dict[int, np.ndarray]:
        """返回该 batch 在所有层上的 [B, H, D] 激活（最后一个 token）。"""
        if not batch_prompts:
            return {}

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
        batch_range = torch.arange(input_ids.shape[0], device=input_device)

        batch_activations_buffer: Dict[int, torch.Tensor] = {}
        hooks = []
        for l in range(num_layers):
            hooks.append(
                adapter.register_head_activation_hook(
                    l, num_heads, head_dim, batch_activations_buffer
                )
            )

        with torch.no_grad():
            _ = model(**enc)

        for h_hook in hooks:
            h_hook.remove()

        last_token_indices_dev = last_token_indices.to(input_device)
        acts_last: Dict[int, np.ndarray] = {}

        for l, heads in heads_by_layer.items():
            if l not in batch_activations_buffer:
                continue
            act = batch_activations_buffer[l]  # [B, Seq, H, D]
            act_device = act.device
            br = batch_range.to(act_device)
            lti = last_token_indices_dev.to(act_device)
            last_act = act[br, lti, :, :]  # [B, H, D]
            acts_last[l] = last_act.detach().cpu().numpy()

        return acts_last

    # 按 batch 同时跑 factual & counterfactual
    for start in tqdm(
        range(0, len(samples), batch_size),
        desc="Collecting signed (fact-cf) activations for v_dir",
    ):
        end = min(start + batch_size, len(samples))
        if start >= end:
            continue

        batch_indices = list(range(start, end))
        batch_fact_prompts = [fact_prompts_dir[i] for i in batch_indices]
        batch_cf_prompts = [cf_prompts_dir[i] for i in batch_indices]
        batch_races = [races[i] for i in batch_indices]  # 0=White,1=Black

        # sign: White -> +1, Black -> -1
        signs = np.array([1.0 if r == 0 else -1.0 for r in batch_races], dtype=np.float32)

        fact_acts = _collect_last_token_activations(batch_fact_prompts)
        cf_acts = _collect_last_token_activations(batch_cf_prompts)

        if not fact_acts or not cf_acts:
            continue

        bsz = len(batch_indices)
        sample_count += bsz

        for l, heads in heads_by_layer.items():
            if l not in fact_acts or l not in cf_acts:
                continue
            fact_last = fact_acts[l]  # [B, H, D]
            cf_last = cf_acts[l]      # [B, H, D]
            if fact_last.shape != cf_last.shape:
                raise ValueError(f"Shape mismatch between fact and cf activations at layer {l}.")

            diff = fact_last - cf_last  # [B, H, D]
            sign_vec = signs.reshape(bsz, 1, 1)  # [B,1,1]
            contrib = sign_vec * diff           # [B, H, D]

            for h_idx in heads:
                key = (l, h_idx)
                if key not in sensitive_heads:
                    continue
                contrib_h = contrib[:, h_idx, :]  # [B, D]
                dir_sum[key] += contrib_h.sum(axis=0)
                dir_sq_sum[key] += (contrib_h ** 2).sum(axis=0)

    if sample_count == 0:
        raise ValueError("No activations collected for factual/counterfactual prompts.")

    # 计算 v_dir 和 std
    head_directions = {}
    head_std = {}
    for key in sensitive_heads:
        mean_vec = dir_sum[key] / float(sample_count)
        mean_sq = dir_sq_sum[key] / float(sample_count)
        var = np.maximum(mean_sq - mean_vec ** 2, 1e-12)
        std_vec = np.sqrt(var)
        head_directions[key] = mean_vec.astype(np.float32)
        head_std[key] = std_vec.astype(np.float32)

    # 写入缓存
    if cache_path:
        cache_payload = {
            "head_directions": head_directions,
            "head_std": head_std,
            "meta": {
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "sample_count": sample_count,
            },
        }
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache_payload, f)
        print(f"Saved head directions cache to: {cache_path}")

    return head_directions, head_std


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
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture for prompt formatting.",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--resume_prompt_mode",
        choices=["summary_only", "category"],
        default="summary_only",
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
        "--intervention_type",
        type=str,
        default="negative",
        choices=["negative", "positive"],
        help="Intervention type: 'negative' (mean ablation) or 'positive' (directional amplification).",
    )
    parser.add_argument(
        "--positive_strength",
        type=float,
        default=1.0,
        help="Strength alpha for positive intervention.",
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
        "--head_counts",
        type=int,
        nargs="+",
        default=None,
        help="Explicit head-count grid, for example: --head_counts 0 10 20 30 40 48.",
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
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

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

        if not summary or (args.resume_prompt_mode == "category" and not category):
            continue

        query = (
            summary
            if args.resume_prompt_mode == "summary_only"
            else build_category_prompt(summary, category)
        )

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
    print(f"Loading intervention data from {embeddings_path} (heatmap-based head selection)")
    print("=" * 80)

    results_data = load_intervention_results(embeddings_path)
    all_sensitive_heads = get_sensitive_heads_sorted_by_heatmap(results_data)
    print(
        f"Loaded {len(all_sensitive_heads)} elbow-selected sensitive heads "
        "(sorted by heatmap KL)"
    )

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
    print(f"Using {len(all_sensitive_heads)} heads with valid embeddings")

    if not all_sensitive_heads:
        raise ValueError("No valid sensitive heads found with embeddings.")

    # 计算头方向（如果需要positive干预）
    head_directions = None
    head_std = None
    if args.intervention_type == "positive":
        direction_cache_path = os.path.join(args.output_dir, "head_directions_fact_cf.pkl")
        head_directions, head_std = compute_head_directions(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            samples=samples,
            sensitive_heads=all_sensitive_heads,
            model_type=model_type,
            input_device=input_device,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=args.batch_size,
            cache_path=direction_cache_path,
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

    if args.head_counts is not None:
        head_counts = sorted(set(args.head_counts))
        if not head_counts or head_counts[0] != 0:
            raise ValueError("--head_counts must include 0 as the baseline.")
        if head_counts[-1] > len(all_sensitive_heads):
            raise ValueError(
                f"Requested {head_counts[-1]} sensitive heads, but the elbow set "
                f"contains only {len(all_sensitive_heads)}."
            )
    else:
        max_n = min(args.max_head_count, len(all_sensitive_heads))
        head_counts = list(range(0, max_n + 1, args.step))
        if max_n not in head_counts:
            head_counts.append(max_n)
        head_counts = sorted(set(head_counts))
    print(f"Will test head counts: {head_counts}")

    metadata = {
        "experiment": "resume_head_count_sensitive",
        "model_path": args.model_path,
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "dataset_json_path": args.dataset_json_path,
        "sample_csv_path": args.sample_csv_path,
        "sample_size": len(samples),
        "embeddings_path": embeddings_path,
        "intervention_type": args.intervention_type,
        "head_counts": head_counts,
        "available_elbow_head_count": len(all_sensitive_heads),
        "elbow_score": results_data.get("elbow_score"),
        "selected_heads_by_count": {
            str(count): [list(head) for head in all_sensitive_heads[:count]]
            for count in head_counts
        },
        "seed": args.seed,
        "resume_prompt_mode": args.resume_prompt_mode,
        "head_ranking_scope": "elbow_heads_sorted_by_importance",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 准备CSV输出
    csv_path = os.path.join(args.output_dir, "intervention_results_by_head_count.csv")
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

    # 对每个头数量进行测试
    for head_count in head_counts:
        print("=" * 80)
        print(f"Testing with {head_count} heads (intervention type: {args.intervention_type})")
        print("=" * 80)

        # 选择前head_count个头
        current_heads = all_sensitive_heads[:head_count] if head_count > 0 else []
        if head_count == 0:
            print("Using baseline (no intervention)")
        else:
            print(f"Using heads: {current_heads[:5]}..." if len(current_heads) > 5 else f"Using heads: {current_heads}")

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

                # 计算事实概率（带干预）
                fact_p_yes = compute_p_yes_with_intervention(
                    model=model,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt=fact_item["query"],
                    model_type=model_type,
                    sensitive_heads=current_heads,
                    white_emb=white_emb,
                    black_emb=black_emb,
                    head_directions=head_directions,
                    head_std=head_std,
                    intervention_type=args.intervention_type,
                    positive_strength=args.positive_strength,
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
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt=cf_item["query"],
                    model_type=model_type,
                    sensitive_heads=current_heads,
                    white_emb=white_emb,
                    black_emb=black_emb,
                    head_directions=head_directions,
                    head_std=head_std,
                    intervention_type=args.intervention_type,
                    positive_strength=args.positive_strength,
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
                    args.intervention_type,
                ])


    csv_file.close()
    print(f"\nSaved results to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
