#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences across race groups for Resume dataset
with **head-level interventions** (negative vs positive) based on
`exp2/analyze_race_sensitive_heads.py` results.

Negative intervention:
    - Mean ablation on race-sensitive heads.
    - For each selected head, replace its activation at the output position
      with the average activation across groups (approximation of mean over
      counterfactual data X_c).

Positive intervention:
    - Directional amplification on race-sensitive heads.
    - For each selected head, compute activation direction on sensitive
      attributes
          v_dir = E[a | org] - E[a | countf]
      where we approximate org = Black, countf = White using the stored
      `black_emb` and `white_emb` in results.pkl.
    - At the output position, shift the head activation by
          a' = a + alpha * v_dir * sign o std
      to amplify the component's function along the sensitive direction.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type
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
)
from sampling import sample_resume_data_by_race, load_samples_by_csv_indices
from hook import (
    get_last_token_indices_safe,
    make_intervention_hook_mean_replacement,
    remove_intervention_hooks,
    make_positive_direction_hook,
    get_activation_hook_for_intervention,
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
        description=(
            "Evaluate p(yes) probability differences across race groups for Resume "
            "dataset with head-level negative/positive interventions."
        )
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
        default="intervention_heads_results",
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
        help="Path to results.pkl file (containing white_emb/black_emb). Overrides --sensitive_heads_dir if provided.",
    )
    parser.add_argument(
        "--intervention_type",
        type=str,
        default="negative",
        choices=["negative", "positive", "baseline"],
        help="Intervention type: 'negative' (mean ablation), 'positive' (directional amplification), or 'baseline'.",
    )
    parser.add_argument(
        "--positive_strength",
        type=float,
        default=1.0,
        help="Strength alpha for positive (directional) intervention.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="If set, append aggregate results to this CSV file (for multi-model aggregation).",
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

    # 采样数据：同 evaluate_intervention.py
    csv_fact_p_yes: Optional[List[Optional[float]]] = None
    baseline_p_yes_for_samples: List[float] = []
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

        query = add_yes_no_instruction(summary)

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
                "query": query,
                "summary": summary,
                "category": category,
            }
        )
        # 若 CSV 中提供了 fact_p_yes，则为 baseline 记录下对应值（与 samples 顺序对齐）
        if csv_fact_p_yes is not None and idx < len(csv_fact_p_yes):
            val = csv_fact_p_yes[idx]
            if val is not None:
                baseline_p_yes_for_samples.append(float(val))

    if not samples:
        raise ValueError("No valid samples found (need race in {White, Black}).")

    prompts = [s["query"] for s in samples]
    races = [s["race"] for s in samples]

    # 判断是否使用干预
    use_intervention = args.intervention_type in ("negative", "positive")

    # 确定 sensitive_heads_path 和 embeddings_path
    sensitive_heads_path = args.sensitive_heads_path
    embeddings_path = args.embeddings_path

    if not sensitive_heads_path or not embeddings_path:
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            exp2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp2")
            exp2_dir = os.path.abspath(exp2_dir)
            heads_dir = os.path.join(exp2_dir, f"sensitive_heads_{model_name}_top100")

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

    sensitive_heads: List[Tuple[int, int]] = []
    white_emb: Dict[Tuple[int, int], np.ndarray] = {}
    black_emb: Dict[Tuple[int, int], np.ndarray] = {}

    if use_intervention:
        if not sensitive_heads_path or not embeddings_path:
            missing = []
            if not sensitive_heads_path:
                missing.append("selected_heads_elbow.json")
            if not embeddings_path:
                missing.append("results.pkl")
            raise FileNotFoundError(
                f"Intervention files not found. Missing: {', '.join(missing)}\n"
                f"Please run analyze_race_sensitive_heads.py first, or specify --sensitive_heads_path and --embeddings_path."
            )

        print("=" * 80)
        print(f"Loading intervention data from {sensitive_heads_path} and {embeddings_path}")
        print("=" * 80)

        with open(sensitive_heads_path, "r", encoding="utf-8") as f:
            selected_heads_data = json.load(f)
        sensitive_heads = [(h["layer"], h["head"]) for h in selected_heads_data]
        print(f"Loaded {len(sensitive_heads)} sensitive heads")

        with open(embeddings_path, "rb") as f:
            embeddings_data = pickle.load(f)

        white_embeddings = embeddings_data.get("white_emb", {})
        black_embeddings = embeddings_data.get("black_emb", {})

        # 转为 tuple key，确保可索引
        white_emb = {
            (int(k[0]), int(k[1])): v for k, v in white_embeddings.items() if isinstance(k, (tuple, list))
        }
        black_emb = {
            (int(k[0]), int(k[1])): v for k, v in black_embeddings.items() if isinstance(k, (tuple, list))
        }

        # 获取模型 head 配置
        config = get_model_config(model)
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        
        print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")
        print(f"Intervention type: {args.intervention_type}")

        # 只保留在 embedding 字典中都存在的 sensitive heads
        valid_heads: List[Tuple[int, int]] = []
        for l, h in sensitive_heads:
            if (l, h) in white_emb and (l, h) in black_emb:
                valid_heads.append((l, h))
        sensitive_heads = valid_heads
        print(f"Using {len(sensitive_heads)} heads with valid embeddings")

        if not sensitive_heads:
            raise ValueError("No valid sensitive heads found with embeddings.")

        # ==================================================================
        # Step 1 (for positive intervention): compute v_dir and per-dim std
        # from factual vs. counterfactual activations across all sampled
        # prompts, with cache.
        #
        # 方向与尺度定义（对每个样本 i，每个敏感头 (l, h)）：
        #   diff_i = a_fact_i - a_cf_i
        #   sign_i = +1 (White) / -1 (Black)
        #   contrib_i = sign_i * diff_i
        #
        #   v_dir = E_i[contrib_i]                  # 方向
        #   std  = Std_i[contrib_i] (逐维标准差)     # 每维尺度
        #
        # 干预时：
        #   a' = a + alpha * sign * (v_dir * std)
        # ==================================================================
        head_directions: Dict[Tuple[int, int], np.ndarray] = {}
        head_std: Dict[Tuple[int, int], np.ndarray] = {}
        if args.intervention_type == "positive":
            direction_cache_path = os.path.join(args.output_dir, "head_directions_fact_cf.pkl")

            if os.path.exists(direction_cache_path):
                print(f"Loading head directions from cache: {direction_cache_path}")
                with open(direction_cache_path, "rb") as f:
                    cached = pickle.load(f)
                cached_dirs = cached.get("head_directions", {})
                cached_std = cached.get("head_std", {})
                # 只保留当前敏感头
                for key, vec in cached_dirs.items():
                    if key in sensitive_heads:
                        head_directions[key] = np.asarray(vec, dtype=np.float32)
                        if key in cached_std:
                            head_std[key] = np.asarray(cached_std[key], dtype=np.float32)
                        else:
                            head_std[key] = np.ones_like(head_directions[key], dtype=np.float32)
                print(f"Loaded {len(head_directions)} head directions (with std) from cache.")
            else:
                print("=" * 80)
                print("Computing head directions v_dir from factual vs. counterfactual activations...")
                print("=" * 80)

                # 构建 factual / counterfactual prompts
                fact_prompts_dir: List[str] = []
                cf_prompts_dir: List[str] = []
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
                if len(fact_prompts_dir) != len(cf_prompts_dir):
                    raise ValueError("Mismatch between factual and counterfactual prompts.")

                # 为每个 (layer, head) 初始化累计和（带符号的贡献）与平方和
                dir_sum: Dict[Tuple[int, int], np.ndarray] = {
                    (l, h): np.zeros(head_dim, dtype=np.float64) for (l, h) in sensitive_heads
                }
                dir_sq_sum: Dict[Tuple[int, int], np.ndarray] = {
                    (l, h): np.zeros(head_dim, dtype=np.float64) for (l, h) in sensitive_heads
                }
                sample_count = 0

                # 预计算每层有哪些敏感头，便于加速
                heads_by_layer: Dict[int, List[int]] = {}
                for l, h in sensitive_heads:
                    heads_by_layer.setdefault(l, []).append(h)
                batch_size = max(1, int(args.batch_size))

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
                        layer_module = model.model.layers[l].self_attn.o_proj
                        hook_fn = get_activation_hook_for_intervention(
                            l, num_heads, head_dim, batch_activations_buffer
                        )
                        hooks.append(layer_module.register_forward_hook(hook_fn))

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

                # 按 batch 同时跑 factual & counterfactual，并根据 race/sign 计算贡献
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
                    signs = np.array([1.0 if r == 0 else -1.0 for r in batch_races], dtype=np.float32)  # [B]

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
                        # 贡献 = sign * (fact - cf)
                        sign_vec = signs.reshape(bsz, 1, 1)  # [B,1,1]
                        contrib = sign_vec * diff           # [B, H, D]

                        for h_idx in heads:
                            key = (l, h_idx)
                            contrib_h = contrib[:, h_idx, :]  # [B, D]
                            dir_sum[key] += contrib_h.sum(axis=0)
                            dir_sq_sum[key] += (contrib_h ** 2).sum(axis=0)

                if sample_count == 0:
                    raise ValueError("No activations collected for factual/counterfactual prompts.")

                # 计算 v_dir = mean_over_samples( 贡献 )
                # 以及逐维 std = sqrt( E[x^2] - (E[x])^2 )
                for key in sensitive_heads:
                    mean_vec = dir_sum[key] / float(sample_count)
                    mean_sq = dir_sq_sum[key] / float(sample_count)
                    var = np.maximum(mean_sq - mean_vec ** 2, 1e-12)
                    std_vec = np.sqrt(var)
                    head_directions[key] = mean_vec.astype(np.float32)
                    head_std[key] = std_vec.astype(np.float32)

                # 写入缓存
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
                os.makedirs(args.output_dir, exist_ok=True)
                with open(direction_cache_path, "wb") as f:
                    pickle.dump(cache_payload, f)
                print(f"Saved head directions cache to: {direction_cache_path}")
    else:
        print("=" * 80)
        print("Evaluating baseline p(yes) probabilities (no intervention)")
        print("=" * 80)

    # 计算 p(yes)（有或无干预）
    if use_intervention:
        print(f"Computing p(yes) with intervention type: {args.intervention_type}")
        intervention_results: List[float] = []

        for idx, prompt in enumerate(tqdm(prompts, desc=f"Computing p(yes) [{args.intervention_type}]")):
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

                if args.intervention_type == "negative":
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

                elif args.intervention_type == "positive":
                    # 正向干预：方向增强，a' = a + alpha * s * v_dir
                    # 其中 s = +1 (White) 或 -1 (Black)
                    if (l, h) not in head_directions:
                        continue
                    dir_np = head_directions[(l, h)]
                    std_np = head_std.get((l, h))
                    if std_np is None:
                        std_np = np.ones_like(dir_np, dtype=np.float32)
                    direction = torch.from_numpy(dir_np * std_np).float()

                    race_group = races[idx]  # 0=White,1=Black
                    sign = 1.0 if race_group == 0 else -1.0

                    hook_fn = make_positive_direction_hook(
                        l,
                        h,
                        direction,
                        output_pos,
                        num_heads,
                        head_dim,
                        strength= args.positive_strength * sign,
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
                        sample_idx=idx,
                        show_warnings=True,
                        prefix=f"Intervention-{args.intervention_type}",
                    )
                    intervention_results.append(float(p_yes))
            finally:
                remove_intervention_hooks(prompt_hooks)

            del input_ids, outputs, logits_row

        results_list = intervention_results
        suffix = f"_{args.intervention_type}"
    else:
        # baseline：优先从 CSV 中直接读取 fact_p_yes（如果可用），否则回退到重新计算
        use_csv_baseline = (
            args.sample_csv_path
            and baseline_p_yes_for_samples
            and len(baseline_p_yes_for_samples) == len(samples)
        )
        if use_csv_baseline:
            print("Using fact_p_yes from CSV as baseline (no forward pass).")
            results_list = baseline_p_yes_for_samples
            suffix = "_baseline_csv"
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
            suffix = "_baseline"

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
            "intervention_type": args.intervention_type,
            "positive_strength": args.positive_strength if args.intervention_type == "positive" else None,
            "num_sensitive_heads": len(sensitive_heads) if use_intervention else 0,
        },
        "results": {
            **summary,
        },
    }

    results_path = os.path.join(args.output_dir, f"results{suffix}.json")
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
    per_sample_path = os.path.join(args.output_dir, f"per_sample_results{suffix}.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    print(f"Saved: {per_sample_path}")

    if args.csv_path:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        file_exists = os.path.exists(args.csv_path)
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)

        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "model",
                        "intervention_type",
                        "positive_strength",
                        "p_yes_mean",
                        "p_yes_white_mean",
                        "p_yes_black_mean",
                        "fairness_gap",
                        "white_n",
                        "black_n",
                        "total_samples",
                    ]
                )

            writer.writerow(
                [
                    model_name,
                    args.intervention_type if use_intervention else "baseline",
                    args.positive_strength if args.intervention_type == "positive" else 0.0,
                    summary["p_yes_mean"],
                    summary["p_yes_white_mean"],
                    summary["p_yes_black_mean"],
                    summary["fairness_gap_black_minus_white"],
                    summary["white_n"],
                    summary["black_n"],
                    summary["n"],
                ]
            )
        print(f"Appended to CSV: {args.csv_path}")

    print("\n" + "=" * 60)
    if use_intervention:
        print(f"P(YES) PROBABILITY DIFFERENCES BY RACE (Intervention type: {args.intervention_type})")
    else:
        print("BASELINE P(YES) PROBABILITY DIFFERENCES BY RACE")
    print("=" * 60)
    print("Overall:")
    print(f"  - Mean p(yes): {summary['p_yes_mean']:.6f} (n={summary['n']})")
    print("\nBy Race Group:")
    print(f"  - White mean p(yes): {summary['p_yes_white_mean']:.6f} (n={summary['white_n']})")
    print(f"  - Black mean p(yes): {summary['p_yes_black_mean']:.6f} (n={summary['black_n']})")
    print("\nFairness Gap:")
    print(f"  - Black - White: {summary['fairness_gap_black_minus_white']:.6f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

