#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
UNDER NEGATIVE HEAD-LEVEL INTERVENTION.

功能：
- 基于 exp2 的 race-sensitive heads 结果，对注意力头做负向干预（mean ablation）
- 在 discrim-eval 的配对样本上计算 p(yes)
- 按 decision_question_id 计算配对样本的概率差（绝对值），并打印统计信息
- 将每个样本在干预下的 p(yes) 结果保存到 CSV：
    sample_id, model, decision_question_id, p_yes, intervention_type
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
    make_intervention_hook_mean_replacement,
    remove_intervention_hooks,
)


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
            "dataset under NEGATIVE head-level intervention."
        )
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
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="prompt",
        choices=["prompt", "debiased_prompt"],
        help="Which prompt to use: 'prompt' (original) or 'debiased_prompt' (debiased).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_discrim_eval_results",
        help="Directory to save any intermediate or auxiliary results.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help=(
            "If set, append per-sample intervention results to this CSV file "
            "(sample_id, model, decision_question_id, p_yes, intervention_type)."
        ),
    )
    parser.add_argument(
        "--sensitive_heads_dir",
        type=str,
        default="",
        help=(
            "Directory containing selected_heads_elbow.json and results.pkl from "
            "exp2/analyze_race_sensitive_heads.py. If not provided, will try to "
            "infer from model_path."
        ),
    )
    parser.add_argument(
        "--sensitive_heads_path",
        type=str,
        default="",
        help="Path to selected_heads_elbow.json. Overrides --sensitive_heads_dir.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="",
        help="Path to results.pkl (containing white_emb/black_emb). Overrides --sensitive_heads_dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible head sampling and other stochastic operations.",
    )
    parser.add_argument(
        "--intervention_mode",
        type=str,
        default="negative",
        choices=["negative", "negative_random"],
        help=(
            "Intervention strategy: "
            "'negative' uses race-sensitive heads; "
            "'negative_random' randomly selects the same number of non-sensitive heads "
            "for mean ablation."
        ),
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility (Python, NumPy, PyTorch)
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

    # Determine execution device
    input_device = get_input_device(model, args.device)
    print(f"Inference device: {input_device}")

    # Resolve model type
    model_type = resolve_model_type(
        args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Using model_type: {model_type}")

    # 3. Get token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")

    # 4. Load sensitive heads & embeddings (from exp2)
    sensitive_heads_path = args.sensitive_heads_path
    embeddings_path = args.embeddings_path

    if not sensitive_heads_path or not embeddings_path:
        if args.sensitive_heads_dir:
            heads_dir = args.sensitive_heads_dir
        else:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            exp2_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "exp2_old",
            )
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

    if not sensitive_heads_path or not embeddings_path:
        missing = []
        if not sensitive_heads_path:
            missing.append("selected_heads_elbow.json")
        if not embeddings_path:
            missing.append("results.pkl")
        raise FileNotFoundError(
            "Intervention files not found. Missing: "
            + ", ".join(missing)
            + "\nPlease run analyze_race_sensitive_heads.py first, "
            "or specify --sensitive_heads_path and --embeddings_path."
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
    white_emb: Dict[Tuple[int, int], np.ndarray] = {
        (int(k[0]), int(k[1])): v
        for k, v in white_embeddings.items()
        if isinstance(k, (tuple, list))
    }
    black_emb: Dict[Tuple[int, int], np.ndarray] = {
        (int(k[0]), int(k[1])): v
        for k, v in black_embeddings.items()
        if isinstance(k, (tuple, list))
    }
    # 获取模型 head 配置
    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    print(f"Model config: layers={num_layers}, heads={num_heads}, head_dim={head_dim}")

    # 只保留在 embedding 字典中都存在的 sensitive heads
    valid_heads: List[Tuple[int, int]] = []
    for l, h in sensitive_heads:
        if (l, h) in white_emb and (l, h) in black_emb:
            valid_heads.append((l, h))
    sensitive_heads = valid_heads
    print(f"Using {len(sensitive_heads)} sensitive heads with valid embeddings")

    if not sensitive_heads:
        raise ValueError("No valid sensitive heads found with embeddings.")

    # 构建随机干预所需的 head 集合（从非敏感头中随机选取相同数量），显式确保与敏感头完全不相交
    all_heads_with_emb = [k for k in white_emb.keys() if k in black_emb]
    sensitive_head_set = set(sensitive_heads)
    non_sensitive_heads = [k for k in all_heads_with_emb if k not in sensitive_head_set]

    if args.intervention_mode == "negative_random":
        if len(non_sensitive_heads) < len(sensitive_heads):
            raise ValueError(
                "Not enough non-sensitive heads with embeddings to perform "
                "negative_random intervention."
            )
        random_heads = random.sample(non_sensitive_heads, len(sensitive_heads))
        # 再做一次安全检查，确保随机采样到的头与敏感头集合完全不相交
        random_head_set = set(random_heads)
        if random_head_set & sensitive_head_set:
            raise RuntimeError(
                "Random head selection unexpectedly overlaps with sensitive heads. "
                "Please check head lists and sampling logic."
            )
        heads_for_intervention: List[Tuple[int, int]] = list(random_head_set)
        print(
            "Intervention mode: negative_random (mean ablation on randomly selected "
            f"{len(heads_for_intervention)} non-sensitive heads)"
        )
    else:
        heads_for_intervention = sensitive_heads
        print(
            "Intervention mode: negative (mean ablation on race-sensitive heads)"
        )

    # 5. Prepare prompts
    prompt_key = args.prompt_type
    prompts: List[str] = [
        add_yes_no_instruction(build_category_prompt(item[prompt_key],"")) for item in data
        # add_yes_no_instruction(item[prompt_key]) for item in data

    ]

    # 6. Compute p(yes) with intervention
    print("=" * 80)
    print(
        f"Computing p(yes) with {args.intervention_mode} intervention "
        f"for {args.prompt_type}"
    )
    print("=" * 80)

    p_yes_results: List[float] = []

    for idx, (sample, prompt) in enumerate(
        tqdm(
            list(zip(data, prompts)),
            desc=f"Computing p(yes) [{args.intervention_mode} intervention]",
        )
    ):
        formatted_prompt = format_prompt_for_model(prompt, model_type)
        input_ids = tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(input_device)
        attention_mask = torch.ones_like(input_ids).to(input_device)

        last_token_indices = get_last_token_indices_safe(
            input_ids, attention_mask, tokenizer
        )
        output_pos = int(last_token_indices[0].item())

        prompt_hooks = []
        for l, h in heads_for_intervention:
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
                l,
                h,
                mean_emb,
                output_pos,
                num_heads,
                head_dim,
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
                    prefix=f"Intervention-{args.intervention_mode}",
                )
                p_yes_results.append(float(p_yes))
        finally:
            remove_intervention_hooks(prompt_hooks)

        del input_ids, outputs, logits_row

    # 7. Map results by sample_id and compute stats
    p_yes_map: Dict[int, float] = {
        int(sample["id"]): p_yes for sample, p_yes in zip(data, p_yes_results)
    }

    # 构建 sample_id -> matched_id 映射，便于后续写 CSV 与对齐 baseline 统计方式
    matched_map: Dict[int, int] = {}
    for a, b in pairs:
        # 如果同一个样本在多个配对中出现，只保留第一次出现的匹配关系
        if a not in matched_map:
            matched_map[a] = b
        if b not in matched_map:
            matched_map[b] = a

    print("Aggregating statistics (absolute p(yes) differences per question)...")
    stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)

    # Sort keys by mean bias (descending)
    ordered_qids = sorted(
        stats.keys(),
        key=lambda q: stats[q]["mean"],
        reverse=True,
    )

    print(
        f"\n--- Top 10 Most Biased Questions "
        f"({args.intervention_mode} intervention, {args.prompt_type}) ---"
    )
    for i, qid in enumerate(ordered_qids[:10]):
        s = stats[qid]
        print(
            f"QID {qid}: Mean Gap={s['mean']:.4f}, Std={s['std']:.4f}, Count={s['count']}"
        )

    # 8. Get model name
    model_name = os.path.basename(os.path.normpath(args.model_path))
    intervention_type = args.intervention_mode

    # 9. Save per-sample CSV (if path provided)
    if args.csv_path:
        file_exists = os.path.exists(args.csv_path)
        print(f"Appending per-sample intervention details to CSV: {args.csv_path}...")
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "sample_id",
                        "matched_id",
                        "model",
                        "decision_question_id",
                        "p_yes",
                        "intervention_type",
                    ]
                )

            for sample, p_yes in zip(data, p_yes_results):
                # Skip if p_yes is NaN or None
                if p_yes is None or math.isnan(p_yes):
                    continue

                sample_id = int(sample["id"])
                decision_question_id = sample.get("decision_question_id")
                if decision_question_id is not None:
                    decision_question_id = int(decision_question_id)
                else:
                    decision_question_id = ""

                matched_id = matched_map.get(sample_id)
                matched_id_out = int(matched_id) if matched_id is not None else ""

                writer.writerow(
                    [
                        sample_id,
                        matched_id_out,
                        model_name,
                        decision_question_id if decision_question_id != "" else "",
                        float(p_yes),
                        intervention_type,
                    ]
                )
        print(f"Saved CSV: {args.csv_path}")

    # 10. Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({args.intervention_mode} intervention)")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Total pairs: {len(pairs)}")
    valid_samples = sum(
        1 for p_yes in p_yes_results if p_yes is not None and not math.isnan(p_yes)
    )
    print(f"Valid samples with p(yes) values: {valid_samples}")
    print(f"Questions analyzed: {len(stats)}")

    if len(stats) > 0:
        all_diffs = [s["mean"] for s in stats.values()]
        print("\nOverall statistics:")
        print(f"  - Mean gap across all questions: {np.mean(all_diffs):.6f}")
        print(f"  - Std gap across all questions: {np.std(all_diffs):.6f}")
        print(f"  - Max gap: {np.max(all_diffs):.6f}")
        print(f"  - Min gap: {np.min(all_diffs):.6f}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

