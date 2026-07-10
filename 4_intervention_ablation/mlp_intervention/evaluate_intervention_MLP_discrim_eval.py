#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probability differences for paired discrim-eval dataset
UNDER MLP-LEVEL NEGATIVE INTERVENTION (mean ablation).

功能：
- 基于 exp2 的 race-sensitive MLP 结果，对选中的 MLP 层做负向干预（统一 mean ablation）
- 在 discrim-eval 的配对样本上计算 p(yes)
- 按 decision_question_id 计算配对样本的概率差（绝对值），并打印统计信息
- 将每个样本在干预下的 p(yes) 结果保存到 CSV：
    sample_id, matched_id, model, decision_question_id, p_yes, intervention_type

与 exp8_old/evaluate_intervention_discrim-eval.py 类似，但干预对象从 head 改为 MLP 层。
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

from probability import (  # type: ignore  # noqa: E402
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
)
from prompt import (  # type: ignore  # noqa: E402
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
)
from sampling import load_discrim_eval_pairs  # type: ignore  # noqa: E402
from util import (  # type: ignore  # noqa: E402
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
)
from hook import (  # type: ignore  # noqa: E402
    get_last_token_indices_safe,
    remove_intervention_hooks,
)
from model_adapter import get_model_adapter  # type: ignore  # noqa: E402


def compute_stats_by_question(
    pairs: List[Tuple[int, int]],
    id_to_sample: Dict[int, dict],
    p_yes_map: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    """Aggregate absolute p_yes differences by decision_question_id.
    与 exp8_old/evaluate_intervention_discrim-eval.py 保持一致。
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate p(yes) probability differences for paired discrim-eval "
            "dataset under MLP-level negative intervention (mean ablation)."
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
        "--prompt_type",
        type=str,
        default="prompt",
        choices=["prompt", "debiased_prompt"],
        help="Which prompt to use: 'prompt' (original) or 'debiased_prompt' (debiased).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_mlp_discrim_eval_results",
        help="Directory to save any intermediate or auxiliary results.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help=(
            "If set, append per-sample intervention results to this CSV file "
            "(sample_id, matched_id, model, decision_question_id, p_yes, intervention_type)."
        ),
    )
    parser.add_argument(
        "--append_csv",
        action="store_true",
        help="Append to --csv_path instead of replacing it.",
    )
    parser.add_argument(
        "--sensitive_mlp_path",
        type=str,
        required=True,
        help="Path to selected_mlp_layers_elbow.json from exp15/select_race_sensitive_MLPs.py.",
    )
    parser.add_argument(
        "--mlp_embeddings_path",
        type=str,
        required=True,
        help="Path to MLP mean embeddings pkl (containing white_emb/black_emb for discrim-eval).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--intervention_type",
        type=str,
        default="mlp_negative",
        help="Label for this intervention type (saved into CSV).",
    )
    args = parser.parse_args()

    # Set random seeds
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

    # 3. Get token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")

    # 4. Load sensitive MLP layers & embeddings
    if not os.path.exists(args.sensitive_mlp_path):
        raise FileNotFoundError(f"sensitive_mlp_path not found: {args.sensitive_mlp_path}")
    if not os.path.exists(args.mlp_embeddings_path):
        raise FileNotFoundError(f"mlp_embeddings_path not found: {args.mlp_embeddings_path}")

    print("=" * 80)
    print(
        f"Loading MLP intervention data from {args.sensitive_mlp_path} and {args.mlp_embeddings_path}"
    )
    print("=" * 80)

    with open(args.sensitive_mlp_path, "r", encoding="utf-8") as f:
        selected_layers_data = json.load(f)
    sensitive_layers = [int(d["layer"]) for d in selected_layers_data]
    print(f"Loaded {len(sensitive_layers)} sensitive MLP layers (before filtering)")

    with open(args.mlp_embeddings_path, "rb") as f:
        emb_data = pickle.load(f)

    white_embeddings = emb_data.get("white_emb", {})
    black_embeddings = emb_data.get("black_emb", {})

    # 转为 int key
    white_emb: Dict[int, np.ndarray] = {int(k): v for k, v in white_embeddings.items()}
    black_emb: Dict[int, np.ndarray] = {int(k): v for k, v in black_embeddings.items()}

    # 获取模型层数和 hidden_size 主要用于 sanity check
    config = get_model_config(model)
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    print(f"Model config: layers={num_layers}, hidden_size={hidden_size}")

    # 只保留同时在 white/black embedding 中存在的层
    valid_layers: List[int] = []
    for l in sensitive_layers:
        if l in white_emb and l in black_emb:
            valid_layers.append(l)
    sensitive_layers = valid_layers
    print(f"Using {len(sensitive_layers)} sensitive MLP layers with valid embeddings")

    if not sensitive_layers:
        raise ValueError("No valid sensitive MLP layers found with embeddings.")

    # 5. Prepare prompts
    prompt_key = args.prompt_type
    prompts: List[str] = [
        add_yes_no_instruction(item[prompt_key]) for item in data
    ]

    # 6. Compute p(yes) with MLP mean ablation intervention
    print("=" * 80)
    print(
        f"Computing p(yes) with MLP mean ablation intervention for {args.prompt_type}"
    )
    print("=" * 80)

    p_yes_results: List[float] = []

    for idx, (sample, prompt) in enumerate(
        tqdm(list(zip(data, prompts)), desc="Computing p(yes) [MLP negative intervention]")
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
        for l in sensitive_layers:
            if l < 0 or l >= num_layers:
                continue
            if l not in white_emb or l not in black_emb:
                continue

            mean_emb_np = (white_emb[l] + black_emb[l]) / 2.0
            mean_emb = (
                torch.from_numpy(mean_emb_np).float()
                if isinstance(mean_emb_np, np.ndarray)
                else mean_emb_np
            )

            hook = adapter.register_mlp_mean_replacement_hook(
                layer_idx=l,
                mean_embedding=mean_emb,
                output_pos=output_pos,
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
                    sample_idx=idx,
                    show_warnings=True,
                    prefix="Intervention-MLP-Negative",
                )
                p_yes_results.append(float(p_yes))
        finally:
            remove_intervention_hooks(prompt_hooks)

        del input_ids, outputs, logits_row

    # 7. Map results by sample_id and compute stats
    p_yes_map: Dict[int, float] = {
        int(sample["id"]): p_yes for sample, p_yes in zip(data, p_yes_results)
    }

    matched_map: Dict[int, int] = {}
    for a, b in pairs:
        if a not in matched_map:
            matched_map[a] = b
        if b not in matched_map:
            matched_map[b] = a

    print("Aggregating statistics (absolute p(yes) differences per question)...")
    stats = compute_stats_by_question(pairs, id_to_sample, p_yes_map)

    ordered_qids = sorted(
        stats.keys(),
        key=lambda q: stats[q]["mean"],
        reverse=True,
    )

    print(
        f"\n--- Top 10 Most Biased Questions (MLP negative intervention, {args.prompt_type}) ---"
    )
    for i, qid in enumerate(ordered_qids[:10]):
        s = stats[qid]
        print(
            f"QID {qid}: Mean Gap={s['mean']:.4f}, Std={s['std']:.4f}, Count={s['count']}"
        )

    # 8. Get model name
    model_name = os.path.basename(os.path.normpath(args.model_path))
    intervention_type = args.intervention_type

    # 9. Save per-sample CSV (if path provided)
    if args.csv_path:
        print(f"Writing per-sample intervention details to CSV: {args.csv_path}...")
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        mode = "a" if args.append_csv else "w"
        write_header = not args.append_csv or not os.path.exists(args.csv_path)
        with open(args.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
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

    metadata = {
        "model_path": args.model_path,
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "mlp_surface": "routed_moe_block_output",
        "dataset": "discrim_eval_transfer",
        "dataset_path": args.dataset_path,
        "prompt_type": args.prompt_type,
        "num_samples": len(data),
        "num_pairs": len(pairs),
        "sensitive_mlp_path": args.sensitive_mlp_path,
        "mlp_embeddings_path": args.mlp_embeddings_path,
        "selected_layers": sensitive_layers,
        "intervention_type": args.intervention_type,
        "seed": args.seed,
        "csv_path": args.csv_path or None,
        "stats_by_question": {str(k): v for k, v in stats.items()},
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 10. Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY (MLP negative intervention)")
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
