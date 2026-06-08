#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 Resume 数据集中种族敏感的 MLP 层（Race-Sensitive MLP Layers）

与 `analyze_race_sensitive_heads.py` 类似，但干预目标从注意力头（self_attn.o_proj）
改为每层的 MLP（model.model.layers[l].mlp），计算每层 MLP 的种族敏感度（Total Effect）。
"""

import json
import os
import pickle
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# 导入上层目录的工具函数
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from util import (  # type: ignore  # noqa: E402
    extract_race_from_query,
    create_counterfactual_by_race,
)
from sampling import sample_resume_data_by_race  # type: ignore  # noqa: E402
from prompt import (  # type: ignore  # noqa: E402
    build_category_prompt,
    format_prompt_for_model,
    resolve_model_type,
    add_yes_no_instruction,
)
from probability import (  # type: ignore  # noqa: E402
    get_target_token_ids,
    YES_CANDIDATES,
    NO_CANDIDATES,
    log_top3_warning,
)
from hook import (  # type: ignore  # noqa: E402
    get_last_token_indices_safe,
    get_mlp_last_token_activation_hook,
    get_mlp_last_token_patch_hook,
)
from util import get_model_config  # type: ignore  # noqa: E402
from cache import MLPDiskCache  # type: ignore  # noqa: E402
from plot import plot_kl_heatmap  # type: ignore  # noqa: E402


class InterventionDataset(Dataset):
    """
    Resume 数据集的干预 Dataset：
    - fact_data：原始样本（包含 query / race 等）
    - cf_data：翻转种族后的反事实样本
    """

    def __init__(self, fact_data, counterfact_data):
        self.fact_data = fact_data
        self.cf_data = counterfact_data

    def __len__(self):
        return len(self.fact_data)

    def __getitem__(self, idx):
        fact = self.fact_data[idx]
        cf = self.cf_data[idx]
        race = extract_race_from_query(fact["query"])

        return {
            "index": idx,
            "fact_prompt": add_yes_no_instruction(fact["query"]),
            "cf_prompt": add_yes_no_instruction(cf["query"]),
            "race": race if race else "Unknown",
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="race_sensitive_mlp_analysis_resume",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture for prompt formatting. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        default=False,
        help="Use random sampling instead of sequential sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (only used when --random_sampling is enabled).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling (balance by race). (Default: True)",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling (use --no-balanced to disable).",
    )
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        default="",
        help="If set, sample by following the order of the CSV's `index` column (top rows first). "
        "This overrides --max_samples/--balanced/--random_sampling.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="When --sample_csv_path is set, take the first N rows' indices from the CSV. "
        "If <= 0, use all indices in the CSV.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="llm_intervention_race_resume_mlp_")
    print(f"Temporary cache: {temp_dir}")

    try:
        # ==========================================================
        # 1. 加载模型
        # ==========================================================
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()

        # 获取模型配置
        config = get_model_config(model)
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]

        print(f"Layers: {num_layers}, Hidden Size: {hidden_size}")

        # 解析模型类型
        model_type = resolve_model_type(
            args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path
        )
        print(f"Using model_type: {model_type}")

        device = torch.device(args.device)

        # 准备 yes/no token id
        yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
        no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
        print(f"Yes token IDs: {yes_ids}")
        print(f"No token IDs: {no_ids}")

        # ==========================================================
        # 2. 加载与采样 Resume 数据
        # ==========================================================
        print(f"Loading dataset from {args.dataset_json_path}...")
        with open(args.dataset_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Dataset should be a list of records.")

        # 采样数据：
        # 1) 若提供 --sample_csv_path，则按 CSV 的 index 顺序取前 sample_size 个样本（覆盖其它采样参数）
        # 2) 否则使用原有的按种族平衡/随机采样逻辑
        if args.sample_csv_path:
            from analyze_race_sensitive_heads import _load_samples_by_csv_indices  # type: ignore

            print(f"Sampling by CSV order: {args.sample_csv_path}")
            sampled_data = _load_samples_by_csv_indices(
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

        white_count = sum(1 for item in sampled_data if item.get("race", "").lower() == "white")
        black_count = sum(1 for item in sampled_data if item.get("race", "").lower() == "black")
        print(f"Sampled {len(sampled_data)} samples")
        if (not args.sample_csv_path) and args.balanced:
            print(f"  - White: {white_count}, Black: {black_count}")
        if args.sample_csv_path:
            print(f"  - White: {white_count}, Black: {black_count} (CSV-driven sampling)")

        # 准备 fact 数据：从 summary 构建 query
        fact_data = []
        for item in sampled_data:
            summary = item.get("summary", "")
            category = item.get("category", "")
            race = item.get("race", "")

            query = summary
            fact_item = {
                "query": query,
                "summary": summary,
                "category": category,
                "race": race,
                "ID": item.get("ID", 0),
            }
            fact_data.append(fact_item)

        # 创建反事实数据（翻转种族）
        print("Creating counterfactual data (flipping race)...")
        cf_data = []
        for fact_item in fact_data:
            cf_item = create_counterfactual_by_race(fact_item)
            cf_data.append(cf_item)

        min_len = min(len(fact_data), len(cf_data))
        fact_data = fact_data[:min_len]
        cf_data = cf_data[:min_len]

        # 种族列表（与样本顺序一一对应）
        races_list = []
        for item in fact_data:
            r = extract_race_from_query(item["query"])
            races_list.append(r if r else "Unknown")

        # 初始化缓存：只需要 MLP 的 CF 激活缓存
        cf_mlp_cache = MLPDiskCache(min_len, num_layers, hidden_size, "cf_mlp", temp_dir)

        # 存储 Fact 的最终概率分布
        fact_probs_list: List[torch.Tensor] = []

        dataset_obj = InterventionDataset(fact_data, cf_data)
        dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

        # ==========================================================
        # Step 1: Data Preparation
        #   1) 跑 CF，缓存每层在 last token 的 MLP 输出 (作为干预源)
        #   2) 跑 Fact，缓存最后 token 的概率分布 (作为 KL 基准)
        # ==========================================================
        print("=" * 80)
        print("Step 1: Preparing Data (Fact Logits, CF MLP Activations)...")
        print("=" * 80)

        batch_mlp_buffer: Dict[int, torch.Tensor] = {}

        # 1) 先跑 CF，收集所有层的 MLP 激活
        hooks = []
        for l in range(num_layers):
            if hasattr(model.model.layers[l], "mlp"):
                layer_module = model.model.layers[l].mlp
            else:
                raise ValueError("Cannot find mlp module in transformer layer")
            hook_fn = get_mlp_last_token_activation_hook(l, batch_mlp_buffer)
            hooks.append(layer_module.register_forward_hook(hook_fn))

        # DEBUG: 输出第一个样本用于调试
        first_batch_printed = False
        for batch in tqdm(dataloader, desc="Collecting CF MLP Activations"):
            indices = batch["index"]
            
            if not first_batch_printed:
                print("=" * 80)
                print("DEBUG: First batch to be processed by model (CF activations):")
                for i, idx in enumerate(indices):
                    print(f"  Sample {i} (index {idx.item()}):")
                    print(f"    CF prompt: {batch['cf_prompt'][i]}")
                    print(f"    Race: {batch['race'][i]}")
                print("=" * 80)
                first_batch_printed = True

            cf_prompts_formatted = [
                format_prompt_for_model(prompt, model_type) for prompt in batch["cf_prompt"]
            ]
            cf_inputs = tokenizer(
                cf_prompts_formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(device)

            input_ids = cf_inputs["input_ids"]
            attention_mask = cf_inputs.get("attention_mask", torch.ones_like(input_ids))
            last_token_indices = get_last_token_indices_safe(
                input_ids, attention_mask, tokenizer
            )
            batch_range = torch.arange(input_ids.shape[0], device=device)

            batch_mlp_buffer.clear()
            with torch.no_grad():
                _ = model(**cf_inputs)

            for l in range(num_layers):
                if l in batch_mlp_buffer:
                    act = batch_mlp_buffer[l]  # [B, Seq, Hidden]
                    act_device = act.device
                    batch_range_on_device = batch_range.to(act_device)
                    last_token_indices_on_device = last_token_indices.to(act_device)
                    last_act = act[batch_range_on_device, last_token_indices_on_device, :]  # [B, Hidden]
                    cf_mlp_cache.save_batch(indices.numpy(), l, last_act.cpu().float().numpy())
                    del batch_mlp_buffer[l]

        for h in hooks:
            h.remove()

        # 2) 再跑 Fact，收集最终概率分布
        print("Collecting Fact probabilities...")
        for batch in tqdm(dataloader, desc="Fact Inference"):
            indices = batch["index"]
            fact_prompts_formatted = [
                format_prompt_for_model(prompt, model_type) for prompt in batch["fact_prompt"]
            ]
            fact_inputs = tokenizer(
                fact_prompts_formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(device)
            input_ids = fact_inputs["input_ids"]
            attention_mask = fact_inputs.get("attention_mask", torch.ones_like(input_ids))
            last_token_indices = get_last_token_indices_safe(
                input_ids, attention_mask, tokenizer
            )
            batch_range = torch.arange(input_ids.shape[0], device=device)

            with torch.no_grad():
                outputs = model(**fact_inputs)

            logits_device = outputs.logits.device
            batch_range_on_logits = batch_range.to(logits_device)
            last_token_indices_on_logits = last_token_indices.to(logits_device)
            logits = outputs.logits[batch_range_on_logits, last_token_indices_on_logits, :]

            # 检查每个样本的 top-3 是否包含 yes/no token
            for i in range(logits.shape[0]):
                logits_row = logits[i, :].float()
                sample_idx = int(indices[i])
                log_top3_warning(
                    logits_row,
                    tokenizer=tokenizer,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    sample_idx=sample_idx,
                    prefix="Fact",
                    show_warnings=True,
                )

            probs = torch.softmax(logits.float(), dim=1).cpu()
            fact_probs_list.append(probs)

        fact_probs_all = torch.cat(fact_probs_list, dim=0)
        print("Data preparation done.")

        # ==========================================================
        # Step 2: Causal Intervention on MLP Layers (Total Effect)
        # ==========================================================
        print("=" * 80)
        print("Step 2: Running Causal Intervention on MLP layers (Total Effect)...")
        print("=" * 80)

        layer_kl_scores = np.zeros(num_layers, dtype=np.float64)

        # DEBUG: 标记是否已输出首次干预的调试信息
        first_intervention_printed = False

        for l in range(num_layers):
            print(f"Processing Layer {l}/{num_layers - 1}...")

            if not hasattr(model.model.layers[l], "mlp"):
                raise ValueError(f"Layer {l} has no mlp module")

            target_mlp = model.model.layers[l].mlp
            layer_kl_sum = 0.0
            total_samples = 0

            for batch in tqdm(dataloader, desc=f"Layer {l} Batches", leave=False):
                indices = batch["index"]
                current_batch_size = len(indices)
                total_samples += current_batch_size

                fact_prompts_formatted = [
                    format_prompt_for_model(prompt, model_type) for prompt in batch["fact_prompt"]
                ]
                fact_inputs = tokenizer(
                    fact_prompts_formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                ).to(device)
                input_ids = fact_inputs["input_ids"]
                attention_mask = fact_inputs.get("attention_mask", torch.ones_like(input_ids))
                last_token_indices = get_last_token_indices_safe(
                    input_ids, attention_mask, tokenizer
                )

                # 当前层 CF MLP 激活: [B, Hidden]
                cf_layer_data = cf_mlp_cache.get_batch_layer(indices, l)
                cf_layer_tensor = torch.from_numpy(cf_layer_data)

                # 目标分布（原始 Fact 概率）
                target_probs_batch = fact_probs_all[indices].to(device)

                # DEBUG: 输出首次干预的调试信息
                if not first_intervention_printed:
                    print("=" * 80)
                    print("DEBUG: First intervention to be applied (MLP layer):")
                    print(f"  Layer: {l}")
                    print(f"  Batch size: {current_batch_size}")
                    for i, idx in enumerate(indices):
                        print(f"  Sample {i} (index {idx.item()}):")
                        print(f"    Fact prompt: {batch['fact_prompt'][i]}")
                        print(f"    Race: {batch['race'][i]}")
                    print(f"  Intervention logic: Replace last-token MLP output of layer {l} with counterfactual activation")
                    print("=" * 80)
                    first_intervention_printed = True

                # 注册干预 hook：将该层最后 token 的 MLP 输出替换为 CF 激活
                hook_fn = get_mlp_last_token_patch_hook(cf_layer_tensor, last_token_indices)
                hook_handle = target_mlp.register_forward_hook(hook_fn)

                try:
                    with torch.no_grad():
                        outputs = model(**fact_inputs)

                    logits_device = outputs.logits.device
                    batch_range = torch.arange(current_batch_size, device=logits_device)
                    last_token_indices_on_logits = last_token_indices.to(logits_device)
                    logits = outputs.logits[batch_range, last_token_indices_on_logits, :]
                    patched_probs = torch.softmax(logits.float(), dim=1)

                    target_probs_on_device = target_probs_batch.to(logits_device)
                    epsilon = 1e-10
                    p = torch.clamp(target_probs_on_device, min=epsilon)
                    q = torch.clamp(patched_probs, min=epsilon)
                    kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)

                    layer_kl_sum += kl.sum().item()
                finally:
                    hook_handle.remove()

            avg_kl = layer_kl_sum / max(total_samples, 1)
            layer_kl_scores[l] = avg_kl
            print(f"Layer {l} Mean KL: {avg_kl:.6f}")

        # ==========================================================
        # Step 3: Visualization & Saving
        # ==========================================================
        print("Step 3: Saving results (per-layer KL heatmap)...")

        # 这里将一维的 layer_kl_scores 视为 [num_layers, 1] 的“热力图”
        heatmap_kl = layer_kl_scores.reshape(num_layers, 1)
        plot_kl_heatmap(
            heatmap_kl,
            os.path.join(args.output_dir, "heatmap_kl_mlp.png"),
            title="KL Divergence (Total Effect, MLP layers) - Race (Resume)",
            num_layers=num_layers,
            num_heads=1,
        )

        results_data = {
            "layer_kl_scores": layer_kl_scores,
            "model": args.model_path,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "total_samples": int(min_len),
            "white_count": int(white_count),
            "black_count": int(black_count),
            "intervention_method": "mlp_total_effect",
            "intervention_description": "Replace last-token MLP output of a layer with counterfactual activation",
        }

        with open(os.path.join(args.output_dir, "results_mlp.pkl"), "wb") as f:
            pickle.dump(results_data, f)

        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY (MLP Layers, Resume Dataset)")
        print("=" * 60)
        print(f"Model: {args.model_path}")
        print(f"Total Samples: {min_len}")
        print(f"  - White: {white_count}")
        print(f"  - Black: {black_count}")
        print(f"\nModel Architecture:")
        print(f"  - Layers: {num_layers}")
        print(f"  - Hidden Size: {hidden_size}")
        print(f"\nIntervention Method:")
        print(f"  - Method: MLP Total Effect")
        print(f"  - Description: Replace last-token MLP output of a layer with counterfactual activation")
        valid_kl = layer_kl_scores[np.isfinite(layer_kl_scores)]
        if len(valid_kl) > 0:
            print(f"\nKL Divergence Statistics:")
            print(f"  - Max KL: {np.max(valid_kl):.6f}")
            print(f"  - Min KL: {np.min(valid_kl):.6f}")
            print(f"  - Mean KL: {np.mean(valid_kl):.6f}")
            print(f"  - Median KL: {np.median(valid_kl):.6f}")
            print(f"  - Std KL: {np.std(valid_kl):.6f}")
        print(f"\nOutput Directory: {args.output_dir}")
        print(f"  - Heatmap: heatmap_kl_mlp.png")
        print(f"  - Results: results_mlp.pkl")
        print("=" * 60)
        print("Done.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
