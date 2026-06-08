#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 Resume 数据集中种族敏感的注意力头（Race-Sensitive Attention Heads）
修改版本：干预逻辑修改为：
- 每一次干预的时候，将每个头输出的激活值替换为事实时的激活值
- 而将选取的头修改为反事实的激活值
"""

import json
import os
import csv
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

from util import (
    extract_race_from_query, create_counterfactual_by_race,
    compute_elbow_point, compute_elbow_point_by_acceleration, compute_rank_array
)
from sampling import sample_resume_data_by_race
from prompt import build_category_prompt, format_prompt_for_model, resolve_model_type
from probability import get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES, log_top3_warning
from hook import (
    get_last_token_indices_safe,
    get_activation_hook_for_intervention, get_patch_hook_modified,
    create_config_detection_hook
)
from util import get_model_config
from model_adapter import get_model_adapter
from cache import DiskActivationCache
from plot import plot_kl_heatmap, plot_elbow_point_vs_rank, plot_rank_heatmap
from prompt import add_yes_no_instruction


class InterventionDataset(Dataset):
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
            "race": race if race else "Unknown"
        }

def _load_samples_by_csv_indices(
    dataset: List[dict],
    csv_path: str,
    sample_size: int,
) -> List[dict]:
    """
    Load samples by following the order of `index` column in a CSV file.
    The CSV is expected to have a header containing at least: index

    Args:
        dataset: full dataset list (records)
        csv_path: path to CSV (e.g. biased_samples_*/biased_samples_ranking.csv)
        sample_size: number of samples to take from the CSV order (<=0 means all)

    Returns:
        sampled_data: list of dataset records in the CSV-specified order
    """
    if not csv_path:
        raise ValueError("csv_path must be non-empty.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    indices: List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "index" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'index' column. Got columns: {reader.fieldnames}")
        for row in reader:
            if "index" not in row:
                continue
            try:
                indices.append(int(row["index"]))
            except Exception:
                continue

    if sample_size and sample_size > 0:
        indices = indices[:sample_size]

    sampled: List[dict] = []
    for idx in indices:
        if 0 <= idx < len(dataset):
            sampled.append(dataset[idx])
        else:
            # Skip invalid indices but keep going.
            continue
    return sampled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_json_path", type=str, 
                       default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json")
    parser.add_argument("--output_dir", type=str, default="race_sensitive_heads_analysis")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model architecture for prompt formatting. Use 'auto' to infer from model/tokenizer.",
    )
    parser.add_argument("--random_sampling", action="store_true", default=False,
                        help="Use random sampling instead of sequential sampling.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (only used when --random_sampling is enabled).")
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
    parser.add_argument(
        "--elbow_method",
        type=str,
        default="acceleration",
        choices=["original", "acceleration"],
        help="Method to compute elbow point: 'original' (distance) or 'acceleration' (2nd derivative)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="llm_intervention_race_resume_")
    print(f"Temporary cache: {temp_dir}")

    try:
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

        adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
        print(f"Using architecture adapter: {adapter.family} ({adapter.head_activation_kind})")

        # 获取模型配置
        config = adapter.get_config()
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        hidden_size = config["hidden_size"]
        
        print(f"Layers: {num_layers}, Heads: {num_heads}, Head Dim: {head_dim}, Hidden Size: {hidden_size}")
        print(f"Verification: num_heads * head_dim = {num_heads * head_dim}, hidden_size = {hidden_size}")

        # 解析模型类型
        model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
        print(f"Using model_type: {model_type}")

        device = torch.device(args.device)

        # 准备 yes/no token id
        yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
        no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
        print(f"Yes token IDs: {yes_ids}")
        print(f"No token IDs: {no_ids}")
        
        # 加载数据
        print(f"Loading dataset from {args.dataset_json_path}...")
        with open(args.dataset_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            raise ValueError("Dataset should be a list of records.")
        
        # 采样数据：
        # 1) 若提供 --sample_csv_path，则按 CSV 的 index 顺序取前 sample_size 个样本（覆盖其它采样参数）
        # 2) 否则使用原有的按种族平衡/随机采样逻辑
        if args.sample_csv_path:
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
        
        # 打印采样统计
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
        
            
            fact_item = {
                "query": summary,
                "summary": summary,
                "category": category,
                "race": race,
                "ID": item.get("ID", 0),
            }
            fact_data.append(fact_item)
        
        # 创建反事实数据
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

        # 先检测实际的模型配置（通过运行一个样本）
        print("Detecting actual model configuration from model...")
        temp_buffer = {}
        
        # 注册临时 hook 检测
        detect_hook_fn = create_config_detection_hook(temp_buffer)
        temp_hook = adapter.register_config_detection_hook(temp_buffer)
        # 运行一个样本
        if len(fact_data) > 0:
            test_prompt = format_prompt_for_model(fact_data[0]["query"], model_type)
            # DEBUG: 输出第一个样本用于调试
            print("=" * 80)
            print("DEBUG: First sample to be processed by model:")
            print(f"  Original query: {fact_data[0]['query']}")
            print(f"  Formatted prompt: {test_prompt}")
            print(f"  Race: {fact_data[0].get('race', 'Unknown')}")
            print(f"  Category: {fact_data[0].get('category', 'Unknown')}")
            print("=" * 80)
            test_inputs = tokenizer([test_prompt], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                _ = model(**test_inputs)
        temp_hook.remove()
        
        # 更新配置
        if 'num_heads' in temp_buffer and temp_buffer['num_heads'] is not None:
            detected_num_heads = temp_buffer['num_heads']
            detected_head_dim = temp_buffer['head_dim']
            
            if detected_num_heads != num_heads or detected_head_dim != head_dim:
                print(f"Detected configuration mismatch!")
                print(f"  Initial: num_heads={num_heads}, head_dim={head_dim}")
                print(f"  Actual: num_heads={detected_num_heads}, head_dim={detected_head_dim}")
                print(f"Updating configuration to detected values")
                num_heads = detected_num_heads
                head_dim = detected_head_dim
        else:
            print(f"Warning: Could not detect model configuration. Using initial values: num_heads={num_heads}, head_dim={head_dim}")

        # 初始化缓存：CF 激活（用于干预）和 Fact 激活（用于计算种族均值和干预）
        cf_activations_cache = DiskActivationCache(min_len, num_layers, num_heads, head_dim, "cf", temp_dir)
        fact_activations_cache = DiskActivationCache(min_len, num_layers, num_heads, head_dim, "fact", temp_dir)

        # 存储 Fact 的最终概率分布
        fact_probs_list = []

        dataset_obj = InterventionDataset(fact_data, cf_data)
        dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

        # ==========================================
        # Step 1: Data Preparation
        #   1) 跑 CF，缓存每层每头在 last token 的输出 (作为干预源)
        #   2) 跑 Fact，缓存最后 token 的概率分布 (作为 KL 基准) 和激活值 (用于干预)
        # ==========================================
        print("=" * 80)
        print("Step 1: Preparing Data (Fact Logits & Activations, CF Activations)...")
        print("=" * 80)

        batch_activations_buffer = {}

        # 注册 Hooks（仅用于 CF 跑）
        hooks = []
        for l in range(num_layers):
            hooks.append(adapter.register_head_activation_hook(l, num_heads, head_dim, batch_activations_buffer))

        # 1) 先跑 CF，收集所有层的 head 激活
        for batch in tqdm(dataloader, desc="Collecting CF Activations"):
            indices = batch["index"]

            # 根据模型类型格式化 prompt
            cf_prompts_formatted = [format_prompt_for_model(prompt, model_type) for prompt in batch["cf_prompt"]]
            cf_inputs = tokenizer(cf_prompts_formatted, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
            attention_mask = cf_inputs.get("attention_mask", torch.ones_like(cf_inputs["input_ids"]))
            last_token_indices = get_last_token_indices_safe(
                cf_inputs["input_ids"], attention_mask, tokenizer
            )
            batch_range = torch.arange(cf_inputs["input_ids"].shape[0], device=device)

            batch_activations_buffer.clear()
            with torch.no_grad():
                _ = model(**cf_inputs)

            # 将每层的 [B, Seq, H, D] 存成 [B, H, D]（last token）
            for l in range(num_layers):
                if l in batch_activations_buffer:
                    act = batch_activations_buffer[l]
                    act_device = act.device
                    batch_range_on_device = batch_range.to(act_device)
                    last_token_indices_on_device = last_token_indices.to(act_device)
                    last_act = act[batch_range_on_device, last_token_indices_on_device, :, :]  # [B, H, D]
                    cf_activations_cache.save_batch(indices.numpy(), l, last_act.cpu().float().numpy())
                    del batch_activations_buffer[l]

        # 移除 CF hook
        for h in hooks:
            h.remove()

        # 2) 再跑 Fact，收集最终概率分布和激活值
        print("Collecting Fact probabilities and activations...")
        hooks_fact = []
        for l in range(num_layers):
            hooks_fact.append(adapter.register_head_activation_hook(l, num_heads, head_dim, batch_activations_buffer))
        
        for batch in tqdm(dataloader, desc="Fact Inference"):
            indices = batch["index"]
            fact_prompts_formatted = [format_prompt_for_model(prompt, model_type) for prompt in batch["fact_prompt"]]
            fact_inputs = tokenizer(fact_prompts_formatted, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
            attention_mask = fact_inputs.get("attention_mask", torch.ones_like(fact_inputs["input_ids"]))
            last_token_indices = get_last_token_indices_safe(
                fact_inputs["input_ids"], attention_mask, tokenizer
            )
            batch_range = torch.arange(fact_inputs["input_ids"].shape[0], device=device)

            batch_activations_buffer.clear()
            with torch.no_grad():
                outputs = model(**fact_inputs)

            # 收集最终概率分布
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
            
            # 收集激活值
            for l in range(num_layers):
                if l in batch_activations_buffer:
                    act = batch_activations_buffer[l]
                    act_device = act.device
                    batch_range_on_device = batch_range.to(act_device)
                    last_token_indices_on_device = last_token_indices.to(act_device)
                    last_act = act[batch_range_on_device, last_token_indices_on_device, :, :]  # [B, H, D]
                    fact_activations_cache.save_batch(indices.numpy(), l, last_act.cpu().float().numpy())
                    del batch_activations_buffer[l]
        
        # 移除 Fact hook
        for h in hooks_fact:
            h.remove()

        # 合并所有样本的 Fact 概率 [N, V]
        fact_probs_all = torch.cat(fact_probs_list, dim=0)
        print("Data preparation done.")

        # ==========================================
        # Step 2: 按种族计算每个头的激活值均值
        # ==========================================
        print("Step 2: Computing race-specific mean activations per head...")
        white_indices = [i for i, r in enumerate(races_list) if r == "White"]
        black_indices = [i for i, r in enumerate(races_list) if r == "Black"]
        print(f"White: {len(white_indices)}, Black: {len(black_indices)}")

        # 为每个 (layer, head) 计算 White/Black 的激活值均值与标准差
        white_head_means = {}
        black_head_means = {}
        white_stds = {}
        black_stds = {}
        combined_stds = {}
        
        for l in range(num_layers):
            for h in range(num_heads):
                # 获取该头所有样本的激活值: [N, Head_Dim]
                head_activations = fact_activations_cache.get_column(l, h)
                
                # 计算 White 均值
                if white_indices:
                    white_head_means[(l, h)] = np.mean(head_activations[white_indices], axis=0)
                    white_stds[(l, h)] = np.std(head_activations[white_indices], axis=0) if len(white_indices) > 1 else np.ones(head_dim, dtype=np.float32)
                else:
                    white_head_means[(l, h)] = np.zeros(head_dim, dtype=np.float32)
                    white_stds[(l, h)] = np.ones(head_dim, dtype=np.float32)
                
                # 计算 Black 均值
                if black_indices:
                    black_head_means[(l, h)] = np.mean(head_activations[black_indices], axis=0)
                    black_stds[(l, h)] = np.std(head_activations[black_indices], axis=0) if len(black_indices) > 1 else np.ones(head_dim, dtype=np.float32)
                else:
                    black_head_means[(l, h)] = np.zeros(head_dim, dtype=np.float32)
                    black_stds[(l, h)] = np.ones(head_dim, dtype=np.float32)

                # 计算 combined_std
                if len(head_activations) > 0:
                    combined_stds[(l, h)] = np.mean(head_activations, axis=0)
                else:
                    combined_stds[(l, h)] = np.zeros(head_dim, dtype=np.float32)
        
        print(f"Computed mean activations for {num_layers * num_heads} heads.")

        # ==========================================
        # Step 3: Causal Intervention Loop (Modified Total Effect)
        # ==========================================
        print("=" * 80)
        print("Step 3: Running Modified Causal Intervention (Total Effect, full forward passes)...")
        print("Intervention logic: Set all heads to fact activations, then set selected head to counterfactual activation.")
        print("Warning: This step is computationally expensive.")
        print("=" * 80)

        heatmap_kl = np.zeros((num_layers, num_heads), dtype=np.float64)

        # DEBUG: 标记是否已输出首次干预的调试信息
        first_intervention_printed = False

        # 外层循环：层
        for l in range(num_layers):
            print(f"Processing Layer {l}/{num_layers - 1}...")

            layer_head_kl_sum = torch.zeros(num_heads, dtype=torch.float64, device="cpu")
            total_samples = 0

            # 遍历数据 Batch
            for batch in tqdm(dataloader, desc=f"Layer {l} Batches", leave=False):
                indices = batch["index"]
                current_batch_size = len(indices)
                total_samples += current_batch_size

                # Fact 输入
                fact_prompts_formatted = [format_prompt_for_model(prompt, model_type) for prompt in batch["fact_prompt"]]
                fact_inputs = tokenizer(fact_prompts_formatted, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
                attention_mask = fact_inputs.get("attention_mask", torch.ones_like(fact_inputs["input_ids"]))
                last_token_indices = get_last_token_indices_safe(
                    fact_inputs["input_ids"], attention_mask, tokenizer
                )

                # 当前层 Fact 激活: [B, H, D]
                fact_layer_data = fact_activations_cache.get_batch_layer(indices, l)
                # 当前层 CF 激活: [B, H, D]
                cf_layer_data = cf_activations_cache.get_batch_layer(indices, l)
                
                target_module = adapter.get_head_activation_module(l)
                try:
                    target_device = next(target_module.parameters()).device
                except StopIteration:
                    target_device = next(model.parameters()).device
                fact_layer_tensor = torch.from_numpy(fact_layer_data).to(target_device)
                cf_layer_tensor = torch.from_numpy(cf_layer_data).to(target_device)

                # 目标分布（原始 Fact 概率）
                target_probs_batch = fact_probs_all[indices].to(device)

                # 遍历该层所有头，对每个头做一次干预并全前向
                for h in range(num_heads):
                    # DEBUG: 输出首次干预的调试信息
                    if not first_intervention_printed:
                        print("=" * 80)
                        print("DEBUG: First intervention to be applied:")
                        print(f"  Layer: {l}, Head: {h}")
                        print(f"  Batch size: {current_batch_size}")
                        for i, idx in enumerate(indices):
                            print(f"  Sample {i} (index {idx.item()}):")
                            print(f"    Fact prompt: {batch['fact_prompt'][i]}")
                            print(f"    Race: {batch['race'][i]}")
                        print(f"  Intervention logic: Set all heads to fact activations, then set head {h} to counterfactual activation")
                        print("=" * 80)
                        first_intervention_printed = True
                    hook_handle = adapter.register_head_patch_hook(
                        l, h, fact_layer_tensor, cf_layer_tensor, last_token_indices,
                        num_heads, head_dim
                    )

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

                        layer_head_kl_sum[h] += kl.sum().item()
                    finally:
                        hook_handle.remove()

            avg_kl = layer_head_kl_sum / total_samples
            heatmap_kl[l, :] = avg_kl.numpy()

        # ==========================================
        # Step 4: Visualization & Saving
        # ==========================================
        print("Step 4: Saving results (heatmap & elbow-based head selection)...")
        
        # 绘制 KL 热力图
        plot_kl_heatmap(
            heatmap_kl,
            os.path.join(args.output_dir, "heatmap_kl.png"),
            title="KL Divergence (Modified Total Effect: All heads=fact, Selected head=CF) - Race",
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # Elbow Method
        flat_kl = heatmap_kl.flatten()
        flat_kl = flat_kl[np.isfinite(flat_kl)]
        elbow_score = 0.0
        elbow_idx = 0
        if len(flat_kl) > 0:
            sorted_scores = np.sort(flat_kl)[::-1]
            if args.elbow_method == "acceleration":
                elbow_idx, elbow_score = compute_elbow_point_by_acceleration(sorted_scores)
            else:
                elbow_idx, elbow_score = compute_elbow_point(sorted_scores)
        else:
            elbow_idx = 0
            elbow_score = 0.0

        print(f"Elbow Score: {elbow_score}")
        
        selected = []
        for l in range(num_layers):
            for h in range(num_heads):
                if heatmap_kl[l, h] >= elbow_score:
                    selected.append({"layer": l, "head": h})
        
        # 计算排名数组
        rank_array = compute_rank_array(heatmap_kl)
        elbow_rank_value = int(elbow_idx) + 1
        elbow_kl_value = float(elbow_score)
        
        results_data = {
            "heatmap": heatmap_kl,
            "selected_heads": selected,
            "white_emb": white_head_means,
            "black_emb": black_head_means,
            "white_std": white_stds,
            "black_std": black_stds,
            "combined_std": combined_stds,
            "elbow_score": elbow_score,
            "elbow_idx": elbow_idx,
            "elbow_rank": elbow_rank_value,
            "elbow_kl_value": elbow_kl_value,
            "rank_array": rank_array,
            "intervention_method": "modified_total_effect",
            "intervention_description": "Set all heads to fact activations, then set selected head to counterfactual activation",
        }
        
        with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
            pickle.dump(results_data, f)

        with open(os.path.join(args.output_dir, "selected_heads_elbow.json"), "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        
        # 绘制肘点与排名图
        plot_elbow_point_vs_rank(
            heatmap_kl,
            elbow_idx,
            elbow_score,
            os.path.join(args.output_dir, "elbow_point_vs_rank.png"),
            title="Elbow Point vs Rank: KL Divergence (Modified Total Effect) - Race",
        )
        
        # 绘制排名热力图
        plot_rank_heatmap(
            heatmap_kl,
            os.path.join(args.output_dir, "rank_heatmap.png"),
            title="Rank Heatmap (KL Divergence, Modified Total Effect) - Race",
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # ==========================================
        # Print Summary
        # ==========================================
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Total Samples: {min_len}")
        print(f"  - White: {len(white_indices)}")
        print(f"  - Black: {len(black_indices)}")
        print(f"\nModel Architecture:")
        print(f"  - Layers: {num_layers}")
        print(f"  - Heads per Layer: {num_heads}")
        print(f"  - Head Dimension: {head_dim}")
        print(f"  - Total Attention Heads: {num_layers * num_heads}")
        print(f"\nIntervention Method:")
        print(f"  - Method: Modified Total Effect")
        print(f"  - Description: Set all heads to fact activations, then set selected head to counterfactual activation")
        print(f"\nKL Divergence Statistics:")
        valid_kl = heatmap_kl[np.isfinite(heatmap_kl)]
        if len(valid_kl) > 0:
            print(f"  - Max KL: {np.max(valid_kl):.6f}")
            print(f"  - Min KL: {np.min(valid_kl):.6f}")
            print(f"  - Mean KL: {np.mean(valid_kl):.6f}")
            print(f"  - Median KL: {np.median(valid_kl):.6f}")
            print(f"  - Std KL: {np.std(valid_kl):.6f}")
        print(f"\nSelected Race-Sensitive Heads:")
        print(f"  - Elbow Score Threshold: {elbow_score:.6f}")
        print(f"  - Elbow Point Rank: {elbow_rank_value}")
        print(f"  - Elbow Point KL Value: {elbow_kl_value:.6f}")
        print(f"  - Number of Selected Heads: {len(selected)}")
        print(f"  - Percentage: {len(selected)/(num_layers*num_heads)*100:.2f}%")
        
        # 按层统计选中的头
        layer_counts = {}
        for head_info in selected:
            layer = head_info["layer"]
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        if layer_counts:
            print(f"\nSelected Heads by Layer:")
            sorted_layers = sorted(layer_counts.items())
            for layer, count in sorted_layers:
                heads_in_layer = [h["head"] for h in selected if h["layer"] == layer]
                print(f"  - Layer {layer:2d}: {count:2d} heads {heads_in_layer}")
        
        # Top 10 heads by KL divergence
        top_heads = []
        for l in range(num_layers):
            for h in range(num_heads):
                if np.isfinite(heatmap_kl[l, h]):
                    top_heads.append((l, h, heatmap_kl[l, h]))
        top_heads.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 10 Heads by KL Divergence:")
        for i, (l, h, kl_val) in enumerate(top_heads[:10], 1):
            selected_mark = "✓" if {"layer": l, "head": h} in selected else " "
            print(f"  {i:2d}. Layer {l:2d}, Head {h:2d}: KL={kl_val:.6f} {selected_mark}")
        
        print(f"\nOutput Directory: {args.output_dir}")
        print(f"  - Heatmap: heatmap_kl.png")
        print(f"  - Rank Heatmap: rank_heatmap.png")
        print(f"  - Elbow Point vs Rank: elbow_point_vs_rank.png")
        print(f"  - Results: results.pkl")
        print(f"  - Selected Heads: selected_heads_elbow.json")
        print("="*60)
        print("Done.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
