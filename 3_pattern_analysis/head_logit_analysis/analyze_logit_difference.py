#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP12: 分析注意力头对 Logits 的贡献（Logit Difference Analysis）

本脚本实现以下过程：
1. 获取 $W_U$ 矩阵（lm_head.weight）
2. 提取注意力头输出 $h_{l,h}$（经过 $W_O$ 投影）
3. 计算 logits：$h_{l,h} \cdot W_U^T$
4. 计算 Logit Difference：$\Delta_{l,h} = \text{logits}_{l,h}[\text{id}_{yes}] - \text{logits}_{l,h}[\text{id}_{no}]$
5. 计算歧视性变动（使用反事实输入对）

参考 exp2_old/analyze_race_sensitive_heads.py 获取激活值的过程。
"""

import json
import os
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# 导入上层目录的工具函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from util import (
    extract_race_from_query, create_counterfactual_by_race,
    get_model_config
)
from sampling import sample_resume_data_by_race
from prompt import build_category_prompt, format_prompt_for_model, resolve_model_type, add_yes_no_instruction
from probability import get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES
from hook import (
    get_last_token_indices_safe,
    get_activation_hook_for_intervention,
    create_config_detection_hook
)
from cache import DiskActivationCache
from plot import plot_kl_heatmap


class InterventionDataset(Dataset):
    """用于干预分析的数据集"""
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


def extract_head_output_after_o_proj(
    layer_activations: torch.Tensor,
    head_idx: int,
    o_proj: torch.nn.Module,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    从层激活值中提取指定头经过 o_proj 投影后的输出。
    
    Args:
        layer_activations: 层的激活值 [Batch, Seq, Num_Heads, Head_Dim]
        head_idx: 要提取的头索引
        o_proj: 输出投影层（o_proj）
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        该头经过 o_proj 投影后的输出 [Batch, Seq, Hidden_Dim]
    """
    # 提取指定头的激活值: [Batch, Seq, Head_Dim]
    head_activation = layer_activations[:, :, head_idx, :]
    
    # 获取 o_proj 的权重
    o_proj_weight = o_proj.weight  # [Hidden_Dim, Num_Heads * Head_Dim]
    
    # 提取该头对应的权重切片
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim
    o_proj_slice = o_proj_weight[:, start_idx:end_idx]  # [Hidden_Dim, Head_Dim]
    
    # 计算投影: [Batch, Seq, Head_Dim] @ [Head_Dim, Hidden_Dim] -> [Batch, Seq, Hidden_Dim]
    head_output = torch.matmul(head_activation, o_proj_slice.t())
    
    return head_output


def compute_logit_difference(
    head_output: torch.Tensor,
    w_u: torch.Tensor,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """
    计算注意力头对 Logits 的贡献，并计算 Logit Difference。
    
    Args:
        head_output: 头经过 o_proj 投影后的输出 [Batch, Seq, Hidden_Dim]
        w_u: $W_U$ 矩阵（lm_head.weight）[Vocab_Size, Hidden_Dim]
        yes_id: "Yes" 的 token ID
        no_id: "No" 的 token ID
        
    Returns:
        Logit Difference: [Batch, Seq]
    """
    # 计算 logits: head_output @ w_u.T
    # [Batch, Seq, Hidden_Dim] @ [Hidden_Dim, Vocab_Size] -> [Batch, Seq, Vocab_Size]
    logits = torch.matmul(head_output, w_u.t())
    
    # 提取 Yes 和 No 的 logits
    yes_logits = logits[:, :, yes_id]  # [Batch, Seq]
    no_logits = logits[:, :, no_id]    # [Batch, Seq]
    
    # 计算 Logit Difference
    delta = yes_logits - no_logits  # [Batch, Seq]
    
    return delta


def main():
    parser = argparse.ArgumentParser(
        description="EXP12: Analyze logit difference for attention heads"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_json_path", type=str, 
                       default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json")
    parser.add_argument("--output_dir", type=str, default="logit_difference_analysis")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture for prompt formatting.",
    )
    parser.add_argument("--random_sampling", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
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
        help="Disable balanced sampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    hidden_size = config["hidden_size"]
    
    print(f"Layers: {num_layers}, Heads: {num_heads}, Head Dim: {head_dim}, Hidden Size: {hidden_size}")

    # 解析模型类型
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    print(f"Using model_type: {model_type}")

    device = torch.device(args.device)

    # 准备 yes/no token id
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs: {no_ids}")
    
    # 使用第一个 yes 和 no token ID（如果需要，可以改为使用所有候选的平均值）
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None
    
    if yes_id is None or no_id is None:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    
    # 获取 $W_U$ 矩阵（lm_head.weight）
    if not hasattr(model, "lm_head"):
        raise ValueError("Model does not have lm_head attribute.")
    
    w_u = model.lm_head.weight  # [Vocab_Size, Hidden_Dim]
    print(f"W_U shape: {w_u.shape}")
    
    # 检测实际的模型配置（通过运行一个样本）
    print("Detecting actual model configuration from model...")
    temp_buffer = {}
    detect_hook_fn = create_config_detection_hook(temp_buffer)
    temp_hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(detect_hook_fn)
    
    # 加载数据
    print(f"Loading dataset from {args.dataset_json_path}...")
    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of records.")
    
    # 采样数据
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
    print(f"  - White: {white_count}, Black: {black_count}")
    
    # 准备 fact 数据
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

    # 运行一个测试样本以检测配置
    if len(fact_data) > 0:
        test_prompt = format_prompt_for_model(fact_data[0]["query"], model_type)
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

    # 创建数据集和数据加载器
    dataset_obj = InterventionDataset(fact_data, cf_data)
    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # Step 1: 收集 Fact 和 CF 的激活值
    # ==========================================
    print("=" * 80)
    print("Step 1: Collecting Fact and CF activations...")
    print("=" * 80)

    batch_activations_buffer = {}
    
    # 注册 Hooks 用于收集激活值
    hooks = []
    for l in range(num_layers):
        if hasattr(model.model.layers[l].self_attn, "o_proj"):
            layer_module = model.model.layers[l].self_attn.o_proj
        else:
            raise ValueError("Cannot find o_proj")
        hook_fn = get_activation_hook_for_intervention(
            l, num_heads, head_dim, batch_activations_buffer
        )
        hooks.append(layer_module.register_forward_hook(hook_fn))

    # 存储每层每头的激活值（Fact 和 CF）
    fact_activations = {}  # {(layer, head): [N, Head_Dim]}
    cf_activations = {}    # {(layer, head): [N, Head_Dim]}

    # 收集 Fact 激活值
    print("Collecting Fact activations...")
    for batch in tqdm(dataloader, desc="Fact Activations"):
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
            _ = model(**fact_inputs)

        # 提取每层每头的激活值（最后一个 token）
        for l in range(num_layers):
            if l in batch_activations_buffer:
                act = batch_activations_buffer[l]  # [B, Seq, H, D]
                act_device = act.device
                batch_range_on_device = batch_range.to(act_device)
                last_token_indices_on_device = last_token_indices.to(act_device)
                last_act = act[batch_range_on_device, last_token_indices_on_device, :, :]  # [B, H, D]
                
                # 存储每个头的激活值
                for h in range(num_heads):
                    head_act = last_act[:, h, :].cpu().numpy()  # [B, Head_Dim]
                    key = (l, h)
                    if key not in fact_activations:
                        fact_activations[key] = []
                    fact_activations[key].append(head_act)
                
                del batch_activations_buffer[l]

    # 收集 CF 激活值
    print("Collecting CF activations...")
    for batch in tqdm(dataloader, desc="CF Activations"):
        indices = batch["index"]
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

        # 提取每层每头的激活值（最后一个 token）
        for l in range(num_layers):
            if l in batch_activations_buffer:
                act = batch_activations_buffer[l]  # [B, Seq, H, D]
                act_device = act.device
                batch_range_on_device = batch_range.to(act_device)
                last_token_indices_on_device = last_token_indices.to(act_device)
                last_act = act[batch_range_on_device, last_token_indices_on_device, :, :]  # [B, H, D]
                
                # 存储每个头的激活值
                for h in range(num_heads):
                    head_act = last_act[:, h, :].cpu().numpy()  # [B, Head_Dim]
                    key = (l, h)
                    if key not in cf_activations:
                        cf_activations[key] = []
                    cf_activations[key].append(head_act)
                
                del batch_activations_buffer[l]

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 合并所有批次的激活值
    print("Merging activations...")
    for key in fact_activations:
        fact_activations[key] = np.concatenate(fact_activations[key], axis=0)  # [N, Head_Dim]
    for key in cf_activations:
        cf_activations[key] = np.concatenate(cf_activations[key], axis=0)  # [N, Head_Dim]

    print(f"Fact activations collected: {len(fact_activations)} (layer, head) pairs")
    print(f"CF activations collected: {len(cf_activations)} (layer, head) pairs")

    # ==========================================
    # Step 2: 计算 Logit Difference
    # ==========================================
    print("=" * 80)
    print("Step 2: Computing Logit Differences...")
    print("=" * 80)

    # 存储每层每头的 Logit Difference
    fact_logit_diffs = {}  # {(layer, head): [N]}
    cf_logit_diffs = {}    # {(layer, head): [N]}
    
    # 将 W_U 移到合适的设备
    w_u_device = w_u.device

    for l in range(num_layers):
        # 获取该层的 o_proj
        o_proj = model.model.layers[l].self_attn.o_proj
        o_proj_device = next(o_proj.parameters()).device
        
        for h in range(num_heads):
            key = (l, h)
            
            if key not in fact_activations or key not in cf_activations:
                continue
            
            # 获取 o_proj 的权重并确定数据类型
            o_proj_weight = o_proj.weight  # [Hidden_Dim, Num_Heads * Head_Dim]
            o_proj_dtype = o_proj_weight.dtype
            
            # 获取该头的激活值，并转换为与 o_proj 相同的数据类型
            fact_head_act = torch.from_numpy(fact_activations[key]).to(dtype=o_proj_dtype, device=o_proj_device)  # [N, Head_Dim]
            cf_head_act = torch.from_numpy(cf_activations[key]).to(dtype=o_proj_dtype, device=o_proj_device)  # [N, Head_Dim]
            
            # 提取该头经过 o_proj 投影后的输出
            # 注意：我们需要将 [N, Head_Dim] 扩展为 [N, 1, Head_Dim] 以便与 o_proj 计算
            fact_head_act_expanded = fact_head_act.unsqueeze(1)  # [N, 1, Head_Dim]
            cf_head_act_expanded = cf_head_act.unsqueeze(1)  # [N, 1, Head_Dim]
            
            # 提取该头对应的权重切片
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            o_proj_slice = o_proj_weight[:, start_idx:end_idx]  # [Hidden_Dim, Head_Dim]
            
            # 计算投影: [N, 1, Head_Dim] @ [Head_Dim, Hidden_Dim] -> [N, 1, Hidden_Dim]
            fact_head_output = torch.matmul(fact_head_act_expanded, o_proj_slice.t())  # [N, 1, Hidden_Dim]
            cf_head_output = torch.matmul(cf_head_act_expanded, o_proj_slice.t())  # [N, 1, Hidden_Dim]
            
            # 移除序列维度: [N, Hidden_Dim]
            fact_head_output = fact_head_output.squeeze(1)
            cf_head_output = cf_head_output.squeeze(1)
            
            # 将 W_U 移到相同的设备和数据类型
            w_u_on_device = w_u.to(dtype=o_proj_dtype, device=o_proj_device)
            
            # 计算 logits: [N, Hidden_Dim] @ [Hidden_Dim, Vocab_Size] -> [N, Vocab_Size]
            fact_logits = torch.matmul(fact_head_output, w_u_on_device.t())  # [N, Vocab_Size]
            cf_logits = torch.matmul(cf_head_output, w_u_on_device.t())  # [N, Vocab_Size]
            
            # 提取 Yes 和 No 的 logits
            yes_id_tensor = torch.tensor(yes_id, device=o_proj_device)
            no_id_tensor = torch.tensor(no_id, device=o_proj_device)
            
            fact_yes_logits = fact_logits[:, yes_id_tensor]  # [N]
            fact_no_logits = fact_logits[:, no_id_tensor]    # [N]
            cf_yes_logits = cf_logits[:, yes_id_tensor]      # [N]
            cf_no_logits = cf_logits[:, no_id_tensor]        # [N]
            
            # 计算 Logit Difference
            fact_delta = fact_yes_logits - fact_no_logits  # [N]
            cf_delta = cf_yes_logits - cf_no_logits        # [N]
            
            fact_logit_diffs[key] = fact_delta.detach().cpu().numpy()
            cf_logit_diffs[key] = cf_delta.detach().cpu().numpy()

    print(f"Computed logit differences for {len(fact_logit_diffs)} (layer, head) pairs")

    # ==========================================
    # Step 3: 计算歧视性变动
    # ==========================================
    print("=" * 80)
    print("Step 3: Computing Discriminatory Changes...")
    print("=" * 80)

    # 歧视性变动：$\tilde{\Delta}_{l,h} = \Delta_{l,h}^{CF} - \Delta_{l,h}^{Fact}$
    discriminatory_changes = {}  # {(layer, head): [N]}
    
    for key in fact_logit_diffs:
        if key in cf_logit_diffs:
            fact_delta = fact_logit_diffs[key]  # [N]
            cf_delta = cf_logit_diffs[key]       # [N]
            
            # 歧视性变动 = CF 的 Logit Difference - Fact 的 Logit Difference
            tilde_delta = cf_delta - fact_delta  # [N]
            discriminatory_changes[key] = tilde_delta

    print(f"Computed discriminatory changes for {len(discriminatory_changes)} (layer, head) pairs")

    # ==========================================
    # Step 4: 统计分析和保存结果
    # ==========================================
    print("=" * 80)
    print("Step 4: Statistical Analysis and Saving Results...")
    print("=" * 80)

    # 计算每个头的平均歧视性变动
    mean_discriminatory_changes = {}  # {(layer, head): float}
    std_discriminatory_changes = {}   # {(layer, head): float}
    
    for key, tilde_delta in discriminatory_changes.items():
        mean_discriminatory_changes[key] = float(np.mean(tilde_delta))
        std_discriminatory_changes[key] = float(np.std(tilde_delta))

    # 创建热力图数据（平均歧视性变动）
    heatmap_data = np.zeros((num_layers, num_heads), dtype=np.float64)
    for (l, h), mean_val in mean_discriminatory_changes.items():
        heatmap_data[l, h] = mean_val

    # 保存结果
    results_data = {
        "fact_logit_diffs": {f"L{l}_H{h}": fact_logit_diffs[(l, h)].tolist() 
                             for (l, h) in fact_logit_diffs},
        "cf_logit_diffs": {f"L{l}_H{h}": cf_logit_diffs[(l, h)].tolist() 
                          for (l, h) in cf_logit_diffs},
        "discriminatory_changes": {f"L{l}_H{h}": discriminatory_changes[(l, h)].tolist() 
                                  for (l, h) in discriminatory_changes},
        "mean_discriminatory_changes": {f"L{l}_H{h}": mean_discriminatory_changes[(l, h)] 
                                        for (l, h) in mean_discriminatory_changes},
        "std_discriminatory_changes": {f"L{l}_H{h}": std_discriminatory_changes[(l, h)] 
                                       for (l, h) in std_discriminatory_changes},
        "heatmap_data": heatmap_data.tolist(),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "yes_id": yes_id,
        "no_id": no_id,
        "num_samples": min_len,
    }

    # 保存为 pickle
    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_data, f)
    
    # 保存为 JSON（只保存统计信息，不保存所有样本的数据）
    json_results = {
        "mean_discriminatory_changes": {f"L{l}_H{h}": mean_discriminatory_changes[(l, h)] 
                                        for (l, h) in mean_discriminatory_changes},
        "std_discriminatory_changes": {f"L{l}_H{h}": std_discriminatory_changes[(l, h)] 
                                      for (l, h) in std_discriminatory_changes},
        "heatmap_data": heatmap_data.tolist(),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "yes_id": yes_id,
        "no_id": no_id,
        "num_samples": min_len,
    }
    
    with open(os.path.join(args.output_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    # 绘制热力图
    print("Generating heatmap visualization...")
    plot_kl_heatmap(
        heatmap_data,
        os.path.join(args.output_dir, "heatmap_discriminatory_changes.png"),
        title="Mean Discriminatory Changes (Logit Difference: CF - Fact)",
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # 打印统计信息
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Total Samples: {min_len}")
    print(f"  - White: {white_count}, Black: {black_count}")
    print(f"\nModel Architecture:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Heads per Layer: {num_heads}")
    print(f"  - Head Dimension: {head_dim}")
    print(f"  - Hidden Size: {hidden_size}")
    print(f"\nLogit Difference Analysis:")
    print(f"  - Yes Token ID: {yes_id}")
    print(f"  - No Token ID: {no_id}")
    
    # 找出歧视性变动最大的头
    sorted_heads = sorted(mean_discriminatory_changes.items(), 
                         key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 10 Heads by Absolute Discriminatory Change:")
    for i, ((l, h), mean_val) in enumerate(sorted_heads[:10], 1):
        std_val = std_discriminatory_changes[(l, h)]
        print(f"  {i:2d}. Layer {l:2d}, Head {h:2d}: Mean={mean_val:8.4f}, Std={std_val:8.4f}")
    
    print(f"\nOutput Directory: {args.output_dir}")
    print(f"  - Full Results: results.pkl")
    print(f"  - Summary: results_summary.json")
    print("="*60)
    print("Done.")


if __name__ == "__main__":
    main()
