#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP4: 精准微调 (Precision Fine-tuning) with Fairness Constraints

本脚本实现基于公平性约束的精准微调，结合以下技术：
1. 参数高效的 LoRA 注入（仅更新低秩矩阵）
2. 基于 ACE (Affine Concept Editing) 的公平锚点定义
3. 公平性约束损失函数（Fairness Loss）

训练流程：
第一步：参数高效的 LoRA 注入
  - 冻结原始权重 W_0，只在旁边外挂两个低秩矩阵 A 和 B
  - 更新量 ΔW = B × A，由于 B 和 A 的秩（Rank）很小，参数量极少
  - 仅对选定的种族敏感头（从 exp2_old/exp_heads.sh 输出选择）应用 LoRA

第二步：定义"公平锚点" (Fairness Anchor)
  - 收集数据：从 exp2_old/analyze_race_sensitive_heads.py 的 results.pkl 中
    提取不同种族（White vs Black）的激活向量均值
  - 提取方向向量 (d̃)：计算两个群体平均激活状态的差值，并进行白化处理
  - 确定中立点 (b)：将两个群体的中心投影到这条轴上，取它们的中点
    这个中点 b 代表绝对的公平

第三步：公平约束损失函数 (Fairness Loss)
  - 在训练时，使用公平性约束损失进行优化（KL 散度仅用于监控）
  - 让模型在处理信息时，经过 LoRA 调整后的激活值，在敏感属性轴上的投影
    必须靠近那个"中立点 b"
  - 如果投影偏离了 b，则产生损失值 L_f，逼迫模型修正

优化目标：
  L = λ * L_f
  其中：
  - L_f: 公平性损失（约束激活值投影到中立点）
  - λ: 公平性损失权重
  - KL 散度：仅用于训练过程监控（不参与反向更新）
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _apply_head_grad_mask_(
    model: nn.Module,
    head_masks: Dict[str, torch.Tensor],
) -> None:
    """将梯度按 head mask 置零，使更新只发生在指定 heads 对应的 hidden slices 上。"""
    if not head_masks:
        return

    for name, param in model.named_parameters():
        if (not param.requires_grad) or (param.grad is None):
            continue

        for layer_key, mask_1d in head_masks.items():
            if layer_key not in name:
                continue

            # 仅对 LoRA 参数做 mask（避免误伤其他可训练参数）
            # 常见命名：...q_proj.lora_A... / ...q_proj.lora_B...
            if "lora_A" in name:
                # lora_A: [r, in_features]，对输入维 in_features 做 mask
                if param.grad.ndim == 2 and param.grad.shape[1] == mask_1d.numel():
                    param.grad.mul_(mask_1d.to(param.grad.device).unsqueeze(0))
            elif "lora_B" in name:
                # lora_B: [out_features, r]，对输出维 out_features 做 mask
                if param.grad.ndim == 2 and param.grad.shape[0] == mask_1d.numel():
                    param.grad.mul_(mask_1d.to(param.grad.device).unsqueeze(1))
            else:
                # 其他 LoRA/adapter 参数暂不处理
                pass



from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sampling import sample_resume_data_by_race, load_samples_by_csv_indices
from util import create_counterfactual_by_race, get_model_config
from prompt import (
    build_resume_prompt,
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
)
from probability import (
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
)
from hook import (
    get_activation_hook_for_intervention,
    get_last_token_indices_safe,
    create_config_detection_hook,
)
from model_adapter import get_model_adapter


def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fairness_anchors(heads_analysis_dir: str, expected_head_dim: Optional[int] = None) -> Dict:
    """
    从 exp2_old 的分析结果中加载公平锚点信息。
    
    Args:
        heads_analysis_dir: exp2_old/analyze_race_sensitive_heads.py 的输出目录
        expected_head_dim: 期望的 head_dim，如果提供，将只保留维度匹配的头
        
    Returns:
        包含以下信息的字典：
        - selected_heads: 选中的种族敏感头列表 [{"layer": l, "head": h}, ...]（已过滤维度匹配的头）
        - white_emb: White 群体的激活均值 {(layer, head): np.array}
        - black_emb: Black 群体的激活均值 {(layer, head): np.array}
        - fairness_directions: 公平方向向量 {(layer, head): np.array}
        - fairness_anchors: 公平锚点 {(layer, head): float}
    """
    results_path = os.path.join(heads_analysis_dir, "results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"Loading fairness anchors from {results_path}")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    
    selected_heads = results.get("selected_heads", [])
    white_emb = results.get("white_emb", {})
    black_emb = results.get("black_emb", {})
    
    print(f"Loaded {len(selected_heads)} selected heads from results.pkl")
    if expected_head_dim is not None:
        print(f"Expected head_dim: {expected_head_dim}")
    
    # 计算公平方向向量和锚点
    fairness_directions = {}
    fairness_anchors = {}
    filtered_selected_heads = []
    
    # 统计维度分布
    dim_distribution = {}
    
    for head_info in selected_heads:
        layer = head_info["layer"]
        head = head_info["head"]
        key = (layer, head)
        
        if key not in white_emb or key not in black_emb:
            print(f"Warning: Missing embeddings for layer {layer}, head {head}")
            continue
        
        white_mean = white_emb[key]  # 应该是 [head_dim]
        black_mean = black_emb[key]  # 应该是 [head_dim]
        
        # 确保是 1D 数组
        if white_mean.ndim > 1:
            white_mean = white_mean.flatten()
        if black_mean.ndim > 1:
            black_mean = black_mean.flatten()
        
        # 检查维度是否一致
        if white_mean.shape != black_mean.shape:
            print(f"Warning: Dimension mismatch for layer {layer}, head {head}: "
                  f"white_shape={white_mean.shape}, black_shape={black_mean.shape}")
            continue
        
        # 统计维度分布
        actual_head_dim = white_mean.shape[0]
        dim_distribution[actual_head_dim] = dim_distribution.get(actual_head_dim, 0) + 1
        
        # 如果提供了期望的 head_dim，检查是否匹配
        if expected_head_dim is not None and actual_head_dim != expected_head_dim:
            continue
        
        # 计算方向向量：d = white_mean - black_mean
        direction = white_mean - black_mean
        
        # 归一化（白化处理的简化版本）
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.zeros_like(direction)
        
        fairness_directions[key] = direction
        
        # 计算中立点 b：两个群体在方向向量上的投影的中点
        white_proj = np.dot(white_mean, direction)
        black_proj = np.dot(black_mean, direction)
        anchor = (white_proj + black_proj) / 2.0
        
        fairness_anchors[key] = anchor
        filtered_selected_heads.append(head_info)
        if len(filtered_selected_heads) <= 10:  # 只打印前10个
            print(f"  Layer {layer}, Head {head}: direction_dim={direction.shape[0]}, anchor={anchor:.4f}")
    
    # 打印维度分布信息
    if dim_distribution:
        print(f"Dimension distribution in results.pkl: {dim_distribution}")
    
    print(f"Computed fairness directions and anchors for {len(fairness_directions)} heads "
          f"(filtered from {len(selected_heads)} heads)")
    
    if len(fairness_directions) == 0:
        raise ValueError(
            f"No valid heads found after dimension filtering! "
            f"Expected head_dim={expected_head_dim}, but found dimensions: {list(dim_distribution.keys())}. "
            f"Please ensure exp2_old analysis was done on the same model."
        )
    
    return {
        "selected_heads": filtered_selected_heads,
        "white_emb": white_emb,
        "black_emb": black_emb,
        "fairness_directions": fairness_directions,
        "fairness_anchors": fairness_anchors,
    }


def build_fact_and_counterfactual_dataset(
    dataset_json_path: str,
    max_samples: int,
    balanced: bool,
    random_sampling: bool,
    seed: int,
    sample_csv_path: Optional[str] = None,
    sample_size: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """构建事实与反事实配对数据集。"""
    print(f"Loading dataset from {dataset_json_path} ...")
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of records.")

    print(f"Total records in JSON: {len(dataset)}")

    if sample_csv_path:
        print(f"Sampling by CSV order: {sample_csv_path}")
        sampled_data, _, _ = load_samples_by_csv_indices(
            dataset=dataset,
            csv_path=sample_csv_path,
            sample_size=sample_size,
        )
        print(f"Sampled {len(sampled_data)} records from CSV (sample_size={sample_size}).")
    else:
        sampled_data = sample_resume_data_by_race(
            data_records=dataset,
            max_samples=max_samples,
            balanced=balanced,
            random_sampling=random_sampling,
            seed=seed,
        )
        print(f"Sampled {len(sampled_data)} records for fine-tuning.")

    fact_data: List[Dict] = []
    cf_data: List[Dict] = []
    for item in sampled_data:
        summary = item.get("summary", "")
        race = item.get("race", "")
        fact_item = {
            "query": summary,
            "summary": summary,
            "race": race,
            "ID": item.get("ID", 0),
        }
        fact_data.append(fact_item)

    print(f"Fact samples after filtering: {len(fact_data)}")

    print("Creating counterfactual data (flipping race)...")
    for fact_item in fact_data:
        cf_item = create_counterfactual_by_race(fact_item)
        cf_data.append(cf_item)

    min_len = min(len(fact_data), len(cf_data))
    fact_data = fact_data[:min_len]
    cf_data = cf_data[:min_len]

    print(f"Final paired samples (fact + cf): {min_len}")
    return fact_data, cf_data


class PairedDataset(Dataset):
    """事实与反事实配对数据集。"""

    def __init__(self, fact_data: List[Dict], cf_data: List[Dict]):
        assert len(fact_data) == len(cf_data)
        self.fact_data = fact_data
        self.cf_data = cf_data

    def __len__(self) -> int:
        return len(self.fact_data)

    def __getitem__(self, idx: int) -> Dict:
        fact_item = self.fact_data[idx]
        cf_item = self.cf_data[idx]
        return {
            "index": idx,
            "fact_query": fact_item.get("query", ""),
            "cf_query": cf_item.get("query", ""),
        }


def compute_targeted_kl(
    fact_logits: torch.Tensor,
    cf_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    只在指定 token（例如 Yes/No）对应的 logits 上计算 KL 散度。
    """
    f_logits = fact_logits[:, target_ids].float()
    c_logits = cf_logits[:, target_ids].float()

    f_probs = F.softmax(f_logits, dim=-1)
    c_probs = F.softmax(c_logits, dim=-1)

    return F.kl_div(c_probs.log(), f_probs, reduction="batchmean")


def get_last_token_logits(
    logits: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """获取最后一个非 padding token 的 logits。"""
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]

    last_token_indices = attention_mask.sum(dim=1) - 1
    last_token_indices = last_token_indices.clamp(min=0, max=seq_len - 1)

    batch_indices = torch.arange(batch_size, device=logits.device)
    last_logits = logits[batch_indices, last_token_indices, :]

    return last_logits


def compute_causal_lm_ce_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute causal LM CE from an existing forward pass."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# 不再使用全局 _activation_cache，改为使用 batch_activations_buffer（与 exp2_old 一致）


def compute_fairness_loss(
    batch_activations_buffer: Dict[int, torch.Tensor],
    fairness_info: Dict,
    last_token_indices: torch.Tensor,
    batch_range: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    计算公平性损失 L_f。
    
    对于每个选定的头，计算其激活值在公平方向上的投影，
    并约束该投影接近公平锚点。
    
    Args:
        batch_activations_buffer: 存储每层激活值的字典，形状为 [Batch, Seq, Num_Heads, Head_Dim]
        fairness_info: 公平性信息（包含方向向量和锚点）
        last_token_indices: 最后一个 token 的位置 [batch_size]
        batch_range: batch 索引 [batch_size]
        device: 目标设备（用于统一损失值的设备）
        
    Returns:
        公平性损失值（在指定设备上）
    """
    fairness_directions = fairness_info["fairness_directions"]
    fairness_anchors = fairness_info["fairness_anchors"]
    
    # 收集所有损失值（可能在不同设备上），然后统一转换到目标设备
    loss_values = []
    
    for key, direction in fairness_directions.items():
        layer_idx, head_idx = key
        anchor = fairness_anchors[key]
        
        # 检查该层的激活值是否存在
        if layer_idx not in batch_activations_buffer:
            continue
        
        # 获取该层的激活值 [Batch, Seq, Num_Heads, Head_Dim]
        layer_activations = batch_activations_buffer[layer_idx]
        act_device = layer_activations.device
        
        # 提取最后一个 token 的激活值
        batch_range_on_device = batch_range.to(act_device)
        last_token_indices_on_device = last_token_indices.to(act_device)
        
        # 提取最后一个 token 的所有头的激活值: [Batch, Num_Heads, Head_Dim]
        last_act_all_heads = layer_activations[batch_range_on_device, last_token_indices_on_device, :, :]
        
        # 提取指定头的激活值: [Batch, Head_Dim]
        last_activations = last_act_all_heads[:, head_idx, :]
        
        # 将方向向量转换为 tensor（在激活值的设备上）
        direction_tensor = torch.from_numpy(direction).float().to(last_activations.device)
        
        # 检查维度匹配
        if last_activations.shape[1] != direction_tensor.shape[0]:
            print(f"Warning: Dimension mismatch for layer {layer_idx}, head {head_idx}: "
                  f"activation_dim={last_activations.shape[1]}, direction_dim={direction_tensor.shape[0]}")
            continue
        
        # 计算投影：projection = activations · direction
        # last_activations: [batch_size, head_dim]
        # direction_tensor: [head_dim]
        projections = torch.sum(last_activations * direction_tensor, dim=1)  # [batch_size]
        
        # 计算与锚点的距离（在激活值的设备上）
        anchor_tensor = torch.tensor(anchor, dtype=torch.float32, device=projections.device)
        loss = torch.mean((projections - anchor_tensor) ** 2)
        
        # 将损失值转换到目标设备并收集
        loss_values.append(loss.to(device))
    
    if len(loss_values) > 0:
        # 所有损失值现在都在同一设备上，可以安全地计算平均值
        total_loss = sum(loss_values)
        return total_loss / len(loss_values)
    else:
        return torch.tensor(0.0, device=device)


def create_head_masks(
    selected_heads: List[Dict],
    num_heads: int,
    head_dim: int,
    layer_key_fn=None,
) -> Dict[str, torch.Tensor]:
    """
    为选定的头创建梯度 Mask。
    
    Args:
        selected_heads: 选中的头列表 [{"layer": l, "head": h}, ...]
        num_heads: 总头数
        head_dim: 每个头的维度
        
    Returns:
        字典，Key 为层标识，Value 为形状为 [num_heads * head_dim] 的 0/1 Mask Tensor
    """
    head_masks = {}
    layers_to_heads = {}
    for h in selected_heads:
        layers_to_heads.setdefault(h["layer"], []).append(h["head"])
    
    full_dim = num_heads * head_dim
    
    for layer_idx, head_indices in layers_to_heads.items():
        # 创建 1D mask
        mask = torch.zeros(full_dim)
        for h_idx in head_indices:
            mask[h_idx * head_dim : (h_idx + 1) * head_dim] = 1.0
            
        # 存储该层对应的 mask
        layer_key = layer_key_fn(layer_idx) if layer_key_fn is not None else f"layers.{layer_idx}.self_attn"
        head_masks[layer_key] = mask
            
    return head_masks


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    model_type: str,
    target_ids: torch.Tensor,
    fairness_info: Dict,
    fairness_lambda: float,
    activation_hooks: List,
    batch_activations_buffer: Dict[int, torch.Tensor],
    num_heads: int,
    head_dim: int,
    head_masks: Optional[Dict[str, torch.Tensor]] = None,
    loss_type: str = "kl",
    resume_prompt_mode: str = "category",
) -> Dict[str, float]:
    """训练一个 epoch，支持 KL 或 Fairness 损失。"""
    model.train()
    total_loss = 0.0
    total_kl_loss = 0.0  # KL 仅用于监控，不参与反向更新
    total_fairness_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        raw_fact_queries = batch["fact_query"]
        raw_cf_queries = batch["cf_query"]

        fact_base_prompts = [
            build_resume_prompt(summary, mode=resume_prompt_mode) for summary in raw_fact_queries
        ]
        cf_base_prompts = [
            build_resume_prompt(summary, mode=resume_prompt_mode) for summary in raw_cf_queries
        ]

        fact_instruction_prompts = [
            add_yes_no_instruction(p) for p in fact_base_prompts
        ]
        cf_instruction_prompts = [add_yes_no_instruction(p) for p in cf_base_prompts]

        fact_prompts = [
            format_prompt_for_model(p, model_type) for p in fact_instruction_prompts
        ]
        cf_prompts = [
            format_prompt_for_model(p, model_type) for p in cf_instruction_prompts
        ]

        # Tokenize
        fact_inputs = tokenizer(
            fact_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        cf_inputs = tokenizer(
            cf_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        fact_input_ids = fact_inputs["input_ids"]
        fact_attention_mask = fact_inputs.get(
            "attention_mask", torch.ones_like(fact_input_ids)
        )
        cf_input_ids = cf_inputs["input_ids"]
        cf_attention_mask = cf_inputs.get(
            "attention_mask", torch.ones_like(cf_input_ids)
        )

        # 获取最后一个 token 的位置（用于提取激活值）
        fact_last_token_indices = get_last_token_indices_safe(
            fact_input_ids, fact_attention_mask, tokenizer
        )
        batch_range = torch.arange(fact_input_ids.shape[0], device=device)

        # 清空激活缓存
        batch_activations_buffer.clear()

        # 前向：事实
        fact_outputs = model(
            input_ids=fact_input_ids, attention_mask=fact_attention_mask
        )
        fact_logits = fact_outputs.logits

        # 计算公平性损失（基于事实样本的激活值）
        fairness_loss = compute_fairness_loss(
            batch_activations_buffer,
            fairness_info,
            fact_last_token_indices,
            batch_range,
            device,
        )

        # 清空激活缓存，准备反事实前向（公平性损失只基于事实样本）
        batch_activations_buffer.clear()

        # 前向：反事实
        cf_outputs = model(input_ids=cf_input_ids, attention_mask=cf_attention_mask)
        cf_logits = cf_outputs.logits
        # Hooks also capture CF activations, but the fairness term only uses fact
        # activations. Drop these unused graph references before backward.
        batch_activations_buffer.clear()

        # 取最后一个 token 的 logits
        fact_last_logits = get_last_token_logits(
            fact_logits, fact_attention_mask
        )
        cf_last_logits = get_last_token_logits(cf_logits, cf_attention_mask)

        # 计算 KL 损失
        kl_loss = compute_targeted_kl(fact_last_logits, cf_last_logits, target_ids)

        ce_loss = None

        # 总损失：根据 loss_type 选择
        if loss_type == "kl":
            loss = kl_loss
        elif loss_type == "fairness":
            loss = fairness_loss * fairness_lambda
        elif loss_type == "fairness_kl":
            loss = fairness_loss * fairness_lambda + kl_loss
        elif loss_type == "kl_ce":
            # Reuse fact/cf logits instead of running two extra forwards.
            fact_ce_loss = compute_causal_lm_ce_from_logits(
                fact_logits, fact_input_ids, fact_attention_mask
            )
            cf_ce_loss = compute_causal_lm_ce_from_logits(
                cf_logits, cf_input_ids, cf_attention_mask
            )
            ce_loss = (fact_ce_loss + cf_ce_loss) / 2.0
            loss = kl_loss + ce_loss * fairness_lambda

        elif loss_type == "fairness_kl_ce":
            # PFairFT-KL-CE: affine fairness loss + KL divergence + CE.
            fact_ce_loss = compute_causal_lm_ce_from_logits(
                fact_logits, fact_input_ids, fact_attention_mask
            )
            cf_ce_loss = compute_causal_lm_ce_from_logits(
                cf_logits, cf_input_ids, cf_attention_mask
            )
            ce_loss = (fact_ce_loss + cf_ce_loss) / 2.0
            loss = fairness_loss * fairness_lambda + kl_loss + ce_loss

        elif loss_type == "fairness_ce":
            # Reuse fact/cf logits instead of running two extra forwards.
            fact_ce_loss = compute_causal_lm_ce_from_logits(
                fact_logits, fact_input_ids, fact_attention_mask
            )
            cf_ce_loss = compute_causal_lm_ce_from_logits(
                cf_logits, cf_input_ids, cf_attention_mask
            )
            ce_loss = (fact_ce_loss + cf_ce_loss) / 2.0
            loss = fairness_loss * fairness_lambda + ce_loss

        loss = loss / max(1, gradient_accumulation_steps)

        loss.backward()

        total_loss += loss.item() * max(1, gradient_accumulation_steps)
        total_kl_loss += kl_loss.item()
        total_fairness_loss += fairness_loss.item()

        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0 or (
            step + 1
        ) == len(dataloader):
            if head_masks is not None:
                _apply_head_grad_mask_(model, head_masks)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item() * max(1, gradient_accumulation_steps):.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "fair": f"{fairness_loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}" if ce_loss is not None else "n/a",
                "avg_loss": f"{total_loss / max(num_batches, 1):.4f}",
            }
        )

        del (
            fact_inputs,
            cf_inputs,
            fact_outputs,
            cf_outputs,
            fact_logits,
            cf_logits,
            fact_last_logits,
            cf_last_logits,
            loss,
            kl_loss,
            fairness_loss,
        )

    avg_loss = total_loss / max(num_batches, 1)
    avg_kl_loss = total_kl_loss / max(num_batches, 1)
    avg_fairness_loss = total_fairness_loss / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "fairness_loss": avg_fairness_loss,
        "kl_loss_monitor": avg_kl_loss,
    }


def save_json(data: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precision fine-tuning with fairness constraints (EXP4)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path (HuggingFace / ModelScope).",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Resume dataset JSON path.",
    )
    parser.add_argument(
        "--heads_analysis_dir",
        type=str,
        required=True,
        help="Directory containing exp2_old heads analysis results (results.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp4/finetune_output",
        help="Directory to save fine-tuned model.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of samples.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling by race.",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling.",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        default=False,
        help="Use random sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        default="",
        help="CSV path for sampling order.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Sample size when using CSV.",
    )
    parser.add_argument(
        "--resume_prompt_mode",
        type=str,
        default="category",
        choices=["summary_only", "category", "no_job_description"],
        help="Resume prompt body before the strict Yes/No instruction.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl",
        choices=["kl", "fairness", "fairness_kl", "kl_ce", "fairness_kl_ce", "fairness_ce"],
        help=(
            "Loss type: 'kl' (KL divergence), 'fairness' (affine fairness offset), "
            "'fairness_kl' (affine fairness + KL), 'kl_ce' (legacy KL + CE), "
            "'fairness_kl_ce' (affine fairness + KL + CE), 'fairness_ce' (affine fairness + CE)."
        ),
    )
    parser.add_argument(
        "--fairness_lambda",
        type=float,
        default=0.1,
        help="Weight for fairness loss (λ in L = λ * L_f). KL is monitor-only.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 precision.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 先确定 head_dim（用于验证公平锚点的维度）。
    # JetMoE 的 scattermoe Triton kernel 不能在 CPU 上跑；额外加载一次完整临时
    # GPU 模型又容易造成显存碎片。因此 JetMoE 分支跳过临时整模型 forward，
    # 后续真正训练模型加载后仍会做一次实际配置检测。
    if "jetmoe" in args.model_path.lower():
        expected_head_dim = None
        print("Skipping temporary full-model config forward for JetMoE; will validate dimensions from results.pkl and the training model.")
    else:
        temp_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if temp_tokenizer.pad_token is None:
            temp_tokenizer.pad_token = temp_tokenizer.eos_token
        temp_tokenizer.padding_side = "right"

        temp_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True
        )

        temp_adapter = get_model_adapter(temp_model, model_type="auto", model_path=args.model_path)
        print(f"Temp architecture adapter: {temp_adapter.family} ({temp_adapter.head_activation_kind})")

        temp_config = temp_adapter.get_config()
        initial_head_dim = temp_config["head_dim"]
        initial_num_heads = temp_config["num_heads"]
        print(f"Initial config from get_model_config: num_heads={initial_num_heads}, head_dim={initial_head_dim}")

        temp_buffer = {}
        test_prompt = "Test prompt for config detection"
        test_inputs = temp_tokenizer([test_prompt], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        temp_hook = temp_adapter.register_config_detection_hook(temp_buffer)
        with torch.no_grad():
            _ = temp_model(**test_inputs)
        temp_hook.remove()

        if 'head_dim' in temp_buffer and temp_buffer['head_dim'] is not None:
            detected_head_dim = temp_buffer['head_dim']
            detected_num_heads = temp_buffer['num_heads']
            print(f"Detected config from create_config_detection_hook: num_heads={detected_num_heads}, head_dim={detected_head_dim}")

            if detected_head_dim != initial_head_dim or detected_num_heads != initial_num_heads:
                print(f"Configuration mismatch detected!")
                print(f"  Initial: num_heads={initial_num_heads}, head_dim={initial_head_dim}")
                print(f"  Detected: num_heads={detected_num_heads}, head_dim={detected_head_dim}")
                print(f"Using detected values (consistent with exp2_old)")
                expected_head_dim = detected_head_dim
            else:
                expected_head_dim = initial_head_dim
        else:
            print(f"Warning: Could not detect config from hook. Using initial values.")
            expected_head_dim = initial_head_dim

        print(f"Final expected head_dim: {expected_head_dim}")
        del temp_model, temp_tokenizer, temp_adapter, test_inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. 加载公平性锚点信息（传入 head_dim 进行验证）
    fairness_info = load_fairness_anchors(args.heads_analysis_dir, expected_head_dim=expected_head_dim)
    selected_heads = fairness_info["selected_heads"]
    print(f"Will apply fairness constraints to {len(selected_heads)} heads (after dimension filtering)")

    # 2. 构建数据集
    fact_data, cf_data = build_fact_and_counterfactual_dataset(
        dataset_json_path=args.dataset_json_path,
        max_samples=args.max_samples,
        balanced=args.balanced,
        random_sampling=args.random_sampling,
        seed=args.seed,
        sample_csv_path=args.sample_csv_path if args.sample_csv_path else None,
        sample_size=args.sample_size,
    )

    dataset = PairedDataset(fact_data, cf_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # 3. 设备和精度设置
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
        print(f"Detected {num_gpus} GPUs. Will use device_map='auto'.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        use_cuda = torch.cuda.is_available()
        use_bf16_supported = use_cuda and torch.cuda.is_bf16_supported()
        if use_bf16_supported:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if use_cuda else torch.float32

    # 4. 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map = "auto" if use_multi_gpu and torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    
    if device_map is None:
        model.to(device)

    model_type = resolve_model_type(
        requested="auto",
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Resolved model_type: {model_type}")

    adapter = get_model_adapter(model, model_type=model_type, model_path=args.model_path)
    print(f"Architecture adapter: {adapter.family} ({adapter.head_activation_kind})")

    # 获取模型配置（与 exp2_old 一致：先用 get_model_config，再用 create_config_detection_hook 检测）
    config = adapter.get_config()
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    print(f"Initial model config: num_heads={num_heads}, head_dim={head_dim}")
    
    # 检测实际配置（与 exp2_old 一致）
    # 注意：在应用 LoRA 之前，模型结构是 model.model.layers
    # 在应用 LoRA 之后，模型结构是 model.base_model.model.layers
    temp_buffer = {}
    detect_hook_fn = create_config_detection_hook(temp_buffer)
    
    # 运行一个测试样本
    if len(fact_data) > 0:
        test_prompt = format_prompt_for_model(fact_data[0]["query"], model_type)
        test_inputs = tokenizer([test_prompt], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
        # 在应用 LoRA 之前，使用 adapter 检测真实 head 切分
        temp_hook = adapter.register_config_detection_hook(temp_buffer)
        with torch.no_grad():
            _ = model(**test_inputs)
        temp_hook.remove()
        
        # 更新配置（如果检测到的不同）
        if 'head_dim' in temp_buffer and temp_buffer['head_dim'] is not None:
            detected_num_heads = temp_buffer['num_heads']
            detected_head_dim = temp_buffer['head_dim']
            
            if detected_num_heads != num_heads or detected_head_dim != head_dim:
                print(f"Detected configuration mismatch!")
                print(f"  Initial: num_heads={num_heads}, head_dim={head_dim}")
                print(f"  Actual: num_heads={detected_num_heads}, head_dim={detected_head_dim}")
                print(f"Updating configuration to detected values (consistent with exp2_old)")
                num_heads = detected_num_heads
                head_dim = detected_head_dim
        else:
            print(f"Warning: Could not detect model configuration. Using initial values: num_heads={num_heads}, head_dim={head_dim}")
    
    print(f"Final model config: num_heads={num_heads}, head_dim={head_dim}")

    # 5. 准备目标 token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")
    target_ids = torch.tensor(yes_ids + no_ids, dtype=torch.long)
    print(f"Target token ids: {target_ids.tolist()}")

    # 6. 应用 LoRA（仅在选定的头上）
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=adapter.lora_target_modules(),
    )
    model = get_peft_model(model, lora_config)
    print("Applied LoRA. Trainable parameters:")
    model.print_trainable_parameters()

    # 创建 Head Masks 以实现精准微调约束
    print(f"Creating gradient masks for {len(selected_heads)} selected heads...")
    head_masks = create_head_masks(selected_heads, num_heads, head_dim, adapter.head_mask_layer_key)

    if device_map is None:
        model.to(device)
    else:
        if hasattr(model, "get_input_embeddings"):
            try:
                embed_layer = model.get_input_embeddings()
                if hasattr(embed_layer, "weight"):
                    device = embed_layer.weight.device
                else:
                    device = next(model.parameters()).device
            except:
                device = next(model.parameters()).device
        else:
            device = next(model.parameters()).device
        print(f"Using device_map='auto'. Input device: {device}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # 7. 注册激活值捕获 hooks（为所有选定的层注册，与 exp2_old 一致）
    # 使用 get_activation_hook_for_intervention，它会捕获所有头的激活值
    # 注意：应用 LoRA 后，需要找到正确的模型结构
    activation_hooks = []
    selected_layers = set(head_info["layer"] for head_info in selected_heads)
    batch_activations_buffer: Dict[int, torch.Tensor] = {}
    
    # PEFT 包装后重新创建 adapter，使模块遍历落到 base model。
    adapter = get_model_adapter(model, model_type=model_type, model_path=args.model_path)
    layers_module = adapter.get_layers()
    print(f"Found layers through adapter (count: {len(layers_module)})")
    
    for layer_idx in selected_layers:
        try:
            if layer_idx >= len(layers_module):
                raise ValueError(f"Layer {layer_idx} out of range (total layers: {len(layers_module)})")
            hook_handle = adapter.register_head_activation_hook(
                layer_idx, num_heads, head_dim, batch_activations_buffer
            )
            activation_hooks.append(hook_handle)
        except Exception as e:
            print(f"Warning: Could not register hook for layer {layer_idx}: {e}")
    
    print(f"Registered {len(activation_hooks)} activation hooks for {len(selected_layers)} layers")

    # 8. 优化器和调度器
    total_steps = max(
        1, len(dataloader) * args.num_epochs // max(1, args.gradient_accumulation_steps)
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # 9. 训练循环
    print("=" * 80)
    print("Starting precision fine-tuning with fairness constraints...")
    print(f"Fairness loss weight (λ): {args.fairness_lambda} (KL is monitor-only)")
    print("=" * 80)

    train_start_time = time.time()
    train_start_iso = datetime.now().isoformat()
    print(f"Training started at: {train_start_iso}")

    training_history: List[Dict] = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            tokenizer=tokenizer,
            max_length=args.max_length,
            model_type=model_type,
            target_ids=target_ids.to(device),
            fairness_info=fairness_info,
            fairness_lambda=args.fairness_lambda,
            activation_hooks=activation_hooks,
            batch_activations_buffer=batch_activations_buffer,
            num_heads=num_heads,
            head_dim=head_dim,
            head_masks=head_masks,
            loss_type=args.loss_type,
            resume_prompt_mode=args.resume_prompt_mode,
        )
        training_history.append({"epoch": epoch + 1, **epoch_metrics})
        print(
            f"  Loss: {epoch_metrics['loss']:.6f}, "
            f"Fairness: {epoch_metrics['fairness_loss']:.6f}, "
            f"KL(monitor): {epoch_metrics['kl_loss_monitor']:.6f}"
        )

    # 10. 移除 hooks
    for hook in activation_hooks:
        hook.remove()

    # 11. 保存模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # 12. 保存训练信息
    train_end_time = time.time()
    train_end_iso = datetime.now().isoformat()
    train_duration = train_end_time - train_start_time

    timing_info = {
        "training_start_time": train_start_iso,
        "training_end_time": train_end_iso,
        "training_duration_seconds": train_duration,
        "training_duration_minutes": train_duration / 60.0,
        "training_duration_hours": train_duration / 3600.0,
        "total_train_samples": len(dataset),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fairness_lambda": args.fairness_lambda,
        "loss_type": args.loss_type,
        "num_selected_heads": len(selected_heads),
        "training_history": training_history,
        "config": {
            "model_path": args.model_path,
            "heads_analysis_dir": args.heads_analysis_dir,
            "max_samples": args.max_samples,
            "balanced": args.balanced,
        },
    }

    timing_json_path = os.path.join(args.output_dir, "training_timing.json")
    save_json(timing_info, timing_json_path)

    print("=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Final model saved to: {final_model_dir}")
    print(f"Total duration: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
    print(f"Timing information saved to: {timing_json_path}")


if __name__ == "__main__":
    main()
