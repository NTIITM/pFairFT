#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hook 机制和干预逻辑集合
从 fairness_llm 项目中收集的所有干预方法和参数提取函数
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# 参数提取相关函数
# ============================================================================

def get_last_token_indices_safe(
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    tokenizer: Any,
) -> torch.Tensor:
    """
    计算有效的 last token index，自动跳过末尾的 EOS 或其他特殊 Token。
    这在使用 chat_template 时尤为重要，因为模板通常会在 Assistant 消息后自动添加 EOS。
    
    Args:
        input_ids: 输入 token IDs
        attention_mask: 注意力掩码
        tokenizer: tokenizer 实例
        
    Returns:
        每个样本的最后一个有效 token 索引
    """
    # 1. 基础计算：mask 的长度 - 1
    last_indices = attention_mask.sum(dim=1) - 1
    
    # 2. 检查并回退
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        curr_idx = last_indices[i].item()
        # 回退逻辑：只要当前指向的是 special token (如 EOS)，就往前退
        while curr_idx > 0:
            token_id = input_ids[i, curr_idx].item()
            if token_id == tokenizer.eos_token_id or token_id in tokenizer.all_special_ids:
                curr_idx -= 1
            else:
                break
        last_indices[i] = curr_idx
        
    return last_indices


def extract_activation_at_layer(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int = 15,
    max_length: int = 256,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    从文本中提取指定层的激活向量（最后一个token）
    
    Args:
        model: 预训练模型
        tokenizer: tokenizer
        text: 输入文本
        layer_idx: 要提取的层索引
        max_length: 最大序列长度
        device: 设备
        
    Returns:
        activation: (hidden_dim,) 的激活向量
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=True
        )
        
        # hidden_states[0] 是 embedding 层，hidden_states[1] 是 layer 0
        hidden_states = outputs.hidden_states
        target_hidden = hidden_states[layer_idx + 1]  # +1 因为第0个是embedding
        
        # 取最后一个token: (hidden_dim,)
        last_token_act = target_hidden[0, -1, :].cpu()
    
    return last_token_act


def create_activation_hook(
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    cache: Dict[int, torch.Tensor],
    output_pos: Optional[int] = None,
    extract_head_wise: bool = False,
) -> Callable:
    """
    创建用于提取激活值的 hook 函数。
    
    Args:
        layer_idx: 层索引
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        cache: 用于存储激活值的字典
        output_pos: 要提取的位置（如果为None，则提取最后一个token）
        extract_head_wise: 是否按头提取（True）还是提取整个hidden state（False）
        
    Returns:
        hook 函数
    """
    def hook(module, inputs, output):
        if extract_head_wise:
            # 提取头级别的激活值
            # inputs[0] 是输入到 o_proj 的 hidden state
            hidden_state = inputs[0]
            bsz, seqlen, _ = hidden_state.shape
            
            # Reshape to separate heads: [Batch, Seq, Heads, HeadDim]
            out_heads = hidden_state.view(bsz, seqlen, num_heads, head_dim)
            
            # 提取指定位置的激活值
            if output_pos is not None and output_pos < seqlen:
                cache[layer_idx] = out_heads[:, output_pos, :, :].detach().cpu()
            else:
                cache[layer_idx] = out_heads[:, -1, :, :].detach().cpu()
        else:
            # 提取整个 hidden state
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            
            if len(hidden_state.shape) == 3:
                # (batch, seq_len, hidden_dim)
                if output_pos is not None and output_pos < hidden_state.shape[1]:
                    last_token_act = hidden_state[:, output_pos, :].detach().cpu()
                else:
                    last_token_act = hidden_state[:, -1, :].detach().cpu()
            elif len(hidden_state.shape) == 2:
                # (batch, hidden_dim) - 已经是最后一个token
                last_token_act = hidden_state.detach().cpu()
            else:
                raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")
            
            cache[layer_idx] = last_token_act
    
    return hook


# get_model_config 已移至 util.py，这里保留导入以保持向后兼容
# 实际使用时应从 util 导入


# ============================================================================
# 干预 Hook 创建函数
# ============================================================================


def make_collect_head_output_hook(
    layer_idx: int,
    head_idx: int,
    output_pos: int,
    num_heads: int,
    head_dim: int,
    buffer: Dict[Tuple[int, int], torch.Tensor],
) -> Callable:
    """用于第一轮前向：收集指定 (layer, head) 在 output_pos 的原始 head 激活。

    Args:
        layer_idx: 层索引
        head_idx: 头索引
        output_pos: 要收集的位置
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        buffer: 缓存字典，键为 (layer, head)，值为 [head_dim] 向量
    """

    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()

        bsz, seqlen, hidden_dim = inp.shape
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
            )
        actual_head_dim = hidden_dim // num_heads
        heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)

        if output_pos < seqlen:
            v = heads_view[0, output_pos, head_idx].detach().cpu()
            buffer[(layer_idx, head_idx)] = v

        return (inp,)

    return pre_hook

def make_intervention_hook_mean_replacement(
    layer_idx: int,
    head_idx: int,
    mean_embedding: torch.Tensor,
    output_pos: Union[int, torch.Tensor],
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建均值替换干预的 hook。
    
    将敏感头的激活值替换为两个组（如 male/female, caucasian/african-american）的均值。
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        mean_embedding: 均值嵌入向量 (head_dim,)
        output_pos: 要干预的位置，可以是单个整数（所有样本相同位置）或张量（每个样本不同位置）
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    mean_embedding = mean_embedding.to(dtype=torch.float32)
    
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, hidden_dim = inp.shape
        
        # 统一根据实际 hidden_dim 推断 head_dim，避免配置和真实结构不一致
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
            )
        actual_head_dim = hidden_dim // num_heads
        heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)

        # 对齐均值向量维度到 actual_head_dim（必要时截断或零填充）
        emb = mean_embedding
        if emb.shape[0] != actual_head_dim:
            if emb.shape[0] > actual_head_dim:
                emb = emb[:actual_head_dim]
            else:
                padded = torch.zeros(actual_head_dim, dtype=emb.dtype, device=emb.device)
                padded[: emb.shape[0]] = emb
                emb = padded
        
        inp_device = inp.device
        mean_emb_device = emb.to(dtype=inp.dtype, device=inp_device)

        if isinstance(output_pos, torch.Tensor):
            # Per-sample intervention
            batch_indices = torch.arange(bsz, device=inp_device)
            pos_indices = output_pos.to(inp_device)
            # 确保索引不越界
            pos_indices = torch.clamp(pos_indices, max=seqlen - 1)
            heads_view[batch_indices, pos_indices, head_idx] = mean_emb_device
        else:
            # Single position for all samples
            if output_pos < seqlen:
                heads_view[:, output_pos, head_idx] = mean_emb_device
        
        return (inp,)
    
    return pre_hook


def make_intervention_hook_debias_projection_from_orig(
    layer_idx: int,
    head_idx: int,
    group1_embedding: torch.Tensor,
    group2_embedding: torch.Tensor,
    combined_std: Optional[torch.Tensor],
    output_pos: int,
    intervention_strength: float,
    num_heads: int,
    head_dim: int,
    orig_buffer: Dict[Tuple[int, int], torch.Tensor],
    use_std: bool = True,
) -> Callable:
    """基于第一轮缓存的原始 head 输出进行 debias 投影，并直接替换当前前向的该 head 激活。

    与 make_intervention_hook_debias_projection 的区别：
    - 方向 d 与目标位置 b 仍由 group1/group2(±std) 决定；
    - 但干预时使用 orig_buffer[(layer, head)] 作为 v_orig 做投影，并用 debias(v_orig) 替换当前前向值，
      避免多 head / 多次干预的级联影响。
    """
    # 预计算偏见消除所需的参数
    group1_embedding = group1_embedding.to(dtype=torch.float32)
    group2_embedding = group2_embedding.to(dtype=torch.float32)

    if use_std and combined_std is not None:
        combined_std = combined_std.to(dtype=torch.float32)

    diff_vector = group1_embedding - group2_embedding

    if use_std and combined_std is not None:
        combined_std_safe = torch.clamp(combined_std, min=1e-10)
        d = diff_vector / combined_std_safe
        d_norm = torch.norm(d)
        if d_norm > 1e-10:
            d = d / d_norm
            b = 0.5 * (
                torch.sum(group1_embedding * d) + torch.sum(group2_embedding * d)
            )
        else:
            d = torch.zeros_like(diff_vector)
            b = 0.0
    else:
        sigma = torch.norm(diff_vector)
        if sigma < 1e-10:
            d = torch.zeros_like(diff_vector)
            b = 0.0
        else:
            d = diff_vector / sigma
            proj_group1 = torch.sum(group1_embedding * d)
            proj_group2 = torch.sum(group2_embedding * d)
            b = 0.5 * (proj_group1 + proj_group2)

    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()

        bsz, seqlen, hidden_dim = inp.shape

        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
                )
            actual_head_dim = hidden_dim // num_heads
            heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)
            if d.shape[0] != actual_head_dim:
                raise ValueError(
                    f"Layer {layer_idx}, Head {head_idx}: d dimension mismatch"
                )
        else:
            heads_view = inp.view(bsz, seqlen, num_heads, head_dim)

        if output_pos < seqlen:
            # 使用缓存中的原始 head 输出做 debias
            if (layer_idx, head_idx) not in orig_buffer:
                raise KeyError(
                    f"orig_buffer is missing activation for (layer={layer_idx}, head={head_idx}). "
                    "Please ensure the first forward pass collected all sensitive heads."
                )

            v_orig = orig_buffer[(layer_idx, head_idx)]
            inp_device = inp.device
            v_orig_device = v_orig.to(dtype=inp.dtype, device=inp_device)

            d_device = d.to(dtype=v_orig_device.dtype, device=inp_device)
            b_device = b.to(dtype=v_orig_device.dtype, device=inp_device)

            proj_v = torch.sum(v_orig_device * d_device)
            v_debiased = v_orig_device - intervention_strength * (proj_v - b_device) * d_device

            heads_view[0, output_pos, head_idx] = v_debiased

        return (inp,)

    return pre_hook


def make_intervention_hook_debias_projection(
    layer_idx: int,
    head_idx: int,
    group1_embedding: torch.Tensor,
    group2_embedding: torch.Tensor,
    combined_std: Optional[torch.Tensor],
    output_pos: int,
    intervention_strength: float,
    num_heads: int,
    head_dim: int,
    use_std: bool = True,
) -> Callable:
    """
    创建投影去偏见干预的 hook。
    
    使用投影方法消除偏见：v' = v - α * (<v, d> - b) * d
    其中 d 是敏感属性方向向量，b 是目标位置，α 是干预强度。
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        group1_embedding: 第一组的嵌入向量（如 male 或 caucasian）
        group2_embedding: 第二组的嵌入向量（如 female 或 african-american）
        combined_std: 组合标准差（可选，用于标准化）
        output_pos: 要干预的位置
        intervention_strength: 干预强度 α
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    # 预计算偏见消除所需的参数
    group1_embedding = group1_embedding.to(dtype=torch.float32)
    group2_embedding = group2_embedding.to(dtype=torch.float32)

    if use_std and combined_std is not None:
        combined_std = combined_std.to(dtype=torch.float32)

    # 计算差异向量
    diff_vector = group1_embedding - group2_embedding

    # 计算方向向量 d 和目标位置 b
    if use_std and combined_std is not None:
        # 避免除以零
        combined_std_safe = torch.clamp(combined_std, min=1e-10)
        # 按维度标准化
        d = diff_vector / combined_std_safe
        d_norm = torch.norm(d)
        if d_norm > 1e-10:
            d = d / d_norm
            b = 0.5 * (
                torch.sum(group1_embedding * d) + torch.sum(group2_embedding * d)
            )
        else:
            d = torch.zeros_like(diff_vector)
            b = 0.0
    else:
        sigma = torch.norm(diff_vector)
        if sigma < 1e-10:
            d = torch.zeros_like(diff_vector)
            b = 0.0
        else:
            d = diff_vector / sigma
            proj_group1 = torch.sum(group1_embedding * d)
            proj_group2 = torch.sum(group2_embedding * d)
            b = 0.5 * (proj_group1 + proj_group2)
    
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, hidden_dim = inp.shape
        
        # 验证 hidden_dim 是否等于 num_heads * head_dim
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
                )
            actual_head_dim = hidden_dim // num_heads
            heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)
            if d.shape[0] != actual_head_dim:
                raise ValueError(
                    f"Layer {layer_idx}, Head {head_idx}: d dimension mismatch"
                )
        else:
            heads_view = inp.view(bsz, seqlen, num_heads, head_dim)
        
        if output_pos < seqlen:
            v = heads_view[0, output_pos, head_idx]
            
            # 偏见消除: v' = v - α * (<v, d> - b) * d
            inp_device = inp.device
            d_device = d.to(dtype=v.dtype, device=inp_device)
            b_device = b.to(dtype=v.dtype, device=inp_device)
            
            proj_v = torch.sum(v * d_device)
            v_prime = v - intervention_strength * (proj_v - b_device) * d_device
            
            heads_view[0, output_pos, head_idx] = v_prime
        
        return (inp,)
    
    return pre_hook


def make_intervention_hook_zero_value(
    layer_idx: int,
    head_idx: int,
    output_pos: int,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建零值干预的 hook：将敏感头的激活值设置为0。
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        output_pos: 要干预的位置
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, hidden_dim = inp.shape
        
        # 验证 hidden_dim 是否等于 num_heads * head_dim
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
                )
            actual_head_dim = hidden_dim // num_heads
            heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)
        else:
            heads_view = inp.view(bsz, seqlen, num_heads, head_dim)
        
        if output_pos < seqlen:
            # 将指定头在 output_pos 位置的激活值设置为0
            zero_embedding = torch.zeros_like(heads_view[0, output_pos, head_idx])
            heads_view[0, output_pos, head_idx] = zero_embedding
        
        return (inp,)
    
    return pre_hook


def make_intervention_hook_probe_projection(
    layer_idx: int,
    head_idx: int,
    probe_weight: torch.Tensor,
    probe_bias: float,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    output_pos: int,
    intervention_strength: float,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建探针投影干预的 hook。
    
    使用探针权重作为偏见方向：v' = v - α * <v, w> * w
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        probe_weight: 探针权重向量
        probe_bias: 探针偏置
        scaler_mean: 标准化均值
        scaler_std: 标准化标准差
        output_pos: 要干预的位置
        intervention_strength: 干预强度 α
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    probe_weight = probe_weight.to(dtype=torch.float32)
    w_norm = torch.norm(probe_weight)
    
    if w_norm > 1e-10:
        w = probe_weight / w_norm
    else:
        w = torch.zeros_like(probe_weight)
    
    scaler_mean = scaler_mean.to(dtype=torch.float32)
    scaler_std = scaler_std.to(dtype=torch.float32)
    
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, _ = inp.shape
        heads_view = inp.view(bsz, seqlen, num_heads, head_dim)
        
        if output_pos < seqlen:
            v = heads_view[0, output_pos, head_idx]
            
            inp_device = inp.device
            w_device = w.to(dtype=v.dtype, device=inp_device)
            
            proj_v = torch.sum(v * w_device)
            v_prime = v - intervention_strength * proj_v * w_device
            
            heads_view[0, output_pos, head_idx] = v_prime
        
        return (inp,)
    
    return pre_hook


def make_intervention_hook_probe_nullspace(
    layer_idx: int,
    head_idx: int,
    probe_weight: torch.Tensor,
    output_pos: int,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建探针零空间投影干预的 hook。
    
    将激活投影到探针权重的零空间：v' = v - <v, w> * w
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        probe_weight: 探针权重向量
        output_pos: 要干预的位置
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    probe_weight = probe_weight.to(dtype=torch.float32)
    w_norm = torch.norm(probe_weight)
    
    if w_norm > 1e-10:
        w = probe_weight / w_norm
    else:
        w = torch.zeros_like(probe_weight)
    
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, _ = inp.shape
        heads_view = inp.view(bsz, seqlen, num_heads, head_dim)
        
        if output_pos < seqlen:
            v = heads_view[0, output_pos, head_idx]
            
            inp_device = inp.device
            w_device = w.to(dtype=v.dtype, device=inp_device)
            
            proj_v = torch.sum(v * w_device)
            v_prime = v - proj_v * w_device
            
            heads_view[0, output_pos, head_idx] = v_prime
        
        return (inp,)
    
    return pre_hook


def make_intervention_hook_idea(
    layer_idx: int,
    head_idx: int,
    probe_weight: torch.Tensor,
    sigma_h: float,
    output_pos: int,
    intervention_strength: float,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建 IDEA 干预方法的 hook。
    
    IDEA 方法：将激活值沿真实性方向平移 α 倍的标准差
    公式: v' = v + α * σ_h * θ_h
    
    Args:
        layer_idx: 层索引
        head_idx: 头索引
        probe_weight: 探针权重向量（真实性方向）
        sigma_h: 标准差 σ_h
        output_pos: 要干预的位置
        intervention_strength: 干预强度 α
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    probe_weight = probe_weight.to(dtype=torch.float32)
    w_norm = torch.norm(probe_weight)
    
    if w_norm > 1e-10:
        theta_h = probe_weight / w_norm  # 真实性方向
    else:
        theta_h = torch.zeros_like(probe_weight)
    
    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()
        
        bsz, seqlen, _ = inp.shape
        heads_view = inp.view(bsz, seqlen, num_heads, head_dim)
        
        if output_pos < seqlen:
            v = heads_view[0, output_pos, head_idx]
            
            inp_device = inp.device
            theta_device = theta_h.to(dtype=v.dtype, device=inp_device)
            
            # 沿真实性方向平移 α * σ_h 的距离
            v_prime = v + intervention_strength * sigma_h * theta_device
            
            heads_view[0, output_pos, head_idx] = v_prime
        
        return (inp,)
    
    return pre_hook
    

def make_positive_direction_hook(
    layer_idx: int,
    head_idx: int,
    direction: torch.Tensor,
    output_pos: int,
    num_heads: int,
    head_dim: int,
    strength: float,
) -> Callable:
    """
    创建正向干预（方向增强）的 hook。

    在输出位置处，将该头的激活 a 修改为:
        a' = a + strength * v_dir
    其中 v_dir 通常由两组激活的平均差值给出（例如 Black - White）。
    """
    direction = direction.to(dtype=torch.float32)

    def pre_hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()

        bsz, seqlen, hidden_dim = inp.shape

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
            )
        actual_head_dim = hidden_dim // num_heads
        heads_view = inp.view(bsz, seqlen, num_heads, actual_head_dim)

        # 对齐方向向量维度到 actual_head_dim
        dir_vec = direction
        if dir_vec.shape[0] != actual_head_dim:
            if dir_vec.shape[0] > actual_head_dim:
                dir_vec = dir_vec[:actual_head_dim]
            else:
                padded = torch.zeros(actual_head_dim, dtype=dir_vec.dtype, device=dir_vec.device)
                padded[: dir_vec.shape[0]] = dir_vec
                dir_vec = padded

        if output_pos < seqlen:
            inp_device = inp.device
            dir_device = dir_vec.to(dtype=inp.dtype, device=inp_device)
            heads_view[0, output_pos, head_idx] = (
                heads_view[0, output_pos, head_idx] + strength * dir_device
            )

        return (inp,)

    return pre_hook


# ============================================================================
# 干预注册和管理函数
# ============================================================================

def register_intervention_hooks(
    model: Any,
    sensitive_heads: List[Tuple[int, int]],
    intervention_mode: str,
    intervention_params: Dict[str, Any],
    output_pos: int,
    num_heads: int,
    head_dim: int,
) -> List[Any]:
    """
    注册干预 hooks 到模型。
    
    Args:
        model: 模型实例
        sensitive_heads: 敏感头列表 [(layer_idx, head_idx), ...]
        intervention_mode: 干预模式 ("mean_replacement", "debias_projection", "zero_value", 
                                    "probe_projection", "probe_nullspace", "idea_intervention")
        intervention_params: 干预参数字典，包含所需的嵌入、权重等
        output_pos: 要干预的位置
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        hook handles 列表，用于后续移除 hooks
    """
    hooks = []
    
    for (l, h) in sensitive_heads:
        # 获取目标模块（通常是 o_proj）
        if hasattr(model.model.layers[l].self_attn, "o_proj"):
            target_module = model.model.layers[l].self_attn.o_proj
        else:
            raise ValueError(f"Cannot find o_proj in layer {l}")
        
        if intervention_mode == "mean_replacement":
            if "mean_embedding" not in intervention_params:
                raise ValueError("mean_replacement requires 'mean_embedding' in intervention_params")
            
            mean_emb = intervention_params["mean_embedding"]
            if isinstance(mean_emb, np.ndarray):
                mean_emb = torch.from_numpy(mean_emb).float()
            
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_mean_replacement(
                    l, h, mean_emb, output_pos, num_heads, head_dim
                )
            )
            hooks.append(hook)
            
        elif intervention_mode == "debias_projection":
            if "group1_embedding" not in intervention_params or "group2_embedding" not in intervention_params:
                raise ValueError("debias_projection requires 'group1_embedding' and 'group2_embedding'")
            
            group1_emb = intervention_params["group1_embedding"]
            group2_emb = intervention_params["group2_embedding"]
            combined_std = intervention_params.get("combined_std", None)
            intervention_strength = intervention_params.get("intervention_strength", 1.0)
            
            if isinstance(group1_emb, np.ndarray):
                group1_emb = torch.from_numpy(group1_emb).float()
            if isinstance(group2_emb, np.ndarray):
                group2_emb = torch.from_numpy(group2_emb).float()
            if combined_std is not None and isinstance(combined_std, np.ndarray):
                combined_std = torch.from_numpy(combined_std).float()
            
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_debias_projection(
                    l, h, group1_emb, group2_emb, combined_std, 
                    output_pos, intervention_strength, num_heads, head_dim
                )
            )
            hooks.append(hook)
            
        elif intervention_mode == "zero_value":
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_zero_value(l, h, output_pos, num_heads, head_dim)
            )
            hooks.append(hook)
            
        elif intervention_mode == "probe_projection":
            if "probe_weight" not in intervention_params:
                raise ValueError("probe_projection requires 'probe_weight' in intervention_params")
            
            probe_w = intervention_params["probe_weight"]
            probe_b = intervention_params.get("probe_bias", 0.0)
            scaler_mean = intervention_params.get("scaler_mean", torch.zeros(head_dim))
            scaler_std = intervention_params.get("scaler_std", torch.ones(head_dim))
            intervention_strength = intervention_params.get("intervention_strength", 1.0)
            
            if isinstance(probe_w, np.ndarray):
                probe_w = torch.from_numpy(probe_w).float()
            if isinstance(scaler_mean, np.ndarray):
                scaler_mean = torch.from_numpy(scaler_mean).float()
            if isinstance(scaler_std, np.ndarray):
                scaler_std = torch.from_numpy(scaler_std).float()
            
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_probe_projection(
                    l, h, probe_w, probe_b, scaler_mean, scaler_std,
                    output_pos, intervention_strength, num_heads, head_dim
                )
            )
            hooks.append(hook)
            
        elif intervention_mode == "probe_nullspace":
            if "probe_weight" not in intervention_params:
                raise ValueError("probe_nullspace requires 'probe_weight' in intervention_params")
            
            probe_w = intervention_params["probe_weight"]
            if isinstance(probe_w, np.ndarray):
                probe_w = torch.from_numpy(probe_w).float()
            
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_probe_nullspace(l, h, probe_w, output_pos, num_heads, head_dim)
            )
            hooks.append(hook)
            
        elif intervention_mode == "idea_intervention":
            if "probe_weight" not in intervention_params or "sigma_h" not in intervention_params:
                raise ValueError("idea_intervention requires 'probe_weight' and 'sigma_h' in intervention_params")
            
            probe_w = intervention_params["probe_weight"]
            sigma_h = intervention_params["sigma_h"]
            intervention_strength = intervention_params.get("intervention_strength", 1.0)
            
            if isinstance(probe_w, np.ndarray):
                probe_w = torch.from_numpy(probe_w).float()
            
            hook = target_module.register_forward_pre_hook(
                make_intervention_hook_idea(
                    l, h, probe_w, sigma_h, output_pos, intervention_strength, num_heads, head_dim
                )
            )
            hooks.append(hook)
            
        else:
            raise ValueError(f"Unknown intervention mode: {intervention_mode}")
    
    return hooks


def remove_intervention_hooks(hooks: List[Any]) -> None:
    """
    移除所有注册的干预 hooks。
    
    Args:
        hooks: hook handles 列表
    """
    for hook in hooks:
        hook.remove()


# ============================================================================
# 干预分析相关的 Hook 函数
# ============================================================================

def get_activation_hook_for_intervention(
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    batch_activations_buffer: Dict[int, torch.Tensor],
) -> Callable:
    """
    创建用于干预分析的激活值提取 hook 函数。
    
    这个函数用于在干预分析过程中提取每层的头级别激活值。
    与 create_activation_hook 不同，这个函数专门用于干预分析场景，
    将激活值存储在 batch_activations_buffer 中，形状为 [Batch, Seq, Num_Heads, Head_Dim]。
    
    Args:
        layer_idx: 层索引
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        batch_activations_buffer: 用于存储激活值的字典，键为层索引，值为激活值张量
        
    Returns:
        hook 函数
    """
    def hook(module, inputs, output):
        # inputs[0] 是 o_proj 的输入: [Batch, Seq, Hidden]
        hidden_state = inputs[0]
        bsz, seqlen, hidden_dim = hidden_state.shape
        
        # 验证配置是否正确：hidden_dim 应该等于 num_heads * head_dim
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            raise ValueError(
                f"Layer {layer_idx}: Configuration mismatch! "
                f"Expected hidden_dim = num_heads * head_dim = {num_heads} * {head_dim} = {expected_hidden}, "
                f"but actual hidden_dim = {hidden_dim}. "
                f"Please ensure correct num_heads and head_dim are passed to this hook."
            )
        
        # 还原为 Heads: [Batch, Seq, Num_Heads, Head_Dim]
        out_heads = hidden_state.view(bsz, seqlen, num_heads, head_dim)
        batch_activations_buffer[layer_idx] = out_heads
    
    return hook


# ============================================================================
# MLP 层相关 Hook（用于基于 MLP 的干预分析）
# ============================================================================

def get_mlp_last_token_activation_hook(
    layer_idx: int,
    batch_mlp_buffer: Dict[int, torch.Tensor],
) -> Callable:
    """
    创建用于收集 MLP 输出激活值的 hook。

    该 hook 挂在每一层的 `mlp` 模块上，直接缓存其前向输出：
    输出张量形状为 [Batch, Seq, Hidden]，后续可结合 last_token_indices
    在脚本中自行截取最后一个 token 的激活值。

    Args:
        layer_idx: 层索引
        batch_mlp_buffer: 字典缓存，用于保存每一层的输出

    Returns:
        forward hook 函数
    """

    def hook(module, inputs, output):
        # output: [Batch, Seq, Hidden]
        batch_mlp_buffer[layer_idx] = output.detach()

    return hook


def make_mlp_intervention_hook_mean_replacement(
    layer_idx: int,
    mean_embedding: torch.Tensor,
    output_pos: Union[int, torch.Tensor],
) -> Callable:
    """
    创建用于 MLP 层统一均值替换干预的 hook。

    干预逻辑：
    - 在最后一个 token 位置，将当前前向的 MLP 输出替换为预先计算的均值向量。

    Args:
        layer_idx: 层索引
        mean_embedding: 均值激活值，形状为 [Hidden]
        output_pos: 所有样本共用的位置，或每个样本各自的最后 token 索引

    Returns:
        forward hook 函数（挂在对应层的 `mlp` 模块上）
    """
    mean_embedding = mean_embedding.to(dtype=torch.float32)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            raise ValueError(f"Expected tensor MLP output, got {type(hidden)}")
        new_output = hidden.clone()
        bsz, seqlen, hidden_dim = new_output.shape
        
        device = new_output.device
        mean_vec = mean_embedding.to(device=device, dtype=new_output.dtype)
        
        # 维度对齐
        if mean_vec.shape[0] != hidden_dim:
            if mean_vec.shape[0] > hidden_dim:
                mean_vec = mean_vec[:hidden_dim]
            else:
                padded = torch.zeros(hidden_dim, dtype=mean_vec.dtype, device=mean_vec.device)
                padded[: mean_vec.shape[0]] = mean_vec
                mean_vec = padded

        if torch.is_tensor(output_pos):
            batch_indices = torch.arange(bsz, device=device)
            positions = output_pos.to(device).clamp(min=0, max=seqlen - 1)
            new_output[batch_indices, positions, :] = mean_vec
        elif int(output_pos) < seqlen:
            new_output[:, int(output_pos), :] = mean_vec.unsqueeze(0).expand(bsz, -1)

        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output
    
    return hook


def get_mlp_last_token_patch_hook(
    cf_batch_tensor: torch.Tensor,
    last_token_indices: torch.Tensor,
) -> Callable:
    """
    创建用于在 MLP 层进行反事实替换的 hook。

    干预逻辑：
    - 仅在最后一个 token 位置，将当前前向的 MLP 输出替换为预先缓存的
      反事实激活值（cf_batch_tensor）。

    Args:
        cf_batch_tensor: 反事实激活值，形状为 [Batch, Hidden]
        last_token_indices: 每个样本最后一个 token 的索引，形状为 [Batch]

    Returns:
        forward hook 函数（挂在对应层的 `mlp` 模块上）
    """

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            raise ValueError(f"Expected tensor MLP output, got {type(hidden)}")
        new_output = hidden.clone()
        bsz = new_output.shape[0]
        device = new_output.device

        batch_idxs = torch.arange(bsz, device=device)
        last_token_indices_on_device = last_token_indices.to(device)

        cf_vals = cf_batch_tensor.to(device).to(new_output.dtype)
        new_output[batch_idxs, last_token_indices_on_device, :] = cf_vals

        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output
    
    return hook


def get_patch_hook_modified(
    head_to_patch: int,
    fact_activation_batch: torch.Tensor,
    cf_activation_batch: torch.Tensor,
    last_token_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """
    创建修改后的干预 hook 函数。
    
    干预逻辑：
    - 先将所有头设置为事实时的激活值
    - 然后将选中的头设置为反事实的激活值
    
    这种干预方法用于计算 Modified Total Effect，即：
    在保持其他头为事实激活值的情况下，只改变选中头的激活值为反事实值。
    
    Args:
        head_to_patch: 要干预的头索引
        fact_activation_batch: 事实时的激活值，形状为 [Batch, Num_Heads, Head_Dim]
        cf_activation_batch: 反事实时的激活值，形状为 [Batch, Num_Heads, Head_Dim]
        last_token_indices: 最后一个token的索引，形状为 [Batch]
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        
    Returns:
        pre_hook 函数
    """
    def hook(module, inputs):
        inp = inputs[0]
        inp = inp.clone()

        bsz, seqlen, hidden_dim = inp.shape
        
        # 验证配置是否正确
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            raise ValueError(
                f"Configuration mismatch in patch hook! "
                f"Expected hidden_dim = num_heads * head_dim = {num_heads} * {head_dim} = {expected_hidden}, "
                f"but actual hidden_dim = {hidden_dim}. "
                f"Please ensure correct num_heads and head_dim are passed to this hook."
            )
        
        heads_view = inp.view(bsz, seqlen, num_heads, head_dim)

        inp_device = inp.device
        batch_idxs = torch.arange(bsz, device=inp_device)
        last_token_indices_on_device = last_token_indices.to(inp_device)
        
        # 先将所有头设置为事实时的激活值
        fact_activation_batch_device = fact_activation_batch.to(inp_device).to(inp.dtype)
        for h_idx in range(num_heads):
            heads_view[batch_idxs, last_token_indices_on_device, h_idx, :] = fact_activation_batch_device[:, h_idx, :]
        
        # 然后将选中的头设置为反事实的激活值
        cf_head_val = cf_activation_batch[:, head_to_patch, :].to(inp_device).to(inp.dtype)
        heads_view[batch_idxs, last_token_indices_on_device, head_to_patch, :] = cf_head_val
        
        return (inp,)

    return hook


def create_config_detection_hook(
    buffer: Dict[str, Any],
) -> Callable:
    """
    创建用于检测模型配置的 hook 函数。
    
    这个函数用于在运行时检测模型的实际配置，通过检查 o_proj 层的输入维度。
    会检测 hidden_dim，然后尝试推断 num_heads 和 head_dim。
    
    Args:
        buffer: 用于存储检测结果的字典，会设置 'hidden_dim', 'num_heads', 'head_dim' 键
        
    Returns:
        hook 函数
    """
    def hook(module, inputs, output):
        hidden_state = inputs[0]
        _, _, hidden_dim = hidden_state.shape
        buffer['hidden_dim'] = hidden_dim
        
        # 尝试推断 num_heads 和 head_dim
        # 方法1：尝试常见的 head_dim 值
        inferred_num_heads = None
        inferred_head_dim = None
        
        for candidate_head_dim in [128, 64, 256, 32, 192]:
            if hidden_dim % candidate_head_dim == 0:
                candidate_num_heads = hidden_dim // candidate_head_dim
                if candidate_num_heads > 0:
                    inferred_num_heads = candidate_num_heads
                    inferred_head_dim = candidate_head_dim
                    break
        
        # 方法2：如果方法1失败，尝试常见的 num_heads 值
        if inferred_num_heads is None:
            for candidate_num_heads in [32, 16, 8, 64, 128, 20, 40]:
                if hidden_dim % candidate_num_heads == 0:
                    inferred_num_heads = candidate_num_heads
                    inferred_head_dim = hidden_dim // candidate_num_heads
                    break
        
        if inferred_num_heads is not None:
            buffer['num_heads'] = inferred_num_heads
            buffer['head_dim'] = inferred_head_dim
        else:
            # 如果无法推断，至少记录 hidden_dim
            buffer['num_heads'] = None
            buffer['head_dim'] = None
    
    return hook
