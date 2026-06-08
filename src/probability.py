#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概率计算和警告过程集合
从 fairness_llm 项目中收集的所有概率计算和警告相关函数
"""

from typing import List, Tuple, Optional, Any
import torch
from tqdm import tqdm


# ============================================================================
# 候选词常量
# ============================================================================

YES_CANDIDATES = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO_CANDIDATES = ["no", "No", "NO", " no", " No", " NO"]
A_CANDIDATES = ["A", "a", " A", " a", "A:", "a:", "A.", "a.", "A)", "a)"]
B_CANDIDATES = ["B", "b", " B", " b", "B:", "b:", "B.", "b.", "B)", "b)"]

RACE_WHITE_CANDIDATES = ["white", "White", "WHITE", " white", " White", " WHITE"]
RACE_BLACK_CANDIDATES = ["black", "Black", "BLACK", " black", " Black", " BLACK"]


# ============================================================================
# Token ID 提取函数
# ============================================================================

def get_target_token_ids(tokenizer: Any, candidates: List[str]) -> List[int]:
    """
    Get the FIRST token ID for each candidate word.
    
    获取每个候选词的第一个 token ID。
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        candidates: List of candidate strings
        
    Returns:
        List of unique first token IDs for each candidate
    """
    target_ids = set()
    for word in candidates:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            target_ids.add(ids[0])
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    tokenizer_class = tokenizer.__class__.__name__.lower()
    if "jetmoe" in tokenizer_name or "jetmoe" in tokenizer_class:
        normalized_candidates = {word.strip() for word in candidates if word.strip()}
        for tid in range(len(tokenizer)):
            decoded = tokenizer.decode([tid]).strip()
            if decoded in normalized_candidates:
                target_ids.add(tid)
    return list(target_ids)


# ============================================================================
# 概率计算函数
# ============================================================================

def p_yes_from_logits_stable(
    logits_row: torch.Tensor,
    yes_ids: List[int],
    no_ids: List[int],
) -> float:
    """
    Stable p(yes) computed from logits using logsumexp.
    
    使用 logsumexp 从 logits 稳定地计算 p(yes) 概率。
    这种方法可以避免数值溢出问题。
    
    Args:
        logits_row: Logits tensor for a single token position (vocab_size,)
        yes_ids: List of token IDs for "yes" candidates
        no_ids: List of token IDs for "no" candidates
        
    Returns:
        Probability of "yes" (float between 0 and 1)
    """
    overlap = set(yes_ids) & set(no_ids)
    if overlap:
        yes_ids = [tid for tid in yes_ids if tid not in overlap]
        no_ids = [tid for tid in no_ids if tid not in overlap]
    if len(yes_ids) == 0 or len(no_ids) == 0:
        raise ValueError("yes_ids/no_ids must be non-empty.")
    if logits_row.dtype != torch.float32:
        logits_row = logits_row.float()

    yes_t = torch.tensor(yes_ids, device=logits_row.device, dtype=torch.long)
    no_t = torch.tensor(no_ids, device=logits_row.device, dtype=torch.long)

    yes_logits = logits_row.index_select(0, yes_t)
    no_logits = logits_row.index_select(0, no_t)

    score_yes = torch.logsumexp(yes_logits, dim=0)
    score_no = torch.logsumexp(no_logits, dim=0)
    p_yes = torch.sigmoid(score_yes - score_no)
    return float(p_yes.item())


def p_ab_from_logits_stable(
    logits_row: torch.Tensor,
    a_ids: List[int],
    b_ids: List[int],
) -> Tuple[float, float]:
    """
    Stable p(A) and p(B) computed from logits using logsumexp.
    
    使用 logsumexp 从 logits 稳定地计算 p(A) 和 p(B) 概率。
    参考 p_yes_from_logits_stable 的实现。
    
    Args:
        logits_row: Logits tensor for a single token position (vocab_size,)
        a_ids: List of token IDs for "A" candidates
        b_ids: List of token IDs for "B" candidates
        
    Returns:
        Tuple of (p(A), p(B)) probabilities (both floats between 0 and 1)
    """
    if len(a_ids) == 0 or len(b_ids) == 0:
        raise ValueError("a_ids/b_ids must be non-empty.")
    if logits_row.dtype != torch.float32:
        logits_row = logits_row.float()

    a_t = torch.tensor(a_ids, device=logits_row.device, dtype=torch.long)
    b_t = torch.tensor(b_ids, device=logits_row.device, dtype=torch.long)

    a_logits = logits_row.index_select(0, a_t)
    b_logits = logits_row.index_select(0, b_t)

    score_a = torch.logsumexp(a_logits, dim=0)
    score_b = torch.logsumexp(b_logits, dim=0)
    
    # Use sigmoid to compute p(A) and p(B) relative to each other
    # p(A) = sigmoid(score_a - score_b)
    # p(B) = sigmoid(score_b - score_a) = 1 - p(A)
    p_a = torch.sigmoid(score_a - score_b)
    p_b = 1.0 - p_a
    
    return float(p_a.item()), float(p_b.item())


def p_from_logits_stable(
    logits_row: torch.Tensor,
    positive_ids: List[int],
    negative_ids: List[int],
) -> float:
    """
    Generic stable probability computation from logits using logsumexp.
    
    通用的从 logits 稳定计算概率的函数，适用于任意正负类别。
    
    Args:
        logits_row: Logits tensor for a single token position (vocab_size,)
        positive_ids: List of token IDs for positive class candidates
        negative_ids: List of token IDs for negative class candidates
        
    Returns:
        Probability of positive class (float between 0 and 1)
    """
    if len(positive_ids) == 0 or len(negative_ids) == 0:
        raise ValueError("positive_ids/negative_ids must be non-empty.")
    if logits_row.dtype != torch.float32:
        logits_row = logits_row.float()

    pos_t = torch.tensor(positive_ids, device=logits_row.device, dtype=torch.long)
    neg_t = torch.tensor(negative_ids, device=logits_row.device, dtype=torch.long)

    pos_logits = logits_row.index_select(0, pos_t)
    neg_logits = logits_row.index_select(0, neg_t)

    score_pos = torch.logsumexp(pos_logits, dim=0)
    score_neg = torch.logsumexp(neg_logits, dim=0)
    p_pos = torch.sigmoid(score_pos - score_neg)
    return float(p_pos.item())


# ============================================================================
# 警告函数
# ============================================================================

def log_top3_warning(
    logits_row: torch.Tensor,
    tokenizer: Any,
    yes_ids: List[int],
    no_ids: List[int],
    sample_idx: int,
    prefix: str = "",
    show_warnings: bool = True,
) -> bool:
    """
    检查单个样本最后一个位置 logits 的 top-3 中是否包含 yes/no token。
    如果 top-3 中不包含 yes/no candidate，则打印 warning。
    
    Args:
        logits_row: 单个位置的 logits，形状为 (vocab_size,)
        tokenizer: HF tokenizer，用于 decode token
        yes_ids/no_ids: yes/no 的 token id 列表
        sample_idx: 样本索引（用于打印）
        prefix: 可选前缀（例如 "Fact"、"CF"、"Intervened" 等）
        show_warnings: 是否打印 warning
        
    Returns:
        has_yes_no: bool，表示 top-3 中是否出现 yes/no token
    """
    if not show_warnings:
        return False

    top3_values, top3_indices = torch.topk(logits_row, k=3)
    top3_token_ids = top3_indices.tolist()
    yes_no_token_ids = set(yes_ids + no_ids)
    has_yes_no = any(tid in yes_no_token_ids for tid in top3_token_ids)
    
    if not has_yes_no:
        top3_tokens = [tokenizer.decode([tid]) for tid in top3_token_ids]
        top3_probs = torch.softmax(top3_values, dim=0).tolist()
        prefix_str = f"[{prefix}] " if prefix else ""
        print(f"\nWARNING {prefix_str}Sample {sample_idx}: Top-3 predictions at output position do NOT contain yes/no candidates!")
        print(f"  Top-3 tokens: {list(zip(top3_tokens, [f'{p:.4f}' for p in top3_probs]))}")
        print(f"  Expected yes/no token IDs: yes={yes_ids}, no={no_ids}")
        
        # Also check top-1 specifically
        top1_token = tokenizer.decode([top3_token_ids[0]])
        top1_prob = top3_probs[0]
        print(f"  Top-1 prediction: '{top1_token}' (prob={top1_prob:.4f})")

    return has_yes_no


def log_topk_warning(
    logits_row: torch.Tensor,
    tokenizer: Any,
    target_ids: List[int],
    sample_idx: int,
    k: int = 3,
    prefix: str = "",
    show_warnings: bool = True,
    target_name: str = "target",
) -> bool:
    """
    通用的 top-k 警告函数，检查 top-k 预测中是否包含目标 token。
    
    Args:
        logits_row: 单个位置的 logits，形状为 (vocab_size,)
        tokenizer: HF tokenizer，用于 decode token
        target_ids: 目标 token id 列表
        sample_idx: 样本索引（用于打印）
        k: 检查 top-k 个预测（默认 3）
        prefix: 可选前缀
        show_warnings: 是否打印 warning
        target_name: 目标名称（用于打印，如 "yes/no", "A/B" 等）
        
    Returns:
        has_target: bool，表示 top-k 中是否出现目标 token
    """
    if not show_warnings:
        return False

    topk_values, topk_indices = torch.topk(logits_row, k=k)
    topk_token_ids = topk_indices.tolist()
    target_token_ids = set(target_ids)
    has_target = any(tid in target_token_ids for tid in topk_token_ids)
    
    if not has_target:
        topk_tokens = [tokenizer.decode([tid]) for tid in topk_token_ids]
        topk_probs = torch.softmax(topk_values, dim=0).tolist()
        prefix_str = f"[{prefix}] " if prefix else ""
        print(f"\nWARNING {prefix_str}Sample {sample_idx}: Top-{k} predictions at output position do NOT contain {target_name} candidates!")
        print(f"  Top-{k} tokens: {list(zip(topk_tokens, [f'{p:.4f}' for p in topk_probs]))}")
        print(f"  Expected {target_name} token IDs: {target_ids}")
        
        # Also check top-1 specifically
        top1_token = tokenizer.decode([topk_token_ids[0]])
        top1_prob = topk_probs[0]
        print(f"  Top-1 prediction: '{top1_token}' (prob={top1_prob:.4f})")

    return has_target


# ============================================================================
# Prompt 格式化检查函数
# ============================================================================

def _is_prompt_already_formatted(prompt: str) -> bool:
    """
    检查 prompt 是否已经被 model format 过。
    
    通过检查是否包含多个模型特定的特殊字符来判断。
    
    Args:
        prompt: 待检查的 prompt 字符串
        
    Returns:
        如果已经格式化则返回 True，否则返回 False
    """
    # Llama 格式的特殊字符
    llama_markers = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ]
    
    # Qwen 格式的特殊字符
    qwen_markers = [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>"
    ]

    olmoe_markers = [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]
    
    # 检查 Llama 格式：如果同时包含多个 Llama 标记，则认为已格式化
    llama_count = sum(1 for marker in llama_markers if marker in prompt)
    if llama_count >= 2:
        return True
    
    # 检查 Qwen 格式：如果同时包含多个 Qwen 标记，则认为已格式化
    qwen_count = sum(1 for marker in qwen_markers if marker in prompt)
    if qwen_count >= 2:
        return True

    olmoe_count = sum(1 for marker in olmoe_markers if marker in prompt)
    if olmoe_count >= 2:
        return True
    
    return False


# ============================================================================
# 批量计算函数
# ============================================================================

@torch.inference_mode()
def compute_p_yes_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    device: str,
    yes_ids: List[int],
    no_ids: List[int],
    model_type: str = "qwen",
    desc: str = "Computing",
    show_warnings: bool = True,
    format_prompt_fn: Optional[Any] = None,
) -> List[float]:
    """
    批量计算 p(yes) 概率。
    
    对每个 prompt(不需要再model format prompt)，计算模型输出 "yes" 的概率。
    
    Args:
        model: 模型实例
        tokenizer: tokenizer 实例
        prompts: prompt 列表
        device: 设备（字符串，如 "cuda" 或 "cpu"）
        yes_ids: "yes" 候选 token IDs
        no_ids: "no" 候选 token IDs
        model_type: 模型类型 ("llama" 或 "qwen")，用于格式化 prompt
        desc: 进度条描述
        show_warnings: 是否显示警告
        format_prompt_fn: 可选的 prompt 格式化函数，如果提供则使用此函数而不是 model_type
        
    Returns:
        p(yes) 概率列表
    """
    results = []
    
    for idx, user_prompt in enumerate(tqdm(prompts, desc=desc)):
        # 检查 prompt 是否已经被格式化
        if _is_prompt_already_formatted(user_prompt):
            import warnings
            warnings.warn(
                f"Prompt at index {idx} appears to be already formatted (contains multiple model-specific special tokens). "
                f"This may cause double-formatting. Consider passing unformatted prompts.",
                UserWarning,
                stacklevel=2
            )
        
        # 格式化 prompt
        if format_prompt_fn is not None:
            full_prompt = format_prompt_fn(user_prompt)
        else:
            # 使用默认的格式化方法（需要从 prompt.py 导入）
            try:
                from prompt import format_prompt_for_model
                full_prompt = format_prompt_for_model(user_prompt, model_type)
            except ImportError:
                # 如果没有 prompt.py，直接使用原始 prompt
                full_prompt = user_prompt
        
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_row = out.logits[0, -1, :].float()

        # 统一使用辅助函数检查 top-3 是否包含 yes/no
        log_top3_warning(
            logits_row,
            tokenizer=tokenizer,
            yes_ids=yes_ids,
            no_ids=no_ids,
            sample_idx=idx,
            prefix=desc,
            show_warnings=show_warnings,
        )

        p_yes = p_yes_from_logits_stable(logits_row, yes_ids=yes_ids, no_ids=no_ids)
        results.append(p_yes)
        
        del input_ids, out, logits_row
    
    return results


@torch.inference_mode()
def compute_p_ab_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    device: str,
    a_ids: List[int],
    b_ids: List[int],
    model_type: str = "qwen",
    desc: str = "Computing",
    show_warnings: bool = True,
    format_prompt_fn: Optional[Any] = None,
) -> List[Tuple[float, float]]:
    """
    批量计算 p(A) 和 p(B) 概率。
    
    对每个 prompt，计算模型输出 "A" 和 "B" 的概率。
    
    Args:
        model: 模型实例
        tokenizer: tokenizer 实例
        prompts: prompt 列表
        device: 设备（字符串，如 "cuda" 或 "cpu"）
        a_ids: "A" 候选 token IDs
        b_ids: "B" 候选 token IDs
        model_type: 模型类型 ("llama" 或 "qwen")，用于格式化 prompt
        desc: 进度条描述
        show_warnings: 是否显示警告
        format_prompt_fn: 可选的 prompt 格式化函数
        
    Returns:
        (p(A), p(B)) 概率元组列表
    """
    results = []
    
    for idx, user_prompt in enumerate(tqdm(prompts, desc=desc)):
        # 检查 prompt 是否已经被格式化
        if _is_prompt_already_formatted(user_prompt):
            import warnings
            warnings.warn(
                f"Prompt at index {idx} appears to be already formatted (contains multiple model-specific special tokens). "
                f"This may cause double-formatting. Consider passing unformatted prompts.",
                UserWarning,
                stacklevel=2
            )
        
        # 格式化 prompt
        if format_prompt_fn is not None:
            full_prompt = format_prompt_fn(user_prompt)
        else:
            try:
                from prompt import format_prompt_for_model
                full_prompt = format_prompt_for_model(user_prompt, model_type)
            except ImportError:
                full_prompt = user_prompt
        
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_row = out.logits[0, -1, :].float()

        # 检查 top-3 是否包含 A/B
        if show_warnings:
            log_topk_warning(
                logits_row,
                tokenizer=tokenizer,
                target_ids=a_ids + b_ids,
                sample_idx=idx,
                k=3,
                prefix=desc,
                show_warnings=show_warnings,
                target_name="A/B",
            )

        p_a, p_b = p_ab_from_logits_stable(logits_row, a_ids=a_ids, b_ids=b_ids)
        results.append((p_a, p_b))
        
        del input_ids, out, logits_row
    
    return results


# ============================================================================
# 辅助函数
# ============================================================================

def get_yes_no_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """
    便捷函数：获取 yes/no token IDs。
    
    Args:
        tokenizer: tokenizer 实例
        
    Returns:
        (yes_ids, no_ids) 元组
    """
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    return yes_ids, no_ids


def get_ab_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """
    便捷函数：获取 A/B token IDs。
    
    Args:
        tokenizer: tokenizer 实例
        
    Returns:
        (a_ids, b_ids) 元组
    """
    a_ids = get_target_token_ids(tokenizer, A_CANDIDATES)
    b_ids = get_target_token_ids(tokenizer, B_CANDIDATES)
    return a_ids, b_ids


def validate_token_ids(
    tokenizer: Any,
    yes_ids: List[int],
    no_ids: Optional[List[int]] = None,
    a_ids: Optional[List[int]] = None,
    b_ids: Optional[List[int]] = None,
) -> None:
    """
    验证 token IDs 是否有效，如果无效则抛出异常。
    
    Args:
        tokenizer: tokenizer 实例
        yes_ids: yes token IDs（必须提供）
        no_ids: no token IDs（可选）
        a_ids: A token IDs（可选）
        b_ids: B token IDs（可选）
        
    Raises:
        ValueError: 如果任何 token IDs 列表为空
    """
    if len(yes_ids) == 0:
        error_msg = "ERROR: Could not find token IDs for 'yes' candidates.\n"
        error_msg += f"  Yes candidates: {YES_CANDIDATES}\n"
        if no_ids is not None:
            error_msg += f"  No candidates: {NO_CANDIDATES}\n"
        raise ValueError(error_msg)
    
    if no_ids is not None and len(no_ids) == 0:
        error_msg = "ERROR: Could not find token IDs for 'no' candidates.\n"
        error_msg += f"  Yes candidates: {YES_CANDIDATES}\n"
        error_msg += f"  No candidates: {NO_CANDIDATES}\n"
        raise ValueError(error_msg)
    
    if a_ids is not None and len(a_ids) == 0:
        error_msg = "ERROR: Could not find token IDs for 'A' candidates.\n"
        error_msg += f"  A candidates: {A_CANDIDATES}\n"
        raise ValueError(error_msg)
    
    if b_ids is not None and len(b_ids) == 0:
        error_msg = "ERROR: Could not find token IDs for 'B' candidates.\n"
        error_msg += f"  B candidates: {B_CANDIDATES}\n"
        raise ValueError(error_msg)
