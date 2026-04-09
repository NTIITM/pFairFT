#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resume 数据集分析工具函数模块
包含种族翻转、提取等工具函数
"""

import json
import os
import pickle
import re
from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import rankdata

# 导入项目中的相关函数
from prompt import format_prompt_for_model, resolve_model_type
from probability import get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES, log_top3_warning, p_yes_from_logits_stable
from hook import get_last_token_indices_safe


# 模型名称映射字典
# key: 模型目录名（basename），value: 友好的显示名称
MODEL_NAME_MAP = {
    # Qwen 模型
    "Qwen3-1.7B": "Qwen 1.7B",
    "Qwen3-4B": "Qwen 4B",
    "Qwen3-8B": "Qwen 8B",
    # Llama 模型
    "Llama-3.2-1B-Instruct": "Llama 1B",
    "Llama-3.2-3B-Instruct": "Llama 3B",
    "Meta-Llama-3-8B-Instruct": "Llama 8B",
    # DeepSeek 模型
    "DeepSeek-V2-Lite-Chat": "DeepSeek-V2-Lite",
}

# 模型目录到显示名称的完整映射（包含路径信息）
# 用于从完整路径或basename映射到友好名称
def get_model_display_name(model_path_or_name: str) -> str:
    """
    获取模型的友好显示名称。
    
    Args:
        model_path_or_name: 模型路径或模型名称（basename）
        
    Returns:
        友好的显示名称，如果未找到则返回原始名称
    """
    # 如果是路径，提取basename
    model_name = os.path.basename(os.path.normpath(model_path_or_name))
    return MODEL_NAME_MAP.get(model_name, model_name)


def get_input_device(model: Any, requested_device: str) -> torch.device:
    """
    Get the device where model input embeddings are located.
    
    获取模型输入嵌入所在的设备。
    
    Args:
        model: 模型实例
        requested_device: 请求的设备（字符串，如 "cuda" 或 "cpu"）
        
    Returns:
        实际使用的设备
    """
    device = torch.device(requested_device)
    use_auto_device = hasattr(model, "hf_device_map") and model.hf_device_map is not None
    if use_auto_device:
        try:
            if hasattr(model, "get_input_embeddings"):
                emb = model.get_input_embeddings()
                if hasattr(emb, "weight"):
                    device = emb.weight.device
                else:
                    device = next(model.parameters()).device
            else:
                device = next(model.parameters()).device
        except Exception:
            device = next(model.parameters()).device
    else:
        model.to(device)
    return device


def get_model_config(model: Any) -> Dict[str, Any]:
    """
    提取模型配置信息，支持多种模型架构。
    
    这个函数会尝试多种方法获取模型配置，包括：
    - 从 model.config 直接读取
    - 从模型层结构推断
    - 支持 Qwen、Llama 等常见架构
    
    Args:
        model: 模型实例
        
    Returns:
        包含模型配置信息的字典，包括：
        - num_layers: 层数
        - hidden_size: 隐藏层维度
        - num_heads: 注意力头数量
        - head_dim: 每个头的维度
    """
    config = {}
    
    # 获取层数
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        config["num_layers"] = len(model.model.layers)
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        config["num_layers"] = len(model.transformer.h)
        layers = model.transformer.h
    elif hasattr(model, "layers"):
        config["num_layers"] = len(model.layers)
        layers = model.layers
    else:
        raise ValueError("Cannot find model layers. Please check model architecture.")
    
    # 优先从 model.config 获取配置
    if hasattr(model, "config"):
        model_config = model.config
        
        # 获取 hidden_size
        if hasattr(model_config, "hidden_size"):
            config["hidden_size"] = model_config.hidden_size
        elif hasattr(model_config, "d_model"):
            config["hidden_size"] = model_config.d_model
        else:
            # 尝试从第一层推断
            if hasattr(layers[0], "self_attn"):
                attn = layers[0].self_attn
                if hasattr(attn, "q_proj"):
                    config["hidden_size"] = attn.q_proj.in_features
                elif hasattr(attn, "embed_dim"):
                    config["hidden_size"] = attn.embed_dim
                else:
                    raise ValueError("Cannot infer hidden_size.")
            else:
                raise ValueError("Cannot infer hidden_size.")
        
        # 获取 num_heads
        # 对于 Qwen3 等模型，优先使用 num_key_value_heads（如果存在）
        # 但通常 num_attention_heads 才是真正的头数
        if hasattr(model_config, "num_attention_heads"):
            config["num_heads"] = model_config.num_attention_heads
        elif hasattr(model_config, "num_heads"):
            config["num_heads"] = model_config.num_heads
        elif hasattr(model_config, "num_key_value_heads"):
            # 某些模型（如 Qwen）使用 num_key_value_heads，但这可能不是真正的头数
            # 尝试从 hidden_size 和 head_dim 推断
            if hasattr(model_config, "head_dim"):
                inferred_num_heads = config["hidden_size"] // model_config.head_dim
                if inferred_num_heads > 0 and config["hidden_size"] % model_config.head_dim == 0:
                    config["num_heads"] = inferred_num_heads
                else:
                    config["num_heads"] = model_config.num_key_value_heads
            else:
                config["num_heads"] = model_config.num_key_value_heads
        else:
            # 尝试从层中推断
            example_attn = layers[0].self_attn if hasattr(layers[0], "self_attn") else None
            if example_attn is not None:
                if hasattr(example_attn, "num_heads"):
                    config["num_heads"] = example_attn.num_heads
                elif hasattr(example_attn, "num_attention_heads"):
                    config["num_heads"] = example_attn.num_attention_heads
                elif hasattr(example_attn, "head_dim"):
                    config["head_dim"] = example_attn.head_dim
                    config["num_heads"] = config["hidden_size"] // config["head_dim"]
                else:
                    # 默认推断：尝试常见的 head_dim 值
                    for head_dim_candidate in [128, 64, 256]:
                        if config["hidden_size"] % head_dim_candidate == 0:
                            config["num_heads"] = config["hidden_size"] // head_dim_candidate
                            config["head_dim"] = head_dim_candidate
                            break
                    else:
                        raise ValueError(f"Cannot infer num_heads from hidden_size={config['hidden_size']}")
            else:
                raise ValueError("Cannot infer num_heads.")
        
        # 获取 head_dim（如果还没有）
        if "head_dim" not in config:
            # MLA 模型（如 DeepSeek-V2）使用 v_head_dim 而非 hidden_size // num_heads
            if hasattr(model_config, "v_head_dim"):
                config["head_dim"] = model_config.v_head_dim
                config["is_mla"] = True
            elif hasattr(model_config, "head_dim"):
                config["head_dim"] = model_config.head_dim
            elif hasattr(model_config, "d_head"):
                config["head_dim"] = model_config.d_head
            else:
                # 计算 head_dim
                if config["hidden_size"] % config["num_heads"] != 0:
                    raise ValueError(
                        f"hidden_size ({config['hidden_size']}) is not divisible by num_heads ({config['num_heads']})"
                    )
                config["head_dim"] = config["hidden_size"] // config["num_heads"]
    else:
        # 如果没有 config，尝试从层结构推断
        if hasattr(layers[0], "self_attn"):
            attn = layers[0].self_attn
            if hasattr(attn, "q_proj"):
                config["hidden_size"] = attn.q_proj.in_features
            elif hasattr(attn, "embed_dim"):
                config["hidden_size"] = attn.embed_dim
            else:
                raise ValueError("Cannot infer hidden_size.")
            
            if hasattr(attn, "num_heads"):
                config["num_heads"] = attn.num_heads
            elif hasattr(attn, "num_attention_heads"):
                config["num_heads"] = attn.num_attention_heads
            elif hasattr(attn, "head_dim"):
                config["head_dim"] = attn.head_dim
                config["num_heads"] = config["hidden_size"] // config["head_dim"]
            else:
                # 默认推断
                config["num_heads"] = config["hidden_size"] // 128
                config["head_dim"] = 128
        
        # 计算 head_dim
        if "head_dim" not in config:
            if config["hidden_size"] % config["num_heads"] != 0:
                raise ValueError(
                    f"hidden_size ({config['hidden_size']}) is not divisible by num_heads ({config['num_heads']})"
                )
            config["head_dim"] = config["hidden_size"] // config["num_heads"]
    
    return config


def extract_race_from_query(query: str) -> Optional[str]:
    """
    从 query 文本中提取种族信息。
    
    Args:
        query: 包含种族信息的文本
        
    Returns:
        提取到的种族字符串 ("White" 或 "Black")，如果未找到则返回 None
    """
    pattern = r'\b(White|Black|white|black)\b'
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        race = match.group(1)
        # 统一返回标准格式（首字母大写）
        race_lower = race.lower()
        if race_lower == "white":
            return "White"
        elif race_lower == "black":
            return "Black"
    return None


# 种族映射常量
_RACE_MAP = {
    "white": "Black",
    "black": "White",
    "White": "Black",
    "Black": "White"
}


def opposite_race(race_value: str) -> str:
    """
    返回相反的种族值。
    
    Args:
        race_value: 当前种族值 ("White" 或 "Black")
        
    Returns:
        相反的种族值
    """
    if not isinstance(race_value, str):
        return race_value
    
    race_lower = race_value.strip()
    return _RACE_MAP.get(race_lower, race_value)


def flip_race_in_text(text: str) -> str:
    """
    在文本中翻转种族信息（White <-> Black）。
    
    Args:
        text: 包含种族信息的文本
        
    Returns:
        翻转种族后的文本
    """
    if not text:
        return text
    
    def replace_race(match):
        race = match.group(0)
        opposite = _RACE_MAP.get(race)
        if opposite:
            # 保持原始大小写格式
            if race[0].isupper():
                return opposite
            else:
                return opposite.lower()
        return race
    
    pattern = r'\b(White|Black|white|black)\b'
    return re.sub(pattern, replace_race, text, flags=re.IGNORECASE)


def create_counterfactual_by_race(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    为给定的数据项创建反事实数据（翻转种族）。
    
    Args:
        data_item: 包含 "query" 或 "summary" 字段的数据项
        
    Returns:
        包含反事实数据的新字典，包含 "query" 字段（用于后续分析）
    """
    # 优先使用 "query"，如果没有则从 summary 构建
    original_query = data_item.get("query", "")
    
    if not original_query:
        # 如果没有 query，尝试从 summary 和 category 构建
        summary = data_item.get("summary", "")
        category = data_item.get("category", "")
    
    # 翻转种族
    counterfactual_query = flip_race_in_text(original_query)
    
    # 创建反事实数据项
    counterfactual_item = {
        "query": counterfactual_query,
    }
    
    # 保留其他字段
    for key in ["ID", "category", "gender"]:
        if key in data_item:
            counterfactual_item[key] = data_item[key]
    
    # 更新 race
    original_race = data_item.get("race", "")
    if original_race:
        counterfactual_item["race"] = opposite_race(original_race)
    
    return counterfactual_item


def compute_elbow_point(sorted_scores: np.ndarray) -> Tuple[int, float]:
    """
    使用肘点方法找到最佳阈值。
    
    Args:
        sorted_scores: 已排序的分数数组（从大到小）
        
    Returns:
        (elbow_idx, elbow_score) 元组
    """
    n = len(sorted_scores)
    if n <= 2:
        return (0, sorted_scores[0] if n > 0 else 0.0)
    
    # 使用最远点方法
    coords = np.vstack((np.arange(n), sorted_scores)).T
    p1, p2 = coords[0], coords[-1]
    vec = p2 - p1
    d = np.cross(vec, coords - p1) / np.linalg.norm(vec)
    elbow_idx = np.argmax(np.abs(d))
    elbow_score = sorted_scores[elbow_idx]
    
    return (elbow_idx, elbow_score)


def compute_elbow_point_by_acceleration(sorted_scores: np.ndarray) -> Tuple[int, float]:
    """
    使用“二阶导数”最大化（加速度法）选取肘点。
    寻找斜率变化最快的地方（急转弯）。
    
    Args:
        sorted_scores: 已排序的分数数组（从大到小）
        
    Returns:
        (elbow_idx, elbow_score) 元组
    """
    n = len(sorted_scores)
    if n <= 3:
        return (0, sorted_scores[0] if n > 0 else 0.0)
    
    # 计算一阶差分
    first_diff = np.diff(sorted_scores)
    
    # 计算二阶差分 (加速度)
    second_diff = np.diff(first_diff)
    
    # 寻找二阶差分绝对值最大的点
    # 注意 np.diff 后的长度变化：n -> n-1 -> n-2
    # 我们关注的是曲线顶部的“急转弯”，通常在靠前的位置
    # 为了防止后期小波动的干扰，我们可以限制搜索范围
    search_range = min(100, len(second_diff))
    acc_abs = np.abs(second_diff[:search_range])
    
    # 在这个范围内寻找最大加速度点
    elbow_idx = np.argmax(acc_abs) + 1 # +1 是因为二阶差分对应的是中间那个点
    
    elbow_score = sorted_scores[elbow_idx]
    
    return (elbow_idx, elbow_score)


def compute_rank_array(heatmap_kl: np.ndarray) -> np.ndarray:
    """
    计算排名数组。
    
    Args:
        heatmap_kl: KL 散度矩阵
        
    Returns:
        排名数组，形状与 heatmap_kl 相同
    """
    flat_kl_values = heatmap_kl.flatten()
    ranks = rankdata(-flat_kl_values, method='ordinal')  # 负号表示从大到小排序
    return ranks.reshape(heatmap_kl.shape)


# ============================================================================
# 基于 result.pkl heatmap 的头选取与排序（用于干预评估）
# ============================================================================

def _normalize_emb_keys(emb: Dict) -> Dict[Tuple[int, int], Any]:
    """将 white_emb/black_emb 的 key 统一为 (int, int) 元组。"""
    out = {}
    for k, v in emb.items():
        if isinstance(k, (tuple, list)) and len(k) >= 2:
            out[(int(k[0]), int(k[1]))] = v
    return out


def get_sensitive_heads_sorted_by_heatmap(
    results_data: Dict[str, Any],
    min_score: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    从 results.pkl 的 heatmap 属性对头排序并选取敏感头（KL 越大越敏感，取 >= min_score 并按 KL 降序）。

    Args:
        results_data: 来自 results.pkl 的字典，需包含 "heatmap", "white_emb", "black_emb"；
            可选 "elbow_score" 作为 min_score 的默认值。
        min_score: 仅保留 heatmap[l,h] >= min_score 的头；若为 None 则使用 results_data 中的
            "elbow_score"；若二者皆无则保留所有有 embedding 的头。

    Returns:
        敏感头列表 [(layer, head), ...]，按 heatmap 值降序排列（仅包含在 white_emb 与 black_emb 中均存在的头）。
    """
    heatmap = results_data.get("heatmap")
    if heatmap is None:
        raise ValueError("results_data must contain 'heatmap'.")
    white_emb = _normalize_emb_keys(results_data.get("white_emb", {}))
    black_emb = _normalize_emb_keys(results_data.get("black_emb", {}))
    threshold = min_score

    num_layers, num_heads = heatmap.shape
    candidates = []
    for l in range(num_layers):
        for h in range(num_heads):
            if (l, h) not in white_emb or (l, h) not in black_emb:
                continue
            val = heatmap[l, h]
            if not np.isfinite(val):
                continue
            if threshold is not None and val < threshold:
                continue
            candidates.append((l, h, float(val)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(l, h) for l, h, _ in candidates]


def get_non_sensitive_heads_from_results(
    results_data: Dict[str, Any],
    elbow_score: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    从 results.pkl 的 heatmap 得到「非敏感头」：有合法 embedding 且 heatmap 值严格小于肘点阈值的头，
    用于随机干预时从非敏感头中采样。

    Args:
        results_data: 来自 results.pkl 的字典，需包含 "heatmap", "white_emb", "black_emb"。
        elbow_score: 敏感头阈值，heatmap[l,h] >= elbow_score 视为敏感；若为 None 则使用
            results_data["elbow_score"]；若仍无则视为无敏感头，返回所有有 embedding 的头。

    Returns:
        非敏感头列表 [(layer, head), ...]，顺序未定义。
    """
    heatmap = results_data.get("heatmap")
    if heatmap is None:
        raise ValueError("results_data must contain 'heatmap'.")
    white_emb = _normalize_emb_keys(results_data.get("white_emb", {}))
    black_emb = _normalize_emb_keys(results_data.get("black_emb", {}))
    threshold = elbow_score
    if threshold is None:
        threshold = results_data.get("elbow_score")

    out = []
    num_layers, num_heads = heatmap.shape
    for l in range(num_layers):
        for h in range(num_heads):
            if (l, h) not in white_emb or (l, h) not in black_emb:
                continue
            val = heatmap[l, h]
            if not np.isfinite(val):
                continue
            if threshold is not None and val >= threshold:
                continue
            out.append((l, h))
    return out


def load_intervention_results(pkl_path: str) -> Dict[str, Any]:
    """
    加载 analyze_race_sensitive_heads 产出的 results.pkl。

    Args:
        pkl_path: results.pkl 文件路径。

    Returns:
        包含 heatmap, white_emb, black_emb, elbow_score 等键的字典。
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Discrim-Eval 数据集相关工具函数
# ============================================================================

def load_discrim_eval_target_samples(
    dataset_path: str, 
    target_qids: List[int]
) -> Tuple[List[dict], Dict[int, List[Tuple[int, int]]]]:
    """
    加载discrim-eval数据集中指定decision_question_ids的样本并构建配对。
    
    Args:
        dataset_path: discrim-eval数据集JSON文件路径
        target_qids: 目标question ID列表
        
    Returns:
        (filtered_samples, pairs_by_qid) 元组
        - filtered_samples: 过滤后的样本列表
        - pairs_by_qid: 按question ID分组的配对列表
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter samples
    filtered_samples = [s for s in data if s["decision_question_id"] in target_qids]
    print(f"Found {len(filtered_samples)} samples with question IDs {target_qids}")
    
    # Build id_to_sample map
    id_to_sample = {item["id"]: item for item in filtered_samples}
    
    # Construct pairs by question ID
    pairs_by_qid = defaultdict(list)
    seen_pairs = set()
    
    for sample in filtered_samples:
        mid = sample.get("matched_id")
        if mid is None:
            continue
        
        mid = int(mid)
        current_id = int(sample["id"])
        qid = sample["decision_question_id"]
        
        # Create sorted pair key
        pair_key = tuple(sorted((current_id, mid)))
        
        if pair_key not in seen_pairs and mid in id_to_sample:
            pairs_by_qid[qid].append(pair_key)
            seen_pairs.add(pair_key)
    
    return filtered_samples, pairs_by_qid


def compute_discrim_eval_paired_differences(
    pairs: List[Tuple[int, int]],
    id_to_sample: Dict[int, dict],
    p_yes_map: Dict[int, float],
) -> List[float]:
    """
    计算discrim-eval数据集中配对样本的p(yes)绝对差值。
    
    Args:
        pairs: 配对列表，每个元素是(sample_id_a, sample_id_b)元组
        id_to_sample: sample_id到样本字典的映射
        p_yes_map: sample_id到p(yes)值的映射
        
    Returns:
        p(yes)绝对差值列表
    """
    diffs = []
    for id_a, id_b in pairs:
        if id_a not in id_to_sample or id_b not in id_to_sample:
            continue
        if id_a not in p_yes_map or id_b not in p_yes_map:
            continue
        
        diff = abs(p_yes_map[id_a] - p_yes_map[id_b])
        diffs.append(diff)
    
    return diffs


def load_discrim_eval_csv_results(
    csv_path: str, 
    model_name: str, 
    target_qids: List[int]
) -> Dict[int, Dict[str, List[float]]]:
    """
    从CSV文件加载discrim-eval数据集的已有结果（Original和Debiased）。
    
    Args:
        csv_path: per_sample_details_all_models.csv文件路径
        model_name: 模型名称
        target_qids: 目标question ID列表
        
    Returns:
        results_by_qid[qid][condition] = list of differences
        condition可以是"Original"或"Debiased"
    """
    print(f"Loading existing results from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 过滤模型和question IDs
    df_filtered = df[
        (df["model"] == model_name) & 
        (df["decision_question_id"].isin(target_qids))
    ].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found for model {model_name} and question IDs {target_qids}")
        return defaultdict(lambda: defaultdict(list))
    
    # 构建id到样本的映射（需要加载数据集来获取配对信息）
    results_by_qid = defaultdict(lambda: defaultdict(list))
    
    # 按question_id和prompt_type分组
    for qid in target_qids:
        qid_data = df_filtered[df_filtered["decision_question_id"] == qid].copy()
        
        for prompt_type in ["prompt", "debiased_prompt"]:
            condition = "Original" if prompt_type == "prompt" else "Debiased"
            prompt_data = qid_data[qid_data["prompt_type"] == prompt_type].copy()
            
            if len(prompt_data) == 0:
                continue
            
            # 构建sample_id到p_yes的映射
            p_yes_map = dict(zip(prompt_data["sample_id"], prompt_data["p_yes"]))
            
            # 找到所有配对并计算差值
            pairs_processed = set()
            diffs = []
            
            for _, row in prompt_data.iterrows():
                sample_id = int(row["sample_id"])
                matched_id = row["matched_id"]
                
                if pd.isna(matched_id) or matched_id == "":
                    continue
                
                try:
                    matched_id = int(matched_id)
                except (ValueError, TypeError):
                    continue
                
                # 避免重复处理同一对
                pair_key = tuple(sorted([sample_id, matched_id]))
                if pair_key in pairs_processed:
                    continue
                pairs_processed.add(pair_key)
                
                # 获取两个样本的p_yes值
                p_yes_a = p_yes_map.get(sample_id)
                p_yes_b = p_yes_map.get(matched_id)
                
                if p_yes_a is not None and p_yes_b is not None:
                    diff = abs(p_yes_a - p_yes_b)
                    diffs.append(diff)
            
            if diffs:
                results_by_qid[qid][condition] = diffs
    
    return results_by_qid


# ============================================================================
# 概率计算相关工具函数
# ============================================================================

def compute_p_yes_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    model_type: str,
    yes_ids: List[int],
    no_ids: List[int],
    sample_idx: Optional[int] = None,
    show_warnings: bool = True,
    prefix: str = "",
) -> float:
    """
    计算单个 prompt 的 p(yes) 概率，并检查前3个token是否包含yes/no。
    
    Args:
        model: 模型实例
        tokenizer: tokenizer 实例
        prompt: prompt 字符串（已添加 yes/no 指令）
        device: 设备
        model_type: 模型类型
        yes_ids: yes token IDs
        no_ids: no token IDs
        sample_idx: 样本索引（用于警告信息，可选）
        show_warnings: 是否显示警告（默认True）
        prefix: 警告信息前缀（可选）
        
    Returns:
        p(yes) 概率
    """
    formatted_prompt = format_prompt_for_model(prompt, model_type)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
    output_pos = int(last_token_indices[0].item())
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_row = outputs.logits[0, output_pos, :].float()
        
        # 检查前3个token是否包含yes/no（如果启用警告）
        if show_warnings:
            log_top3_warning(
                logits_row,
                tokenizer=tokenizer,
                yes_ids=yes_ids,
                no_ids=no_ids,
                sample_idx=sample_idx if sample_idx is not None else 0,
                prefix=prefix,
                show_warnings=show_warnings,
            )
        
        p_yes = p_yes_from_logits_stable(logits_row, yes_ids=yes_ids, no_ids=no_ids)
    
    del input_ids, outputs, logits_row
    return p_yes


def compute_p_yes_from_logits_with_warning(
    logits_row: torch.Tensor,
    tokenizer: Any,
    yes_ids: List[int],
    no_ids: List[int],
    sample_idx: Optional[int] = None,
    show_warnings: bool = True,
    prefix: str = "",
) -> float:
    """
    从 logits 计算 p(yes) 概率，并检查前3个token是否包含yes/no。
    用于需要外部控制模型前向传播的场景（如干预）。
    
    Args:
        logits_row: 最后一个token位置的logits (vocab_size,)
        tokenizer: tokenizer 实例
        yes_ids: yes token IDs
        no_ids: no token IDs
        sample_idx: 样本索引（用于警告信息，可选）
        show_warnings: 是否显示警告（默认True）
        prefix: 警告信息前缀（可选）
        
    Returns:
        p(yes) 概率
    """
    # 检查前3个token是否包含yes/no（如果启用警告）
    if show_warnings:
        log_top3_warning(
            logits_row,
            tokenizer=tokenizer,
            yes_ids=yes_ids,
            no_ids=no_ids,
            sample_idx=sample_idx if sample_idx is not None else 0,
            prefix=prefix,
            show_warnings=show_warnings,
        )
    
    p_yes = p_yes_from_logits_stable(logits_row, yes_ids=yes_ids, no_ids=no_ids)
    return p_yes
