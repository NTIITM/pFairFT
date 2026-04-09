#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集采样逻辑集合
从 fairness_llm 项目中收集的所有数据集采样函数
支持不同数据集的随机采样和平衡采样
"""

import random
from typing import Dict, List, Tuple, Optional, Any, Callable
import csv
import os

# ============================================================================
# 通用采样函数
# ============================================================================

def sample_data(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
    group_key_fn: Optional[Callable[[Dict[str, Any]], Optional[Tuple]]] = None,
) -> List[Dict[str, Any]]:
    """
    通用的数据采样函数，支持随机采样和平衡采样。
    
    采样逻辑：
    - 如果 balanced=False: 
        - random_sampling=False: 按顺序取前 max_samples 个有效样本
        - random_sampling=True: 随机采样 max_samples 个有效样本（使用seed确保可重复）
    - 如果 balanced=True: 按 group_key_fn 返回的键分组，每组采样相同数量，确保平衡
        - random_sampling=False: 每组按顺序取前 n 个样本
        - random_sampling=True: 每组随机采样 n 个样本（使用seed确保可重复）
    
    Args:
        data_records: 原始数据记录列表
        max_samples: 最大样本数
        balanced: 是否使用平衡采样
        random_sampling: 是否使用随机采样（默认False，按顺序取）
        seed: 随机种子（仅在random_sampling=True时使用，默认42）
        group_key_fn: 用于分组的函数，接受一个数据记录，返回分组键（元组）或None
        
    Returns:
        采样后的数据记录列表
    """
    if not data_records:
        return []
    
    # 设置随机种子（如果使用随机采样）
    if random_sampling:
        random.seed(seed)
    
    if not balanced:
        # 非平衡采样
        n_samples = min(len(data_records), max_samples)
        if random_sampling:
            # 随机采样
            return random.sample(data_records, n_samples)
        else:
            # 按顺序取前 max_samples 个
            return data_records[:n_samples]
    
    # 平衡采样：按 group_key_fn 分组
    if group_key_fn is None:
        raise ValueError("group_key_fn must be provided when balanced=True")
    
    # 分组
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for item in data_records:
        group_key = group_key_fn(item)
        if group_key is not None:
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
    
    if not groups:
        raise ValueError("No valid groups found. Check group_key_fn.")
    
    # 计算每个组的最小样本数
    group_counts = [len(group) for group in groups.values()]
    min_per_group = min(group_counts)
    
    # 如果指定了 max_samples，确保总样本数不超过限制
    num_groups = len(groups)
    if max_samples > 0:
        max_per_group = max_samples // num_groups
        n_each = min(min_per_group, max_per_group)
        if max_per_group == 0:
            raise ValueError(
                f"max_samples ({max_samples}) is too small. Need at least {num_groups} samples (1 per group)."
            )
    else:
        n_each = min_per_group
    
    if n_each == 0:
        raise ValueError(
            "No samples available in all groups. Cannot create balanced sample."
        )
    
    # 从每个组中采样
    balanced_samples = []
    for group_key, group_items in groups.items():
        if random_sampling:
            # 随机采样（使用seed确保可重复性）
            selected = random.sample(group_items, min(n_each, len(group_items)))
        else:
            # 按顺序取前 n 个样本
            selected = group_items[:n_each]
        balanced_samples.extend(selected)
    
    # 如果使用随机采样，可以打乱最终顺序（可选）
    if random_sampling:
        random.shuffle(balanced_samples)
    
    return balanced_samples


# ============================================================================
# COMPAS 数据集采样
# ============================================================================

def sample_compas_data(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    统一的 COMPAS 数据采样函数，确保所有脚本在相同条件下采样相同的样本。
    
    采样逻辑：
    - 如果 balanced=False: 
        - random_sampling=False: 按顺序取前 max_samples 个有效样本
        - random_sampling=True: 随机采样 max_samples 个有效样本（使用seed确保可重复）
    - 如果 balanced=True: 按 (race, label) 分组，每组采样相同数量，确保平衡
        - random_sampling=False: 每组按顺序取前 n 个样本
        - random_sampling=True: 每组随机采样 n 个样本（使用seed确保可重复）
    
    Args:
        data_records: 原始数据记录列表，每个记录应包含：
            - "filled_prompt": 填充后的 prompt
            - "original_attributes": 包含 "race" 字段的字典
            - "label": 标签字符串 ("0" 或 "1")
        max_samples: 最大样本数
        balanced: 是否使用平衡采样（按race和label分组）
        random_sampling: 是否使用随机采样（默认False，按顺序取）
        seed: 随机种子（仅在random_sampling=True时使用，默认42）
        
    Returns:
        采样后的数据记录列表，每个记录包含：
            - "id": 原始索引
            - "filled_prompt": 填充后的 prompt
            - "original_attributes": 原始属性
            - "label": 标签字符串
            - "race": 种族编码 (0=Caucasian, 1=African-American)
            - "label_int": 标签整数 (0=No, 1=Yes)
    """
    # 首先提取所有有效样本（有 filled_prompt 的）
    valid_samples = []
    for idx, item in enumerate(data_records):
        filled_prompt = item.get("filled_prompt", "")
        if not filled_prompt:
            continue
        
        original_attrs = item.get("original_attributes", {})
        race_str = original_attrs.get("race", "")
        label_str = item.get("label", "")
        
        # 提取种族和标签信息
        race = None
        if race_str == "Caucasian":
            race = 0
        elif race_str == "African-American":
            race = 1
        
        label = None
        if label_str == "1":
            label = 1
        elif label_str == "0":
            label = 0
        
        # 只保留有完整信息的样本
        if race is not None and label is not None:
            valid_samples.append({
                "id": idx,
                "filled_prompt": filled_prompt,
                "original_attributes": original_attrs,
                "label": label_str,
                "race": race,
                "label_int": label,
            })
    
    if not valid_samples:
        return []
    
    # 定义分组函数
    def group_key_fn(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """按 (race, label) 分组"""
        race = item.get("race")
        label_int = item.get("label_int")
        if race is not None and label_int is not None:
            return (race, label_int)
        return None
    
    # 使用通用采样函数
    return sample_data(
        valid_samples,
        max_samples=max_samples,
        balanced=balanced,
        random_sampling=random_sampling,
        seed=seed,
        group_key_fn=group_key_fn,
    )


# ============================================================================
# Adult/Gender 数据集采样
# ============================================================================

def sample_adult_gender_data(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
    gender_extractor: Optional[Callable[[str], Optional[int]]] = None,
    label_extractor: Optional[Callable[[str], Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Adult/Gender 数据集采样函数。
    
    按 (gender, income_label) 分组进行平衡采样。
    - gender: 0=Female, 1=Male
    - income_label: 0=<=50K, 1=>50K
    
    Args:
        data_records: 原始数据记录列表，每个记录应包含：
            - "query": 查询文本（包含性别信息）
            - "response": 响应文本（"A" 或 "B"）
        max_samples: 最大样本数
        balanced: 是否使用平衡采样（按gender和income_label分组）
        random_sampling: 是否使用随机采样
        seed: 随机种子
        gender_extractor: 从文本中提取性别的函数，返回 0=Female, 1=Male, None=无法确定
        label_extractor: 从响应中提取标签的函数，返回 0=A, 1=B, None=无效
        
    Returns:
        采样后的数据记录列表
    """
    # 默认的提取函数
    if gender_extractor is None:
        import re
        def default_gender_extractor(text: str) -> Optional[int]:
            text_lower = text.lower()
            if re.search(r'\bfemale\b', text_lower):
                return 0
            elif re.search(r'\bmale\b', text_lower):
                return 1
            return None
        gender_extractor = default_gender_extractor
    
    if label_extractor is None:
        def default_label_extractor(response: str) -> Optional[int]:
            response = response.strip().upper()
            if response == "A":
                return 0
            elif response == "B":
                return 1
            return None
        label_extractor = default_label_extractor
    
    # 提取有效样本
    valid_samples = []
    for rec in data_records:
        query = rec.get("query", "")
        response = rec.get("response", "")
        
        gender = gender_extractor(query)
        if gender is None:
            continue
        
        try:
            income_label = label_extractor(response)
            if income_label is None:
                continue
        except ValueError:
            continue
        
        valid_samples.append({
            "prompt": query,
            "income_label": income_label,
            "gender": gender,
        })
    
    if not valid_samples:
        return []
    
    # 定义分组函数：按 (gender, income_label) 分组
    def group_key_fn(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        gender = item.get("gender")
        income_label = item.get("income_label")
        if gender is not None and income_label is not None:
            return (gender, income_label)
        return None
    
    # 使用通用采样函数
    return sample_data(
        valid_samples,
        max_samples=max_samples,
        balanced=balanced,
        random_sampling=random_sampling,
        seed=seed,
        group_key_fn=group_key_fn,
    )


# ============================================================================
# Resume 数据集采样
# ============================================================================

def sample_resume_data(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
    race_extractor: Optional[Callable[[str], Optional[int]]] = None,
    gender_extractor: Optional[Callable[[str], Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Resume 数据集采样函数。
    
    按 (race, gender) 分组进行平衡采样。
    - race: 0=White, 1=Black (或 0=Caucasian, 1=African-American)
    - gender: 0=Female, 1=Male
    
    Args:
        data_records: 原始数据记录列表，每个记录应包含：
            - "query" 或 "fact_query": 查询文本（包含种族和性别信息）
            - "race": 种族字符串（可选，如果存在则直接使用）
            - "gender": 性别字符串（可选，如果存在则直接使用）
        max_samples: 最大样本数
        balanced: 是否使用平衡采样（按race和gender分组）
        random_sampling: 是否使用随机采样
        seed: 随机种子
        race_extractor: 从文本中提取种族的函数，返回 0=White/Caucasian, 1=Black/African-American, None=无法确定
        gender_extractor: 从文本中提取性别的函数，返回 0=Female, 1=Male, None=无法确定
        
    Returns:
        采样后的数据记录列表
    """
    # 默认的提取函数
    if race_extractor is None:
        import re
        def default_race_extractor(text: str) -> Optional[int]:
            text_lower = text.lower()
            if re.search(r'\b(white|caucasian)\b', text_lower):
                return 0
            elif re.search(r'\b(black|african-american)\b', text_lower):
                return 1
            return None
        race_extractor = default_race_extractor
    
    if gender_extractor is None:
        import re
        def default_gender_extractor(text: str) -> Optional[int]:
            text_lower = text.lower()
            if re.search(r'\bfemale\b', text_lower):
                return 0
            elif re.search(r'\bmale\b', text_lower):
                return 1
            return None
        gender_extractor = default_gender_extractor
    
    # 提取有效样本
    valid_samples = []
    for item in data_records:
        # 尝试从不同字段获取查询文本
        query = item.get("query") or item.get("fact_query") or item.get("prompt", "")
        if not query:
            continue
        
        # 尝试直接获取种族和性别，否则从文本中提取
        race = None
        if "race" in item:
            race_str = str(item["race"]).lower()
            if "white" in race_str or "caucasian" in race_str:
                race = 0
            elif "black" in race_str or "african" in race_str:
                race = 1
        
        if race is None:
            race = race_extractor(query)
        
        gender = None
        if "gender" in item:
            gender_str = str(item["gender"]).lower()
            if "female" in gender_str:
                gender = 0
            elif "male" in gender_str:
                gender = 1
        
        if gender is None:
            gender = gender_extractor(query)
        
        if race is None or gender is None:
            continue
        
        # 保留原始数据的所有字段
        sample = item.copy()
        sample["race"] = race
        sample["gender"] = gender
        valid_samples.append(sample)
    
    if not valid_samples:
        return []
    
    # 定义分组函数：按 (race, gender) 分组
    def group_key_fn(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        race = item.get("race")
        gender = item.get("gender")
        if race is not None and gender is not None:
            return (race, gender)
        return None
    
    # 使用通用采样函数
    return sample_data(
        valid_samples,
        max_samples=max_samples,
        balanced=balanced,
        random_sampling=random_sampling,
        seed=seed,
        group_key_fn=group_key_fn,
    )


# ============================================================================
# Resume 数据集按种族简单采样（用于干预评估）
# ============================================================================

def sample_resume_data_by_race(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Resume 数据集按种族简单采样函数（仅按 White/Black 分组）。
    
    用于干预评估场景，只需要按种族平衡采样，不需要考虑性别等其他因素。
    
    采样逻辑：
    - 如果 balanced=False: 
        - random_sampling=False: 按顺序取前 max_samples 个样本
        - random_sampling=True: 随机采样 max_samples 个样本（使用seed确保可重复）
    - 如果 balanced=True: 按 race 分组（White/Black），每组采样相同数量
        - random_sampling=False: 每组按顺序取前 n 个样本
        - random_sampling=True: 每组随机采样 n 个样本（使用seed确保可重复）
        - 如果使用随机采样，最终会打乱顺序
    
    Args:
        data_records: 原始数据记录列表，每个记录应包含：
            - "race": 种族字符串（"white" 或 "black"，不区分大小写）
        max_samples: 最大样本数
        balanced: 是否使用平衡采样（按race分组）
        random_sampling: 是否使用随机采样（默认False，按顺序取）
        seed: 随机种子（仅在random_sampling=True时使用，默认42）
        
    Returns:
        采样后的数据记录列表
    """
    if not data_records:
        return []
    
    # 设置随机种子（如果使用随机采样）
    if random_sampling:
        random.seed(seed)
    
    if not balanced:
        # 非平衡采样
        n_samples = min(len(data_records), max_samples) if max_samples > 0 else len(data_records)
        if random_sampling:
            # 随机采样
            sampled_data = random.sample(data_records, n_samples)
        else:
            # 按顺序取前 max_samples 个样本
            sampled_data = data_records[:n_samples]
        return sampled_data
    
    # 平衡采样：按种族分组
    white_samples = [item for item in data_records if item.get("race", "").lower() == "white"]
    black_samples = [item for item in data_records if item.get("race", "").lower() == "black"]
    
    # 计算每个组的样本数
    if max_samples > 0:
        n_each = max_samples // 2
    else:
        n_each = min(len(white_samples), len(black_samples))
    
    # 从每个组中采样
    if random_sampling:
        white_samples = random.sample(white_samples, min(n_each, len(white_samples)))
        black_samples = random.sample(black_samples, min(n_each, len(black_samples)))
    else:
        white_samples = white_samples[:n_each]
        black_samples = black_samples[:n_each]
    
    sampled_data = white_samples + black_samples
    
    # 如果使用随机采样，打乱最终顺序
    if random_sampling:
        random.shuffle(sampled_data)
    
    return sampled_data


# ============================================================================
# Resume 数据集：按 CSV index 顺序取样（用于干预评估 & 复用已有评估结果）
# ============================================================================

def load_samples_by_csv_indices(
    dataset: List[Dict[str, Any]],
    csv_path: str,
    sample_size: int,
) -> Tuple[List[Dict[str, Any]], List[int], List[Optional[float]]]:
    """
    根据 CSV 中的 index 列顺序，从原始 dataset 中选取样本，并同时返回
    - 使用到的原始索引列表
    - 对应行的 fact_p_yes（如果存在该列），用于 baseline 直接复用 CSV 结果。

    CSV 一般为 biased_samples_*/biased_samples_ranking.csv，包含列：
        index, fact_p_yes, cf_p_yes, fact_race, cf_race 等。
    """
    if not csv_path:
        raise ValueError("csv_path must be non-empty when using CSV-driven sampling.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    indices: List[int] = []
    fact_p_yes_list: List[Optional[float]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "index" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'index' column. Got: {reader.fieldnames}")
        has_fact_p_yes = "fact_p_yes" in reader.fieldnames
        for row in reader:
            try:
                idx = int(row["index"])
            except Exception:
                continue
            indices.append(idx)
            if has_fact_p_yes:
                val = row.get("fact_p_yes", "")
                if val is None or val == "":
                    fact_p_yes_list.append(None)
                else:
                    try:
                        fact_p_yes_list.append(float(val))
                    except Exception:
                        fact_p_yes_list.append(None)
            else:
                fact_p_yes_list.append(None)

    if sample_size and sample_size > 0:
        indices = indices[:sample_size]
        fact_p_yes_list = fact_p_yes_list[:sample_size]

    sampled: List[Dict[str, Any]] = []
    used_indices: List[int] = []
    used_fact_p_yes: List[Optional[float]] = []

    for pos, idx in enumerate(indices):
        if 0 <= idx < len(dataset):
            sampled.append(dataset[idx])
            used_indices.append(idx)
            used_fact_p_yes.append(fact_p_yes_list[pos] if pos < len(fact_p_yes_list) else None)
        else:
            continue

    return sampled, used_indices, used_fact_p_yes


# ============================================================================
# Discrim-Eval 数据集配对加载
# ============================================================================

def load_discrim_eval_pairs(dataset_path: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    加载 Discrim-Eval 配对数据集并构建配对关系。
    
    从 JSON 文件中加载数据，并根据 matched_id 字段构建唯一的配对关系。
    配对由两个样本组成，它们除了种族（race）不同外，其他属性相同。
    
    Args:
        dataset_path: 数据集 JSON 文件路径
        
    Returns:
        (data, pairs) 元组：
        - data: 所有数据记录列表
        - pairs: 配对列表，每个配对是一个 (id_a, id_b) 元组，其中 id_a < id_b
        
    Raises:
        json.JSONDecodeError: 如果 JSON 文件格式无效
    """
    import json
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print("Error reading JSON. Ensure your data processing script saved valid JSON (no NaNs).")
        raise e

    id_to_sample = {item["id"]: item for item in data}
    pairs = []
    
    # Logic to reconstruct pairs based on matched_id
    seen_pairs = set()
    for sample in data:
        mid = sample.get("matched_id")
        
        # JSON null becomes None in Python
        if mid is None:
            continue
            
        mid = int(mid)  # Ensure int
        current_id = int(sample["id"])

        # Create a sorted tuple to ensure (A, B) is same as (B, A)
        pair_key = tuple(sorted((current_id, mid)))
        
        if pair_key not in seen_pairs:
            # Check validity
            if mid in id_to_sample:
                pairs.append(pair_key)
                seen_pairs.add(pair_key)

    return data, pairs


# ============================================================================
# Discrim-Eval 数据集采样
# ============================================================================

def sample_discrim_eval_data(
    data_records: List[Dict[str, Any]],
    max_samples: int,
    balanced: bool = False,
    random_sampling: bool = False,
    seed: int = 42,
    group_by_race: bool = True,
    group_by_gender: bool = False,
) -> List[Dict[str, Any]]:
    """
    Discrim-Eval 数据集采样函数。
    
    支持按 race 或 (race, gender) 分组进行平衡采样。
    
    Args:
        data_records: 原始数据记录列表，每个记录应包含：
            - "race": 种族字符串 ("Caucasian" 或 "African-American")
            - "gender": 性别字符串（可选）
            - "prompt" 或 "filled_template": prompt 文本
        max_samples: 最大样本数
        balanced: 是否使用平衡采样
        random_sampling: 是否使用随机采样
        seed: 随机种子
        group_by_race: 是否按种族分组
        group_by_gender: 是否按性别分组（需要 group_by_race=True）
        
    Returns:
        采样后的数据记录列表
    """
    # 提取有效样本
    valid_samples = []
    for item in data_records:
        race_str = item.get("race", "")
        if not race_str:
            continue
        
        # 标准化种族值
        race = None
        race_lower = race_str.lower()
        if "caucasian" in race_lower or "white" in race_lower:
            race = 0
        elif "african" in race_lower or "black" in race_lower:
            race = 1
        
        if race is None:
            continue
        
        # 提取性别（如果启用）
        gender = None
        if group_by_gender:
            gender_str = item.get("gender", "")
            gender_lower = gender_str.lower()
            if "female" in gender_lower:
                gender = 0
            elif "male" in gender_lower:
                gender = 1
        
        sample = item.copy()
        sample["race"] = race
        if gender is not None:
            sample["gender"] = gender
        valid_samples.append(sample)
    
    if not valid_samples:
        return []
    
    # 定义分组函数
    def group_key_fn(item: Dict[str, Any]) -> Optional[Tuple]:
        if group_by_race and group_by_gender:
            race = item.get("race")
            gender = item.get("gender")
            if race is not None and gender is not None:
                return (race, gender)
        elif group_by_race:
            race = item.get("race")
            if race is not None:
                return (race,)
        else:
            # 不分组，所有样本为一组
            return (0,)
        return None
    
    # 使用通用采样函数
    return sample_data(
        valid_samples,
        max_samples=max_samples,
        balanced=balanced,
        random_sampling=random_sampling,
        seed=seed,
        group_key_fn=group_key_fn,
    )


# ============================================================================
# 辅助函数
# ============================================================================

def print_sampling_stats(
    samples: List[Dict[str, Any]],
    group_key_fn: Optional[Callable[[Dict[str, Any]], Optional[Tuple]]] = None,
    group_names: Optional[Dict[Tuple, str]] = None,
) -> None:
    """
    打印采样统计信息。
    
    Args:
        samples: 采样后的样本列表
        group_key_fn: 用于分组的函数
        group_names: 组名映射字典，用于显示友好的组名
    """
    if not samples:
        print("No samples to display.")
        return
    
    print(f"Total samples: {len(samples)}")
    
    if group_key_fn is not None:
        # 按组统计
        groups: Dict[Tuple, int] = {}
        for item in samples:
            group_key = group_key_fn(item)
            if group_key is not None:
                groups[group_key] = groups.get(group_key, 0) + 1
        
        print("Group distribution:")
        for group_key, count in sorted(groups.items()):
            if group_names and group_key in group_names:
                group_name = group_names[group_key]
            else:
                group_name = str(group_key)
            print(f"  {group_name}: {count}")


def get_balanced_sample_counts(
    data_records: List[Dict[str, Any]],
    group_key_fn: Callable[[Dict[str, Any]], Optional[Tuple]],
    max_samples: int = 0,
) -> Dict[Tuple, int]:
    """
    获取平衡采样时每个组的样本数。
    
    Args:
        data_records: 数据记录列表
        group_key_fn: 分组函数
        max_samples: 最大样本数（0表示不限制）
        
    Returns:
        每个组的样本数字典
    """
    # 分组
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for item in data_records:
        group_key = group_key_fn(item)
        if group_key is not None:
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
    
    if not groups:
        return {}
    
    # 计算每个组的最小样本数
    group_counts = [len(group) for group in groups.values()]
    min_per_group = min(group_counts)
    
    # 如果指定了 max_samples，确保总样本数不超过限制
    num_groups = len(groups)
    if max_samples > 0:
        max_per_group = max_samples // num_groups
        n_each = min(min_per_group, max_per_group)
    else:
        n_each = min_per_group
    
    return {group_key: n_each for group_key in groups.keys()}
