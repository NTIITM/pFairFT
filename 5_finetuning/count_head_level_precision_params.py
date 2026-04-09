#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import json

def get_model_dims(model_path: str) -> Tuple[int, int, int]:
    """通过读取 config.json 获取模型的 d_model, num_heads, head_dim，避免库依赖冲突"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 兼容不同模型的配置字段 (Qwen, Llama 等常见字段)
    d_model = config.get("hidden_size") or config.get("d_model")
    num_heads = config.get("num_attention_heads") or config.get("n_heads")
    
    if d_model is None or num_heads is None:
        raise ValueError(f"Could not find model dimensions in {config_path}")
    
    head_dim = d_model // num_heads
    return d_model, num_heads, head_dim

def count_selected_heads(heads_analysis_dir: str) -> int:
    """统计 results.pkl 中的敏感头数量"""
    results_path = os.path.join(heads_analysis_dir, "results.pkl")
    if not os.path.exists(results_path):
        return 0
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    selected_heads = results.get("selected_heads", [])
    return len(selected_heads)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2_dir", type=str, default="/home/common1/hwluo/project/pFairFT/exp2_old")
    parser.add_argument("--llm_research_dir", type=str, default="/mnt/nfs/huggingface/LLM-Research")
    parser.add_argument("--qwen_dir", type=str, default="/mnt/nfs/huggingface/Qwen")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="/home/common1/hwluo/project/pFairFT/exp4/param_counts_head_level.csv")
    args = parser.parse_args()

    models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B"
    ]

    results = []
    print(f"{'Model':<30} | {'Heads':<6} | {'Head-level Params':<18} | {'Layer-level (Ref)':<18}")
    print("-" * 80)

    for model_name in models:
        # 确定路径
        model_path = os.path.join(args.llm_research_dir, model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(args.qwen_dir, model_name)
        
        heads_dir = os.path.join(args.exp2_dir, f"sensitive_heads_{model_name}_top100")
        
        try:
            d_model, n_heads, h_dim = get_model_dims(model_path)
            num_selected = count_selected_heads(heads_dir)
            
            # 头级参数量计算: 每个头在 q,k,v,o 都有 lora
            # params = num_heads * rank * (in + out)
            # 对于单个 head: in = d_model, out = h_dim (针对 q,k,v) 或 in = h_dim, out = d_model (针对 o)
            # 简化为: num_selected * rank * (d_model + h_dim) * 4
            head_level_params = num_selected * args.lora_rank * (d_model + h_dim) * 4
            
            # 层级参数量估算 (作为对比，假设每层 4 个模块全加 LoRA)
            # 实际上从之前的 CSV 读取更准，这里做一个简易对比
            results.append({
                "model_name": model_name,
                "num_selected_heads": num_selected,
                "d_model": d_model,
                "head_dim": h_dim,
                "head_level_params": head_level_params
            })
            
            print(f"{model_name:<30} | {num_selected:<6} | {head_level_params:<18,} | {'-':<18}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # 保存结果
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "num_selected_heads", "d_model", "head_dim", "head_level_params"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
