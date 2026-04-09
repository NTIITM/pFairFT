#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Qwen and Llama models in grid format for bias evaluation results.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def _load_stats_from_csv(csv_path: str) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    从per-sample CSV读取数据，先统计后返回。
    返回结构:
    stats[model][prompt_type][qid] = {"mean": ..., "std": ...}
    其中 prompt_type ∈ {"prompt", "debiased_prompt"}
    
    CSV格式: sample_id, matched_id, prompt_type, model, decision_question_id, p_yes
    """
    from collections import defaultdict as dd
    
    # 第一步：读取所有样本数据
    sample_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["sample_id"])
                matched_id_str = row["matched_id"].strip()
                matched_id = int(matched_id_str) if matched_id_str else None
                prompt_type = row["prompt_type"]
                model = row["model"]
                decision_question_id_str = row["decision_question_id"].strip()
                decision_question_id = int(decision_question_id_str) if decision_question_id_str else None
                p_yes = float(row["p_yes"])
                
                if decision_question_id is not None:
                    sample_data.append({
                        "sample_id": sample_id,
                        "matched_id": matched_id,
                        "prompt_type": prompt_type,
                        "model": model,
                        "decision_question_id": decision_question_id,
                        "p_yes": p_yes,
                    })
            except (ValueError, KeyError) as e:
                continue  # 跳过无效行
    
    # 第二步：按model和prompt_type分组，构建样本映射
    # data_by_model_prompt[model][prompt_type][sample_id] = {"decision_question_id": qid, "p_yes": p_yes, "matched_id": mid}
    data_by_model_prompt: Dict[str, Dict[str, Dict[int, dict]]] = dd(lambda: dd(dict))
    
    for sample in sample_data:
        model = sample["model"]
        prompt_type = sample["prompt_type"]
        sample_id = sample["sample_id"]
        data_by_model_prompt[model][prompt_type][sample_id] = {
            "decision_question_id": sample["decision_question_id"],
            "p_yes": sample["p_yes"],
            "matched_id": sample["matched_id"],
        }
    
    # 第三步：计算每个question的统计信息（基于配对样本的p_yes差值）
    stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = dd(lambda: dd(dict))
    
    for model in data_by_model_prompt:
        for prompt_type in data_by_model_prompt[model]:
            model_prompt_data = data_by_model_prompt[model][prompt_type]
            
            # 按decision_question_id分组，计算配对差值
            diffs_by_qid: Dict[int, List[float]] = dd(list)
            
            processed_pairs = set()  # 避免重复处理同一对，存储tuple(sorted([id1, id2]))
            processed_samples = set()  # 已处理的样本ID
            
            for sample_id, sample_info in model_prompt_data.items():
                if sample_id in processed_samples:
                    continue
                
                matched_id = sample_info["matched_id"]
                if matched_id is None or matched_id not in model_prompt_data:
                    continue
                
                # 确保只处理一次配对
                pair_key = tuple(sorted([sample_id, matched_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                processed_samples.add(sample_id)
                processed_samples.add(matched_id)
                
                qid = sample_info["decision_question_id"]
                matched_info = model_prompt_data[matched_id]
                
                # 确保是同一个question
                if matched_info["decision_question_id"] != qid:
                    continue
                
                p_yes_a = sample_info["p_yes"]
                p_yes_b = matched_info["p_yes"]
                
                # 计算绝对差值
                diff = abs(p_yes_a - p_yes_b)
                diffs_by_qid[qid].append(diff)
            
            # 计算每个question的mean和std
            for qid, diffs in diffs_by_qid.items():
                if len(diffs) > 0:
                    diffs_array = np.array(diffs, dtype=np.float64)
                    stats[model][prompt_type][qid] = {
                        "mean": float(diffs_array.mean()),
                        "std": float(diffs_array.std(ddof=0)) if len(diffs) > 1 else 0.0,
                    }
    
    return stats


def _plot_single_model_axes(
    ax,
    display_name: str,
    stats_prompt: Dict[int, Dict[str, float]],
    stats_debiased: Dict[int, Dict[str, float]],
):
    """
    在给定的 axes 上画单个模型:
    - x 轴: decision_question_id 按 prompt mean_gap 降序排序
    - y 轴: mean_gap, 带 std 置信带
    """
    # 按 prompt mean 降序排序
    ordered_qids: List[int] = sorted(
        stats_prompt.keys(), key=lambda q: stats_prompt[q]["mean"], reverse=True
    )

    xs = list(range(len(ordered_qids)))

    def extract(series_stats):
        means = [series_stats.get(q, {"mean": 0.0})["mean"] for q in ordered_qids]
        stds = [series_stats.get(q, {"std": 0.0})["std"] for q in ordered_qids]
        return means, stds

    means_p, stds_p = extract(stats_prompt)
    means_d, stds_d = extract(stats_debiased)

    means_p = np.array(means_p, dtype=np.float64)
    stds_p = np.array(stds_p, dtype=np.float64)
    means_d = np.array(means_d, dtype=np.float64)
    stds_d = np.array(stds_d, dtype=np.float64)

    # 画线与置信带
    line_p = ax.plot(xs, means_p, label="Original Prompt", color="tab:blue", linewidth=2)[0]
    ax.fill_between(
        xs,
        means_p - stds_p,
        means_p + stds_p,
        color="tab:blue",
        alpha=0.2,
    )

    line_d = ax.plot(xs, means_d, label="Debiased Prompt", color="tab:orange", linewidth=2)[0]
    ax.fill_between(
        xs,
        means_d - stds_d,
        means_d + stds_d,
        color="tab:orange",
        alpha=0.2,
    )

    # 模型名放在底部作为 x 轴标签
    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    # 返回用于全局 legend 的 handle
    return line_p, line_d


def plot_qwen_and_llama_grids(csv_path: str, out_qwen: str, out_llama: str):
    """
    从 paired_p_yes_stats_all_models.csv 读取结果并画两张 1x3 子图:
    - Qwen: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
    - Llama: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct
    每张图共用 legend，放在顶部。
    """
    _set_font()
    stats = _load_stats_from_csv(csv_path)

    # 模型顺序：从小到大
    qwen_models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
    llama_models = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3-8B-Instruct"]
    llama_display = {
        "Llama-3.2-1B-Instruct": "Llama-3.2 1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2 3B",
        "Meta-Llama-3-8B-Instruct": "Llama-3.2 8B",
    }

    # --- 画 Qwen ---
    # 共享 y 轴，只在最左侧子图显示 y 轴标签和刻度
    fig_qwen, axes_qwen = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_q, labels_q = None, None

    for ax, model in zip(axes_qwen, qwen_models):
        model_stats = stats.get(model, {})
        stats_prompt = model_stats.get("prompt", {})
        stats_debiased = model_stats.get("debiased_prompt", {})

        if not stats_prompt:
            ax.set_title(f"{model}\n(no data)", fontweight="bold")
            ax.set_xticks([])
            continue

        # Qwen 模型名直接使用原名放在底部
        line_p, line_d = _plot_single_model_axes(ax, model, stats_prompt, stats_debiased)
        if handles_q is None:
            handles_q = [line_p, line_d]
            labels_q = [h.get_label() for h in handles_q]

    # 全局 legend 放在更靠上的位置
    if handles_q is not None:
        fig_qwen.legend(
            handles_q,
            labels_q,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )
    # 只在最左侧子图上设置 y 轴标题和刻度
    if len(axes_qwen) > 0:
        axes_qwen[0].set_ylabel("Fairness Violation↓", fontweight="bold")
    # 调整布局与子图间距，让图更整齐
    fig_qwen.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_qwen.savefig(out_qwen, dpi=200)
    plt.close(fig_qwen)

    # --- 画 Llama ---
    # 共享 y 轴，只在最左侧子图显示 y 轴标签和刻度
    fig_llama, axes_llama = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_l, labels_l = None, None

    for ax, model in zip(axes_llama, llama_models):
        model_stats = stats.get(model, {})
        stats_prompt = model_stats.get("prompt", {})
        stats_debiased = model_stats.get("debiased_prompt", {})

        if not stats_prompt:
            ax.set_title(f"{model}\n(no data)", fontweight="bold")
            ax.set_xticks([])
            continue

        disp_name = llama_display.get(model, model)
        line_p, line_d = _plot_single_model_axes(ax, disp_name, stats_prompt, stats_debiased)
        if handles_l is None:
            handles_l = [line_p, line_d]
            labels_l = [h.get_label() for h in handles_l]

    # 全局 legend 放在更靠上的位置
    if handles_l is not None:
        fig_llama.legend(
            handles_l,
            labels_l,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )
    # 只在最左侧子图上设置 y 轴标题和刻度
    if len(axes_llama) > 0:
        axes_llama[0].set_ylabel("Fairness Violation↓", fontweight="bold")
    # 调整布局与子图间距，让图更整齐
    fig_llama.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_llama.savefig(out_llama, dpi=200)
    plt.close(fig_llama)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Qwen and Llama models in grid format.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv",
        help="Path to the per-sample details CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    out_qwen = os.path.join(args.output_dir, "qwen_models_grid.pdf")
    out_llama = os.path.join(args.output_dir, "llama_models_grid.pdf")

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        print("Please run exp.sh first to generate the per-sample details CSV file.")
        exit(1)
    
    print(f"Loading and aggregating statistics from: {args.csv_path}")

    plot_qwen_and_llama_grids(
        args.csv_path,
        out_qwen=out_qwen,
        out_llama=out_llama,
    )
    print(f"Saved Qwen plot: {out_qwen}")
    print(f"Saved Llama plot: {out_llama}")