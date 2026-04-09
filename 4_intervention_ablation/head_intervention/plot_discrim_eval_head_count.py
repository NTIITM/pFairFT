#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 exp10 的 discrim-eval 偏见程度干预实验结果，针对 Llama-3-8B-Instruct 模型。

- 从 intervention_results_${MODEL_NAME}_discrim_eval/ 读取 results_sensitive_heads.csv 和 results_random_heads.csv。
- 选取 head_count = 0, 9, 27 的数据。
- 绘制 Mean |p_yes_a - p_yes_b|（按 question ID 平均后的结果，再总体平均）vs head_count 的折线图。
- 图中包含两条线：敏感头干预 和 随机头干预。
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 字体设置（参考 exp8 的画图风格）
def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def load_exp10_csv_data(csv_path: str) -> Dict[Tuple[int, int], float]:
    """
    从 exp10 的 CSV 文件加载数据。
    CSV 列: head_count, model_name, decision_question_id, mean_p_yes_gap, intervention_type
    返回: {(head_count, decision_question_id): mean_p_yes_gap}
    """
    data: Dict[Tuple[int, int], float] = {}
    if not os.path.exists(csv_path):
        return data

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        
        required_cols = {"head_count", "decision_question_id", "mean_p_yes_gap"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        for row in reader:
            try:
                head_count = int(row["head_count"])
                decision_question_id = int(row["decision_question_id"])
                mean_p_yes_gap = float(row["mean_p_yes_gap"])
                data[(head_count, decision_question_id)] = mean_p_yes_gap
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}, row: {row}")
                continue
    return data


def compute_overall_mean_gap_by_head_count(
    loaded_data: Dict[Tuple[int, int], float],
    target_head_counts: List[int],
) -> Dict[int, float]:
    """
    计算每个 head_count 对应的总体平均 mean_p_yes_gap（跨所有 question ID）。
    """
    mean_gaps_by_head_count: Dict[int, List[float]] = defaultdict(list)
    for (head_count, qid), mean_gap in loaded_data.items():
        if head_count in target_head_counts:
            mean_gaps_by_head_count[head_count].append(mean_gap)
    
    overall_mean_gaps: Dict[int, float] = {}
    for hc in target_head_counts:
        if mean_gaps_by_head_count[hc]:
            overall_mean_gaps[hc] = np.mean(mean_gaps_by_head_count[hc])
        else:
            overall_mean_gaps[hc] = np.nan # Use NaN if no data for this head_count
            
    return overall_mean_gaps


def plot_overall_mean_bias_by_head_count(
    model_name: str,
    sensitive_overall_mean_gaps: Dict[int, float],
    random_overall_mean_gaps: Dict[int, float],
    output_path: str,
    title: str = "Mean Bias Reduction by Head Count (Discrim-Eval)",
    xlabel: str = "Number of Intervened Heads",
    ylabel: str = "Fairness Violation↓",
):
    """
    绘制 discrim-eval 的结果：单模型，两条线（敏感头 vs 随机头）。
    X 轴为干预头数量，Y 轴为总体平均偏见程度。
    """
    _set_font()
    # 单轴图，进一步收窄宽度
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_facecolor("white")

    # 敏感头干预线
    head_counts_s = sorted(sensitive_overall_mean_gaps.keys())
    mean_values_s = [sensitive_overall_mean_gaps[hc] for hc in head_counts_s]
    ax.plot(
        head_counts_s,
        mean_values_s,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Key Heads",
        color="tab:blue",
    )

    # 随机头干预线
    head_counts_r = sorted(random_overall_mean_gaps.keys())
    mean_values_r = [random_overall_mean_gaps[hc] for hc in head_counts_r]
    ax.plot(
        head_counts_r,
        mean_values_r,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Random Heads",
        color="tab:orange",
    )

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(f"{model_name}: {title}", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    all_hc = sorted(list(set(head_counts_s) | set(head_counts_r)))
    ax.set_xticks(all_hc)
    # legend 字号略小于轴标签，避免超出图像范围
    legend_font = max(10, plt.rcParams["font.size"] - 2)
    ax.legend(loc="best", fontsize=legend_font)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_discrim_eval_by_question_id(
    model_name: str,
    sensitive_raw_data: Dict[Tuple[int, int], float],
    random_raw_data: Dict[Tuple[int, int], float],
    output_path: str,
    target_head_counts: List[int],
    title: str = "Mean Bias Reduction per Question (Discrim-Eval)",
    xlabel: str = "Decision Question ID",
    ylabel: str = "Fairness Violation↓",
    is_subplot: bool = False,
    ax: plt.Axes = None,
):
    """
    绘制 discrim-eval 的结果：单模型，X 轴为 question ID，多条线表示不同干预类别和头数量。
    """
    _set_font()

    # 1. 组合所有数据并根据 (intervention_type, head_count) 分组
    combined_data: Dict[Tuple[str, int], Dict[int, float]] = defaultdict(dict) # (type, hc) -> qid -> mean_gap
    all_qids = set()

    for (hc, qid), mean_gap in sensitive_raw_data.items():
        if hc in target_head_counts:
            combined_data[("sensitive", hc)][qid] = mean_gap
            all_qids.add(qid)

    for (hc, qid), mean_gap in random_raw_data.items():
        if hc in target_head_counts:
            combined_data[("random", hc)][qid] = mean_gap
            all_qids.add(qid)
    
    if not all_qids:
        print(f"No data found for plotting by question ID for {model_name}.")
        return

    # 2. 确定 X 轴顺序：根据 Baseline (head_count=0, sensitive) 的 bias 降序排列
    baseline_qid_gaps: Dict[int, float] = combined_data.get(("sensitive", 0), {})
    ordered_qids = sorted(
        list(all_qids),
        key=lambda qid: baseline_qid_gaps.get(qid, -1.0), 
        reverse=True,
    )
    xs = np.arange(len(ordered_qids))

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        standalone = True
    
    ax.set_facecolor("white")

    # 3. 绘制多条线
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_head_counts))) 

    intervention_styles = {
        "sensitive": {"marker": 's', "linestyle": '-'}, 
        "random": {"marker": 'o', "linestyle": '--'},   
    }

    legend_handles = []
    for i, hc in enumerate(target_head_counts):
        current_color = colors[i] 

        # Sensitive line
        key_s = ("sensitive", hc)
        if key_s in combined_data:
            y_values_s = [combined_data[key_s].get(qid, np.nan) for qid in ordered_qids]
            h_s, = ax.plot(
                xs, y_values_s,
                label=f"Key Heads (H={hc})",
                color=current_color,
                marker=intervention_styles["sensitive"]["marker"],
                linestyle=intervention_styles["sensitive"]["linestyle"],
                linewidth=2,
                markersize=6,
            )
            legend_handles.append(h_s)

        # Random line
        key_r = ("random", hc)
        if key_r in combined_data:
            y_values_r = [combined_data[key_r].get(qid, np.nan) for qid in ordered_qids]
            h_r, = ax.plot(
                xs, y_values_r,
                label=f"Random     (H={hc})",
                color=current_color,
                marker=intervention_styles["random"]["marker"],
                linestyle=intervention_styles["random"]["linestyle"],
                linewidth=2,
                markersize=6,
            )
            legend_handles.append(h_r)

    ax.set_xticks([])
    display_mapping = {
        "Meta-Llama-3-8B-Instruct": "Llama 8B",
        "Llama-3.2-3B-Instruct": "Llama 3B",
        "Llama-3.2-1B-Instruct": "Llama 1B",
        "Qwen3-8B": "Qwen 8B",
        "Qwen3-4B": "Qwen 4B",
        "Qwen3-1.7B": "Qwen 1.7B"
    }
    display_name = display_mapping.get(model_name, model_name)
    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    if legend_handles and (standalone or is_subplot):
        ncol = min(2, len(legend_handles))
        legend_font = max(8, plt.rcParams["font.size"] - 4) if is_subplot else max(10, plt.rcParams["font.size"] - 2)
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=legend_font,
            ncol=ncol,
        )

    if standalone:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        plt.close()


def _extract_head_counts(
    sensitive_raw_data: Dict[Tuple[int, int], float],
    random_raw_data: Dict[Tuple[int, int], float],
) -> List[int]:
    head_counts = set()
    for (hc, _qid) in sensitive_raw_data.keys():
        head_counts.add(hc)
    for (hc, _qid) in random_raw_data.keys():
        head_counts.add(hc)
    return sorted(head_counts)


def plot_grouped_3x1(
    results_root: str,
    model_groups: List[List[str]],
    group_names: List[str],
    target_head_counts: List[int],
    output_prefix: str,
):
    """
    画两张 3x1 的图，纵向排列。
    """
    _set_font()
    plt.rcParams["font.size"] = 12 # 稍微减小字体以适应 3x1

    for group, gname in zip(model_groups, group_names):
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        
        for idx, model_name in enumerate(group):
            ax = axes[idx]
            res_dir = os.path.join(results_root, f"intervention_results_{model_name}_discrim_eval")
            s_csv = os.path.join(res_dir, "results_sensitive_heads.csv")
            r_csv = os.path.join(res_dir, "results_random_heads.csv")
            
            s_data = load_exp10_csv_data(s_csv)
            r_data = load_exp10_csv_data(r_csv)
            
            if not s_data and not r_data:
                ax.text(0.5, 0.5, f"No data for {model_name}", ha='center', va='center')
                continue
                
            # grouped_3x1 模式：每个模型使用自己 CSV 里实际出现的 head_count 集合，避免“只画出部分点”
            auto_head_counts = _extract_head_counts(s_data, r_data)
            use_head_counts = auto_head_counts if auto_head_counts else target_head_counts

            plot_discrim_eval_by_question_id(
                model_name=model_name,
                sensitive_raw_data=s_data,
                random_raw_data=r_data,
                output_path="",
                target_head_counts=use_head_counts,
                is_subplot=True,
                ax=ax
            )
            # 只有第一个子图显示 legend 吗？用户说参考 exp9，exp9 是每个子图都有。
            # 这里保持每个子图都有 legend 但字号调小。
            
        plt.tight_layout()
        out_path = f"{output_prefix}_{gname}_3x1.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved grouped plot to: {out_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot discrim-eval intervention results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp10",
        help="Directory containing results or root for grouped plots.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Name of the model (not required for grouped plots).",
    )
    parser.add_argument(
        "--target_head_counts",
        type=int,
        nargs='+',
        default=[0, 9, 18, 27, 36, 45],
        help="Specific head counts to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path or prefix for grouped plots.",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="by_question_id",
        choices=["overall_mean", "by_question_id", "grouped_3x1"],
        help="Plot type.",
    )
    args = parser.parse_args()

    if args.plot_type == "grouped_3x1":
        qwen_models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
        llama_models = [
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
            "Meta-Llama-3-8B-Instruct",
        ]
        plot_grouped_3x1(
            results_root=args.results_dir,
            model_groups=[llama_models, qwen_models],
            group_names=["llama", "qwen"],
            target_head_counts=args.target_head_counts,
            output_prefix=args.output or os.path.join(args.results_dir, "mean_bias_by_group")
        )
        return

    # Original single model logic
    sensitive_csv_path = os.path.join(args.results_dir, "results_sensitive_heads.csv")
    random_csv_path = os.path.join(args.results_dir, "results_random_heads.csv")

    sensitive_raw_data = load_exp10_csv_data(sensitive_csv_path)
    random_raw_data = load_exp10_csv_data(random_csv_path)
    
    if not sensitive_raw_data and not random_raw_data:
        print(f"No data found in {args.results_dir}. Exiting.")
        return

    output_path = args.output

    if args.plot_type == "overall_mean":
        sensitive_overall_mean_gaps = compute_overall_mean_gap_by_head_count(
            sensitive_raw_data, args.target_head_counts
        )
        random_overall_mean_gaps = compute_overall_mean_gap_by_head_count(
            random_raw_data, args.target_head_counts
        )

        if not output_path:
            output_path = os.path.join(args.results_dir, "mean_bias_discrim_eval_overall_mean.png")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        plot_overall_mean_bias_by_head_count(
            model_name=args.model_name,
            sensitive_overall_mean_gaps=sensitive_overall_mean_gaps,
            random_overall_mean_gaps=random_overall_mean_gaps,
            output_path=output_path,
        )
    elif args.plot_type == "by_question_id":
        if not output_path:
            output_path = os.path.join(args.results_dir, "mean_bias_discrim_eval_by_question_id.pdf")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        plot_discrim_eval_by_question_id(
            model_name=args.model_name,
            sensitive_raw_data=sensitive_raw_data,
            random_raw_data=random_raw_data,
            output_path=output_path,
            target_head_counts=args.target_head_counts # Use the same argument for consistency
        )
    print("Done.")


if __name__ == "__main__":
    main()
