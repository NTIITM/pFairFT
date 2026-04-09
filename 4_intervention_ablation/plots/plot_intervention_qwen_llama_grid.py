#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Qwen and Llama models in grid format for discrim-eval bias evaluation
BEFORE and AFTER negative head-level intervention.

联合使用：
- Baseline per-sample CSV（exp1/per_sample_details_all_models.csv）
- Negative intervention per-sample CSV（exp8/per_sample_intervention_negative_all_models.csv）

画法参考 exp1/plot_qwen_llama_grid.py：
- x 轴：按 baseline 的 mean gap 降序排列的 decision_question_id
- y 轴：mean gap（配对 p(yes) 绝对差），比较：
    - Original Prompt baseline (蓝色)
    - Original Prompt + Negative Intervention (红色)
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def _load_baseline_stats(
    csv_path: str,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    从 baseline per-sample CSV 读取统计信息。

    输入 CSV（exp1/per_sample_details_all_models.csv）格式：
        sample_id, matched_id, prompt_type, model, decision_question_id, p_yes

    使用 prompt_type ∈ {"prompt", "debiased_prompt"} 的条目。

    返回:
        stats_baseline[model][prompt_type][qid] = {"mean": ..., "std": ...}
    其中 mean/std 基于配对样本 p_yes 差的绝对值。
    """
    from collections import defaultdict as dd

    # 读取所有样本
    sample_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["sample_id"])
                matched_id_str = row["matched_id"].strip()
                matched_id = int(matched_id_str) if matched_id_str else None
                prompt_type = row["prompt_type"].strip()
                if prompt_type not in {"prompt", "debiased_prompt"}:
                    continue
                model = row["model"]
                decision_question_id_str = row["decision_question_id"].strip()
                decision_question_id = (
                    int(decision_question_id_str) if decision_question_id_str else None
                )
                p_yes = float(row["p_yes"])

                if decision_question_id is not None:
                    sample_data.append(
                        {
                            "sample_id": sample_id,
                            "matched_id": matched_id,
                            "prompt_type": prompt_type,
                            "model": model,
                            "decision_question_id": decision_question_id,
                            "p_yes": p_yes,
                        }
                    )
            except (ValueError, KeyError):
                continue  # 跳过无效行

    # data_by_model_prompt[model][prompt_type][sample_id] = {...}
    data_by_model_prompt: Dict[str, Dict[str, Dict[int, dict]]] = dd(
        lambda: dd(dict)
    )
    for sample in sample_data:
        model = sample["model"]
        prompt_type = sample["prompt_type"]
        sample_id = sample["sample_id"]
        data_by_model_prompt[model][prompt_type][sample_id] = {
            "decision_question_id": sample["decision_question_id"],
            "p_yes": sample["p_yes"],
            "matched_id": sample["matched_id"],
        }

    # 计算每个 question 的配对差值统计
    stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = dd(
        lambda: dd(dict)
    )
    for model, prompt_dict in data_by_model_prompt.items():
        for prompt_type, model_data in prompt_dict.items():
            from collections import defaultdict as dd2

            diffs_by_qid: Dict[int, List[float]] = dd2(list)
            processed_pairs = set()
            processed_samples = set()

            for sample_id, sample_info in model_data.items():
                if sample_id in processed_samples:
                    continue

                matched_id = sample_info["matched_id"]
                if matched_id is None or matched_id not in model_data:
                    continue

                pair_key = tuple(sorted([sample_id, matched_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                processed_samples.add(sample_id)
                processed_samples.add(matched_id)

                qid = sample_info["decision_question_id"]
                matched_info = model_data[matched_id]

                if matched_info["decision_question_id"] != qid:
                    continue

                p_yes_a = sample_info["p_yes"]
                p_yes_b = matched_info["p_yes"]
                diff = abs(p_yes_a - p_yes_b)
                diffs_by_qid[qid].append(diff)

            for qid, diffs in diffs_by_qid.items():
                if len(diffs) == 0:
                    continue
                arr = np.array(diffs, dtype=np.float64)
                stats[model][prompt_type][qid] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)) if len(diffs) > 1 else 0.0,
                }

    return stats


def _load_intervention_stats(
    csv_path: str,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    从干预后的 per-sample CSV 读取统计信息。

    输入 CSV（exp8/per_sample_intervention_*_all_models.csv）格式：
        sample_id, matched_id, model, decision_question_id, p_yes, intervention_type

    返回:
        stats_intervention[model][qid] = {"mean": ..., "std": ...}
    基于同一 question 下配对样本 p_yes 差的绝对值（与 baseline 逻辑对齐）。
    """
    from collections import defaultdict as dd

    sample_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["sample_id"])
                matched_id_str = row.get("matched_id", "").strip()
                matched_id = int(matched_id_str) if matched_id_str else None
                model = row["model"]
                decision_question_id_str = row["decision_question_id"].strip()
                decision_question_id = (
                    int(decision_question_id_str) if decision_question_id_str else None
                )
                p_yes = float(row["p_yes"])

                if decision_question_id is not None:
                    sample_data.append(
                        {
                            "sample_id": sample_id,
                            "matched_id": matched_id,
                            "model": model,
                            "decision_question_id": decision_question_id,
                            "p_yes": p_yes,
                        }
                    )
            except (ValueError, KeyError):
                continue

    # data_by_model[model][sample_id] = {...}
    data_by_model: Dict[str, Dict[int, dict]] = dd(dict)
    for s in sample_data:
        model = s["model"]
        sample_id = s["sample_id"]
        data_by_model[model][sample_id] = {
            "decision_question_id": s["decision_question_id"],
            "p_yes": s["p_yes"],
            "matched_id": s["matched_id"],
        }

    # 使用 matched_id 显式构造配对，与 baseline 统计方式一致
    stats: Dict[str, Dict[int, Dict[str, float]]] = dd(dict)
    for model, model_data in data_by_model.items():
        from collections import defaultdict as dd2

        diffs_by_qid: Dict[int, List[float]] = dd2(list)
        processed_pairs = set()
        processed_samples = set()

        for sample_id, sample_info in model_data.items():
            if sample_id in processed_samples:
                continue

            matched_id = sample_info["matched_id"]
            if matched_id is None or matched_id not in model_data:
                continue

            pair_key = tuple(sorted([sample_id, matched_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            processed_samples.add(sample_id)
            processed_samples.add(matched_id)

            qid = sample_info["decision_question_id"]
            matched_info = model_data[matched_id]

            if matched_info["decision_question_id"] != qid:
                continue

            p_yes_a = sample_info["p_yes"]
            p_yes_b = matched_info["p_yes"]
            diff = abs(p_yes_a - p_yes_b)
            diffs_by_qid[qid].append(diff)

        for qid, diffs in diffs_by_qid.items():
            if len(diffs) == 0:
                continue
            arr = np.array(diffs, dtype=np.float64)
            stats[model][qid] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if len(diffs) > 1 else 0.0,
            }

    return stats


def _plot_single_model_axes(
    ax,
    display_name: str,
    stats_baseline_prompt: Dict[int, Dict[str, float]],
    stats_baseline_debiased: Dict[int, Dict[str, float]],
    stats_intervention_negative: Dict[int, Dict[str, float]],
    stats_intervention_random: Optional[Dict[int, Dict[str, float]]] = None,
):
    """
    在给定 axes 上画单个模型:
    - x 轴: decision_question_id 按 baseline(prompt) mean_gap 降序排序
    - y 轴: mean_gap, 比较：
    - Baseline (Original Prompt)
    - Negative Intervention (Original Prompt)
    - Random Negative Intervention (Original Prompt, 可选)
    """
    if not stats_baseline_prompt:
        ax.set_title(f"{display_name}\n(no baseline data)", fontweight="bold")
        ax.set_xticks([])
        return None, None, None

    ordered_qids: List[int] = sorted(
        stats_baseline_prompt.keys(),
        key=lambda q: stats_baseline_prompt[q]["mean"],
        reverse=True,
    )
    xs = list(range(len(ordered_qids)))

    def extract(series_stats):
        means = [series_stats.get(q, {"mean": 0.0})["mean"] for q in ordered_qids]
        stds = [series_stats.get(q, {"std": 0.0})["std"] for q in ordered_qids]
        return np.asarray(means, dtype=np.float64), np.asarray(
            stds, dtype=np.float64
        )

    means_p, stds_p = extract(stats_baseline_prompt)
    means_n, stds_n = extract(stats_intervention_negative)
    means_r, stds_r = None, None
    if stats_intervention_random is not None and len(
        stats_intervention_random
    ) > 0:
        means_r, stds_r = extract(stats_intervention_random)

    line_p = ax.plot(
        xs,
        means_p,
        label="Baseline (Original)",
        color="tab:blue",
        linewidth=2,
    )[0]
    ax.fill_between(
        xs,
        means_p - stds_p,
        means_p + stds_p,
        color="tab:blue",
        alpha=0.2,
    )

    line_n = ax.plot(
        xs,
        means_n,
        label="Intervention on Key Heads",
        color="tab:red",
        linewidth=2,
    )[0]
    ax.fill_between(
        xs,
        means_n - stds_n,
        means_n + stds_n,
        color="tab:red",
        alpha=0.2,
    )

    line_r = None
    if means_r is not None and stds_r is not None:
        line_r = ax.plot(
            xs,
            means_r,
            label="Intervention on Random Heads",
            color="tab:green",
            linewidth=2,
        )[0]
        ax.fill_between(
            xs,
            means_r - stds_r,
            means_r + stds_r,
            color="tab:green",
            alpha=0.2,
        )

    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    return line_p, line_n, line_r


def plot_intervention_qwen_and_llama_grids(
    baseline_csv: str,
    intervention_csv_negative: str,
    intervention_csv_random: Optional[str],
    out_qwen: str,
    out_llama: str,
):
    _set_font()
    stats_baseline = _load_baseline_stats(baseline_csv)
    stats_intervention_negative = _load_intervention_stats(
        intervention_csv_negative
    )
    stats_intervention_random = (
        _load_intervention_stats(intervention_csv_random)
        if intervention_csv_random
        else {}
    )

    # 模型顺序：从小到大
    qwen_models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
    llama_models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-8B-Instruct",
    ]
    llama_display = {
        "Llama-3.2-1B-Instruct": "Llama-3.2 1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2 3B",
        "Meta-Llama-3-8B-Instruct": "Llama-3.2 8B",
    }

    # --- Qwen 图（共享 y 轴）---
    fig_qwen, axes_qwen = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_q, labels_q = [], []

    for ax, model in zip(axes_qwen, qwen_models):
        model_stats = stats_baseline.get(model, {})
        stats_b_prompt = model_stats.get("prompt", {})
        stats_b_debiased = model_stats.get("debiased_prompt", {})
        stats_n = stats_intervention_negative.get(model, {})
        stats_r = stats_intervention_random.get(model, {})

        if not stats_b_prompt:
            ax.set_title(f"{model}\n(no baseline data)", fontweight="bold")
            ax.set_xticks([])
            continue

        line_p, line_n, line_r = _plot_single_model_axes(
            ax,
            model,
            stats_b_prompt,
            stats_b_debiased,
            stats_n,
            stats_r if stats_r else None,
        )
        if (
            not handles_q
            and line_p is not None
            and line_n is not None
        ):
            handles_q = [line_p, line_n]
            if line_r is not None:
                handles_q.append(line_r)
            labels_q = [h.get_label() for h in handles_q]

    if handles_q:
        fig_qwen.legend(
            handles_q,
            labels_q,
            loc="upper center",
            ncol=len(labels_q),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=16,
        )
    # 将 y 轴标题设置为第一个子图的 y 轴标签
    if len(axes_qwen) > 0:
        axes_qwen[0].set_ylabel("Fairness Violation↓", fontweight="bold")
    fig_qwen.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_qwen.savefig(out_qwen, dpi=200)
    plt.close(fig_qwen)

    # --- Llama 图（共享 y 轴）---
    fig_llama, axes_llama = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_l, labels_l = [], []

    for ax, model in zip(axes_llama, llama_models):
        model_stats = stats_baseline.get(model, {})
        stats_b_prompt = model_stats.get("prompt", {})
        stats_b_debiased = model_stats.get("debiased_prompt", {})
        stats_n = stats_intervention_negative.get(model, {})
        stats_r = stats_intervention_random.get(model, {})

        if not stats_b_prompt:
            ax.set_title(f"{model}\n(no baseline data)", fontweight="bold")
            ax.set_xticks([])
            continue

        disp_name = llama_display.get(model, model)
        line_p, line_n, line_r = _plot_single_model_axes(
            ax,
            disp_name,
            stats_b_prompt,
            stats_b_debiased,
            stats_n,
            stats_r if stats_r else None,
        )
        if (
            not handles_l
            and line_p is not None
            and line_n is not None
        ):
            handles_l = [line_p, line_n]
            if line_r is not None:
                handles_l.append(line_r)
            labels_l = [h.get_label() for h in handles_l]

    if handles_l:
        fig_llama.legend(
            handles_l,
            labels_l,
            loc="upper center",
            ncol=len(labels_l),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=16,
        )
    # 将 y 轴标题设置为第一个子图的 y 轴标签
    if len(axes_llama) > 0:
        axes_llama[0].set_ylabel("Fairness Violation↓", fontweight="bold")
    fig_llama.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_llama.savefig(out_llama, dpi=200)
    plt.close(fig_llama)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Qwen and Llama models before/after negative intervention."
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv",
        help="Path to the baseline per-sample details CSV file.",
    )
    parser.add_argument(
        "--intervention_csv",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp8/per_sample_intervention_negative_all_models.csv",
        help="Path to the negative intervention per-sample CSV file.",
    )
    parser.add_argument(
        "--intervention_csv_random",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp8/per_sample_intervention_negative_random_all_models.csv",
        help=(
            "Optional path to the random negative intervention per-sample CSV file. "
            "If provided, an additional curve will be plotted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp8",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    out_qwen = os.path.join(args.output_dir, "qwen_models_intervention_grid.pdf")
    out_llama = os.path.join(args.output_dir, "llama_models_intervention_grid.pdf")

    if not os.path.exists(args.baseline_csv):
        print(f"Error: Baseline CSV file not found: {args.baseline_csv}")
        exit(1)
    if not os.path.exists(args.intervention_csv):
        print(f"Error: Intervention CSV file not found: {args.intervention_csv}")
        exit(1)
    if args.intervention_csv_random:
        if not os.path.exists(args.intervention_csv_random):
            print(
                f"Warning: Random intervention CSV file not found: "
                f"{args.intervention_csv_random}. Random curve will not be plotted."
            )
            args.intervention_csv_random = ""

    print(f"Loading baseline statistics from: {args.baseline_csv}")
    print(f"Loading intervention statistics from: {args.intervention_csv}")
    if args.intervention_csv_random:
        print(
            f"Loading random-intervention statistics from: "
            f"{args.intervention_csv_random}"
        )

    plot_intervention_qwen_and_llama_grids(
        baseline_csv=args.baseline_csv,
        intervention_csv_negative=args.intervention_csv,
        intervention_csv_random=args.intervention_csv_random or None,
        out_qwen=out_qwen,
        out_llama=out_llama,
    )

    print(f"Saved Qwen intervention plot: {out_qwen}")
    print(f"Saved Llama intervention plot: {out_llama}")

