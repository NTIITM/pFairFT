#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只画 exp1/plot_qwen_llama_grid.py 中某一个模型对应的单子图（Original Prompt vs Debiased Prompt）。
默认画 Qwen3-4B，风格参考 exp8/plot_intervention_single_LLM.py。

数据来源：exp1 per_sample_details_all_models.csv（与 plot_qwen_llama_grid.py 相同）。
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
    从 per-sample CSV 读取数据，统计后返回。
    返回: stats[model][prompt_type][qid] = {"mean": ..., "std": ...}
    prompt_type ∈ {"prompt", "debiased_prompt"}
    与 exp1/plot_qwen_llama_grid.py 中 _load_stats_from_csv 逻辑一致。
    """
    from collections import defaultdict as dd

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
            except (ValueError, KeyError):
                continue

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

    stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = dd(lambda: dd(dict))
    for model in data_by_model_prompt:
        for prompt_type in data_by_model_prompt[model]:
            model_prompt_data = data_by_model_prompt[model][prompt_type]
            diffs_by_qid: Dict[int, List[float]] = dd(list)
            processed_pairs = set()
            processed_samples = set()

            for sample_id, sample_info in model_prompt_data.items():
                if sample_id in processed_samples:
                    continue
                matched_id = sample_info["matched_id"]
                if matched_id is None or matched_id not in model_prompt_data:
                    continue
                pair_key = tuple(sorted([sample_id, matched_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                processed_samples.add(sample_id)
                processed_samples.add(matched_id)
                qid = sample_info["decision_question_id"]
                matched_info = model_prompt_data[matched_id]
                if matched_info["decision_question_id"] != qid:
                    continue
                p_yes_a = sample_info["p_yes"]
                p_yes_b = matched_info["p_yes"]
                diff = abs(p_yes_a - p_yes_b)
                diffs_by_qid[qid].append(diff)

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
    在给定 axes 上画单个模型：Original Prompt vs Debiased Prompt。
    与 plot_qwen_llama_grid.py 中 _plot_single_model_axes 一致。
    """
    if not stats_prompt:
        ax.set_title(f"{display_name}\n(no data)", fontweight="bold")
        ax.set_xticks([])
        return None, None

    ordered_qids = sorted(stats_prompt.keys(), key=lambda q: stats_prompt[q]["mean"], reverse=True)
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

    line_p = ax.plot(xs, means_p, label="Original Prompt", color="tab:blue", linewidth=2)[0]
    ax.fill_between(xs, means_p - stds_p, means_p + stds_p, color="tab:blue", alpha=0.2)

    line_d = ax.plot(xs, means_d, label="Debiased Prompt", color="tab:orange", linewidth=2)[0]
    ax.fill_between(xs, means_d - stds_d, means_d + stds_d, color="tab:orange", alpha=0.2)

    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)
    return line_p, line_d


def plot_single_model_prompt(
    csv_path: str,
    model_name: str,
    output_path: str,
    display_name: str = None,
):
    """
    只画指定模型在 exp1 下的 Original vs Debiased 单图，
    风格与 plot_intervention_single_LLM 单子图一致。
    """
    _set_font()
    stats = _load_stats_from_csv(csv_path)
    model_stats = stats.get(model_name, {})
    stats_prompt = model_stats.get("prompt", {})
    stats_debiased = model_stats.get("debiased_prompt", {})

    if not stats_prompt:
        print(f"No data for model {model_name}. Skipping.")
        return

    if display_name is None:
        display_name = "Qwen 4B" if model_name == "Qwen3-4B" else model_name

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    line_p, line_d = _plot_single_model_axes(ax, display_name, stats_prompt, stats_debiased)
    if line_p is None:
        plt.close(fig)
        return

    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    handles = [line_p, line_d]
    ax.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=16)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved single-model prompt plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a single model (default: Qwen3-4B) Original vs Debiased Prompt, same as one subplot of exp1 qwen_models_grid."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv",
        help="Path to the per-sample details CSV (same as plot_qwen_llama_grid).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-4B",
        help="Model to plot (default: Qwen3-4B).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path. Default: <csv_dir>/single_prompt_<model_name>.pdf",
    )
    parser.add_argument(
        "--display_name",
        type=str,
        default="",
        help="X-axis label. Default: 'Qwen 4B' for Qwen3-4B, else model_name.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV not found: {args.csv_path}")
        exit(1)

    output_path = args.output
    if not output_path:
        base_dir = os.path.dirname(os.path.abspath(args.csv_path))
        safe = args.model_name.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(base_dir, f"single_prompt_{safe}.pdf")

    display_name = args.display_name or None
    plot_single_model_prompt(
        csv_path=args.csv_path,
        model_name=args.model_name,
        output_path=output_path,
        display_name=display_name,
    )


if __name__ == "__main__":
    main()
