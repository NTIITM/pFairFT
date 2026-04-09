#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

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
    from collections import defaultdict as dd

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
                continue

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
    model_filter: Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    sample_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["sample_id"])
                matched_id_str = row.get("matched_id", "").strip()
                matched_id = int(matched_id_str) if matched_id_str else None
                decision_question_id_str = row["decision_question_id"].strip()
                decision_question_id = (
                    int(decision_question_id_str) if decision_question_id_str else None
                )
                p_yes = float(row["p_yes"])
                if decision_question_id is None:
                    continue
                sample_data.append(
                    {
                        "sample_id": sample_id,
                        "matched_id": matched_id,
                        "decision_question_id": decision_question_id,
                        "p_yes": p_yes,
                    }
                )
            except (ValueError, KeyError):
                continue

    data_by_id: Dict[int, dict] = {}
    for s in sample_data:
        data_by_id[int(s["sample_id"])] = {
            "decision_question_id": int(s["decision_question_id"]),
            "p_yes": float(s["p_yes"]),
            "matched_id": s["matched_id"],
        }

    from collections import defaultdict as dd2

    diffs_by_qid: Dict[int, List[float]] = dd2(list)
    processed_pairs = set()
    processed_samples = set()

    for sample_id, sample_info in data_by_id.items():
        if sample_id in processed_samples:
            continue

        matched_id = sample_info["matched_id"]
        if matched_id is None or int(matched_id) not in data_by_id:
            continue

        matched_id = int(matched_id)
        pair_key = tuple(sorted([sample_id, matched_id]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        processed_samples.add(sample_id)
        processed_samples.add(matched_id)

        qid = int(sample_info["decision_question_id"])
        matched_info = data_by_id[matched_id]
        if int(matched_info["decision_question_id"]) != qid:
            continue

        diff = abs(float(sample_info["p_yes"]) - float(matched_info["p_yes"]))
        diffs_by_qid[qid].append(diff)

    stats: Dict[int, Dict[str, float]] = {}
    for qid, diffs in diffs_by_qid.items():
        if not diffs:
            continue
        arr = np.array(diffs, dtype=np.float64)
        stats[qid] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)) if len(diffs) > 1 else 0.0,
        }
    return stats


def _plot_single_model_axes(
    ax,
    display_name: str,
    stats_baseline_prompt: Dict[int, Dict[str, float]],
    stats_partial: Optional[Dict[int, Dict[str, float]]],
    stats_all: Optional[Dict[int, Dict[str, float]]],
):
    if not stats_baseline_prompt:
        ax.set_title(f"{display_name}\n(no baseline data)", fontweight="bold")
        ax.set_xticks([])
        return

    ordered_qids: List[int] = sorted(
        stats_baseline_prompt.keys(),
        key=lambda q: stats_baseline_prompt[q]["mean"],
        reverse=True,
    )
    xs = list(range(len(ordered_qids)))

    def extract(series_stats):
        means = [series_stats.get(q, {"mean": 0.0})["mean"] for q in ordered_qids]
        stds = [series_stats.get(q, {"std": 0.0})["std"] for q in ordered_qids]
        return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)

    means_b, stds_b = extract(stats_baseline_prompt)
    ax.plot(xs, means_b, label="Baseline (Original)", color="tab:blue", linewidth=2)
    ax.fill_between(xs, means_b - stds_b, means_b + stds_b, color="tab:blue", alpha=0.2)

    if stats_partial is not None and len(stats_partial) > 0:
        means_p, stds_p = extract(stats_partial)
        ax.plot(xs, means_p, label="Projection Debias (partial heads, exp17)", color="tab:purple", linewidth=2)
        ax.fill_between(xs, means_p - stds_p, means_p + stds_p, color="tab:purple", alpha=0.2)

    if stats_all is not None and len(stats_all) > 0:
        means_a, stds_a = extract(stats_all)
        ax.plot(xs, means_a, label="Projection Debias (all heads, exp25)", color="tab:orange", linewidth=2)
        ax.fill_between(xs, means_a - stds_a, means_a + stds_a, color="tab:orange", alpha=0.2)

    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)


def _group_models(models: List[str]) -> Tuple[List[str], List[str]]:
    qwen = [m for m in models if m.lower().startswith("qwen")]
    llama = [m for m in models if "llama" in m.lower() or "meta-llama" in m.lower()]
    # stable order
    qwen_order = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
    llama_order = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3-8B-Instruct"]
    qwen_sorted = [m for m in qwen_order if m in qwen] + [m for m in qwen if m not in qwen_order]
    llama_sorted = [m for m in llama_order if m in llama] + [m for m in llama if m not in llama_order]
    return qwen_sorted, llama_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/common1/hwluo/project/pFairFT")
    parser.add_argument("--models", type=str, nargs="*", default=[
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-8B-Instruct",
    ])
    parser.add_argument("--baseline_csv", type=str, default="")
    parser.add_argument("--exp17_partial_csv", type=str, default="/home/common1/hwluo/project/pFairFT/exp17/per_sample_intervention_projection_all_models.csv")
    parser.add_argument("--all_dir", type=str, default="/home/common1/hwluo/project/pFairFT/exp25/results_{model}/per_sample_intervention_all_heads.csv")
    parser.add_argument("--out_qwen", type=str, default="/home/common1/hwluo/project/pFairFT/exp25/qwen_models_intervention_grid_exp25.pdf")
    parser.add_argument("--out_llama", type=str, default="/home/common1/hwluo/project/pFairFT/exp25/llama_models_intervention_grid_exp25.pdf")
    args = parser.parse_args()

    _set_font()

    baseline_csv = args.baseline_csv or os.path.join(args.base_dir, "exp1", "per_sample_details_all_models.csv")
    stats_baseline = _load_baseline_stats(baseline_csv)

    qwen_models, llama_models = _group_models(args.models)

    def plot_group(models: List[str], out_path: str, display_map: Optional[Dict[str, str]] = None):
        if not models:
            return
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]

        handles, labels = [], []
        for ax, model in zip(axes, models):
            model_stats = stats_baseline.get(model, {})
            stats_b_prompt = model_stats.get("prompt", {})

            exp17_partial_csv = args.exp17_partial_csv
            all_path = args.all_dir.format(model=model)

            stats_partial = _load_intervention_stats(exp17_partial_csv, model_filter=model) if os.path.exists(exp17_partial_csv) else None
            stats_all = _load_intervention_stats(all_path) if os.path.exists(all_path) else None

            disp = display_map.get(model, model) if display_map else model
            _plot_single_model_axes(ax, disp, stats_b_prompt, stats_partial, stats_all)

            if not handles:
                handles, labels = ax.get_legend_handles_labels()

        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels),
                bbox_to_anchor=(0.5, 0.98),
                fontsize=12,
            )

        axes[0].set_ylabel("Fairness Violation↓", fontweight="bold")
        fig.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    llama_display = {
        "Llama-3.2-1B-Instruct": "Llama-3.2 1B",
        "Llama-3.2-3B-Instruct": "Llama-3.2 3B",
        "Meta-Llama-3-8B-Instruct": "Llama-3.2 8B",
    }

    plot_group(qwen_models, args.out_qwen)
    plot_group(llama_models, args.out_llama, display_map=llama_display)


if __name__ == "__main__":
    main()
