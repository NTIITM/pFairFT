#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
) -> Dict[str, Dict[int, Dict[str, float]]]:
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

    data_by_model: Dict[str, Dict[int, dict]] = dd(dict)
    for s in sample_data:
        model = s["model"]
        sample_id = s["sample_id"]
        data_by_model[model][sample_id] = {
            "decision_question_id": s["decision_question_id"],
            "p_yes": s["p_yes"],
            "matched_id": s["matched_id"],
        }

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
    stats_exp8_negative: Dict[int, Dict[str, float]],
    stats_exp17_projection: Dict[int, Dict[str, float]],
    stats_exp8_random: Optional[Dict[int, Dict[str, float]]] = None,
):
    if not stats_baseline_prompt:
        ax.set_title(f"{display_name}\n(no baseline data)", fontweight="bold")
        ax.set_xticks([])
        return None

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
    means_n, stds_n = extract(stats_exp8_negative)
    means_p, stds_p = extract(stats_exp17_projection)

    line_b = ax.plot(xs, means_b, label="Baseline (Original)", color="tab:blue", linewidth=2)[0]
    ax.fill_between(xs, means_b - stds_b, means_b + stds_b, color="tab:blue", alpha=0.2)

    line_n = ax.plot(xs, means_n, label="Negative Intervention (exp8)", color="tab:red", linewidth=2)[0]
    ax.fill_between(xs, means_n - stds_n, means_n + stds_n, color="tab:red", alpha=0.2)

    line_p = ax.plot(xs, means_p, label="Projection Debias (exp17)", color="tab:purple", linewidth=2)[0]
    ax.fill_between(xs, means_p - stds_p, means_p + stds_p, color="tab:purple", alpha=0.2)

    if stats_exp8_random is not None and len(stats_exp8_random) > 0:
        means_r, stds_r = extract(stats_exp8_random)
        ax.plot(xs, means_r, label="Random Negative (exp8)", color="tab:green", linewidth=2)
        ax.fill_between(xs, means_r - stds_r, means_r + stds_r, color="tab:green", alpha=0.2)

    ax.set_xlabel(display_name, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    return line_b


def plot_all_grids():
    _set_font()

    base_dir = "/home/common1/hwluo/project/pFairFT"

    baseline_csv = os.path.join(base_dir, "exp1", "per_sample_details_all_models.csv")
    exp8_negative_csv = os.path.join(base_dir, "exp8", "per_sample_intervention_negative_all_models.csv")
    exp8_random_csv = os.path.join(base_dir, "exp8", "per_sample_intervention_negative_random_all_models.csv")
    exp17_projection_csv = os.path.join(base_dir, "exp17", "per_sample_intervention_projection_all_models.csv")

    out_qwen = os.path.join(base_dir, "exp17", "qwen_models_intervention_grid_all.pdf")
    out_llama = os.path.join(base_dir, "exp17", "llama_models_intervention_grid_all.pdf")

    stats_baseline = _load_baseline_stats(baseline_csv)
    stats_exp8_negative = _load_intervention_stats(exp8_negative_csv)
    stats_exp8_random = _load_intervention_stats(exp8_random_csv) if os.path.exists(exp8_random_csv) else {}
    stats_exp17_projection = _load_intervention_stats(exp17_projection_csv)

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

    fig_qwen, axes_qwen = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_q, labels_q = [], []

    for ax, model in zip(axes_qwen, qwen_models):
        model_stats = stats_baseline.get(model, {})
        stats_b_prompt = model_stats.get("prompt", {})

        line_b = _plot_single_model_axes(
            ax=ax,
            display_name=model,
            stats_baseline_prompt=stats_b_prompt,
            stats_exp8_negative=stats_exp8_negative.get(model, {}),
            stats_exp17_projection=stats_exp17_projection.get(model, {}),
            stats_exp8_random=stats_exp8_random.get(model, {}) if stats_exp8_random else None,
        )

        if not handles_q and line_b is not None:
            handles_q, labels_q = ax.get_legend_handles_labels()

    if handles_q:
        fig_qwen.legend(
            handles_q,
            labels_q,
            loc="upper center",
            ncol=len(labels_q),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=14,
        )

    if len(axes_qwen) > 0:
        axes_qwen[0].set_ylabel("Fairness Violation↓", fontweight="bold")

    fig_qwen.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_qwen.savefig(out_qwen, dpi=200)
    plt.close(fig_qwen)

    fig_llama, axes_llama = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    handles_l, labels_l = [], []

    for ax, model in zip(axes_llama, llama_models):
        model_stats = stats_baseline.get(model, {})
        stats_b_prompt = model_stats.get("prompt", {})

        disp_name = llama_display.get(model, model)
        line_b = _plot_single_model_axes(
            ax=ax,
            display_name=disp_name,
            stats_baseline_prompt=stats_b_prompt,
            stats_exp8_negative=stats_exp8_negative.get(model, {}),
            stats_exp17_projection=stats_exp17_projection.get(model, {}),
            stats_exp8_random=stats_exp8_random.get(model, {}) if stats_exp8_random else None,
        )

        if not handles_l and line_b is not None:
            handles_l, labels_l = ax.get_legend_handles_labels()

    if handles_l:
        fig_llama.legend(
            handles_l,
            labels_l,
            loc="upper center",
            ncol=len(labels_l),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=14,
        )

    if len(axes_llama) > 0:
        axes_llama[0].set_ylabel("Fairness Violation↓", fontweight="bold")

    fig_llama.tight_layout(rect=[0.0, 0.08, 0.98, 0.90], w_pad=0.25)
    fig_llama.savefig(out_llama, dpi=200)
    plt.close(fig_llama)

    print(f"Saved Qwen plot: {out_qwen}")
    print(f"Saved Llama plot: {out_llama}")


if __name__ == "__main__":
    plot_all_grids()
