#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot a SINGLE LLM (default: Meta-Llama-3-8B-Instruct) in the same style as
exp8/plot_intervention_qwen_llama_grid.py.

Usage example (for Llama-3-8B-Instruct):

python plot_intervention_single_LLM.py \
  --baseline_csv /home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv \
  --intervention_csv /home/common1/hwluo/project/pFairFT/exp8/per_sample_intervention_negative_all_models.csv \
  --intervention_csv_random /home/common1/hwluo/project/pFairFT/exp8/per_sample_intervention_negative_random_all_models.csv \
  --model_name Meta-Llama-3-8B-Instruct \
  --output /home/common1/hwluo/project/pFairFT/exp8/llama_8b_intervention_single.pdf
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
    """Match the font style of plot_intervention_qwen_llama_grid.py."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def _load_baseline_stats(
    csv_path: str,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Load baseline per-sample stats.

    Same logic as exp8/plot_intervention_qwen_llama_grid.py.
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

    stats: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = dd(lambda: dd(dict))
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
    Load intervention per-sample stats.

    Same logic as exp8/plot_intervention_qwen_llama_grid.py.
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
    stats_baseline_debiased: Dict[int, Dict[str, float]],
    stats_intervention_negative: Dict[int, Dict[str, float]],
    stats_intervention_random: Optional[Dict[int, Dict[str, float]]] = None,
):
    """
    Draw a single model in the same way as in plot_intervention_qwen_llama_grid.py.
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
        return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)

    means_p, stds_p = extract(stats_baseline_prompt)
    means_n, stds_n = extract(stats_intervention_negative)
    means_r, stds_r = None, None
    if stats_intervention_random is not None and len(stats_intervention_random) > 0:
        means_r, stds_r = extract(stats_intervention_random)

    # Baseline
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

    # Key heads intervention
    line_n = ax.plot(
        xs,
        means_n,
        label="Key Heads",
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
        # Random heads intervention
        line_r = ax.plot(
            xs,
            means_r,
            label="Random Heads",
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


def plot_single_model_intervention(
    baseline_csv: str,
    intervention_csv_negative: str,
    intervention_csv_random: Optional[str],
    model_name: str,
    output_path: str,
):
    """
    Plot a single model's intervention result in the same style as one subfigure
    in exp8/qwen_models_intervention_grid.pdf.
    """
    _set_font()
    stats_baseline = _load_baseline_stats(baseline_csv)
    stats_intervention_negative = _load_intervention_stats(intervention_csv_negative)
    stats_intervention_random = (
        _load_intervention_stats(intervention_csv_random)
        if intervention_csv_random
        else {}
    )

    model_stats = stats_baseline.get(model_name, {})
    stats_b_prompt = model_stats.get("prompt", {})
    stats_b_debiased = model_stats.get("debiased_prompt", {})
    stats_n = stats_intervention_negative.get(model_name, {})
    stats_r = stats_intervention_random.get(model_name, {})

    # 单子图：比原来更宽、更矮一些
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # 横坐标使用友好显示名（默认 Llama3-8B）
    display_name = (
        "Llama3-8B" if model_name == "Meta-Llama-3-8B-Instruct" else model_name
    )

    line_p, line_n, line_r = _plot_single_model_axes(
        ax,
        display_name,
        stats_b_prompt,
        stats_b_debiased,
        stats_n,
        stats_r if stats_r else None,
    )

    # y-axis label consistent with exp8 (only one axis here)
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")

    # Legend 放在图内部，避免超出图范围
    handles = [h for h in (line_p, line_n, line_r) if h is not None]
    if handles:
        ax.legend(
            handles,
            [h.get_label() for h in handles],
            loc="best",
            fontsize=16,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved single-model intervention plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a single LLM intervention curve in the same style as exp8 grid subplots."
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
        "--model_name",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="Model name to plot (default: Meta-Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Output figure path. "
            "Default: <baseline_csv_dir>/single_intervention_<model_name>.pdf"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.baseline_csv):
        print(f"Error: Baseline CSV file not found: {args.baseline_csv}")
        exit(1)
    if not os.path.exists(args.intervention_csv):
        print(f"Error: Intervention CSV file not found: {args.intervention_csv}")
        exit(1)
    if args.intervention_csv_random and not os.path.exists(args.intervention_csv_random):
        print(
            f"Warning: Random intervention CSV file not found: "
            f"{args.intervention_csv_random}. Random curve will not be plotted."
        )
        args.intervention_csv_random = ""

    output_path = args.output
    if not output_path:
        base_dir = os.path.dirname(os.path.abspath(args.baseline_csv))
        safe_model = args.model_name.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(
            base_dir, f"single_intervention_{safe_model}.pdf"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plot_single_model_intervention(
        baseline_csv=args.baseline_csv,
        intervention_csv_negative=args.intervention_csv,
        intervention_csv_random=args.intervention_csv_random or None,
        model_name=args.model_name,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

