#!/usr/bin/env python
"""Aggregate sensitive-vs-random head ablations for one MOE model."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Series = Dict[int, float]


def _mean_by_head_count(path: str, value_column: str) -> Series:
    grouped: Dict[int, List[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[int(row["head_count"])].append(float(row[value_column]))
    return {count: float(np.mean(values)) for count, values in grouped.items()}


def _qid_by_head_count(path: str) -> Dict[int, Dict[int, float]]:
    grouped: Dict[int, Dict[int, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[int(row["head_count"])][int(row["decision_question_id"])] = float(
                row["mean_p_yes_gap"]
            )
    return dict(grouped)


def _seed_from_path(path: str) -> int:
    directory = os.path.basename(os.path.dirname(path))
    return int(directory.rsplit("_seed_", 1)[-1])


def _aggregate(series: Iterable[Series]) -> Dict[int, Tuple[float, float, int]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for run in series:
        for count, value in run.items():
            buckets[count].append(value)
    return {
        count: (float(np.mean(values)), float(np.std(values)), len(values))
        for count, values in buckets.items()
    }


def _set_font(size: int = 18) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = size
    plt.rcParams["font.weight"] = "bold"


def _display_name(model_name: str) -> str:
    if model_name == "Qwen1.5-MoE-A2.7B-Chat":
        return "Qwen-MOE"
    return model_name


def _plot_head_count(
    model_name: str,
    sensitive: Dict[int, Tuple[float, float, int]],
    random: Dict[int, Tuple[float, float, int]],
    output_path: str,
    title: str,
    font_size: int,
    sensitive_label: str,
    random_label: str,
) -> None:
    _set_font(font_size)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for values, name, color, marker in (
        (sensitive, sensitive_label, "tab:blue", "o"),
        (random, random_label, "tab:orange", "s"),
    ):
        counts = sorted(values)
        means = np.array([values[count][0] for count in counts])
        ax.plot(
            counts,
            means,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=name,
            color=color,
        )
    all_counts = sorted(set(sensitive) | set(random))
    ax.set_xticks(all_counts)
    ax.set_xlabel("Number of Intervened Heads", fontweight="bold")
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    display_name = _display_name(model_name)
    ax.set_title(
        display_name if not title else f"{display_name}: {title}",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=max(10, font_size - 2))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_qids(
    model_name: str,
    sensitive: Dict[int, Dict[int, float]],
    random_runs: List[Dict[int, Dict[int, float]]],
    output_path: str,
) -> Dict[str, int]:
    _set_font(12)
    head_counts = sorted(sensitive)
    baseline = sensitive.get(0, {})
    qids = sorted(baseline, key=baseline.get, reverse=True)
    x = np.arange(len(qids))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(head_counts)))
    for color, head_count in zip(colors, head_counts):
        sensitive_values = sensitive.get(head_count, {})
        ax.plot(
            x,
            [sensitive_values.get(qid, np.nan) for qid in qids],
            label=f"Key Heads (H={head_count})",
            color=color,
            marker="s",
            linestyle="-",
            linewidth=2,
            markersize=6,
        )
        random_at_count = [run[head_count] for run in random_runs if head_count in run]
        if random_at_count:
            random_matrix = np.asarray(
                [[run.get(qid, np.nan) for qid in qids] for run in random_at_count],
                dtype=np.float64,
            )
            ax.plot(
                x,
                np.nanmean(random_matrix, axis=0),
                label=f"Random     (H={head_count})",
                color=color,
                marker="o",
                linestyle="--",
                linewidth=2,
                markersize=6,
            )
    ax.set_xticks([])
    ax.set_xlabel(_display_name(model_name), fontweight="bold")
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "head_counts": head_counts,
        "num_qids": len(qids),
        "num_random_seeds": len(random_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume_root",
        default="",
        help="Optional directory containing sensitive/ and random_seed_42/ resume outputs.",
    )
    parser.add_argument(
        "--resume_only",
        action="store_true",
        help="Aggregate only Resume exp9 outputs; do not require Discrim-Eval files.",
    )
    args = parser.parse_args()

    resume_root = args.resume_root or os.path.join(
        args.ablation_root, "head_resume_topk"
    )
    resume_sensitive_path = os.path.join(
        resume_root,
        "sensitive/intervention_results_by_head_count.csv",
    )
    resume_random_paths = [
        os.path.join(
            resume_root,
            "random_seed_42/intervention_results_by_head_count_random.csv",
        )
    ]
    discrim_sensitive_path = os.path.join(
        args.ablation_root, "head_discrim_topk/negative_seed_42/results.csv"
    )
    discrim_random_paths = [
        os.path.join(
            args.ablation_root,
            "head_discrim_topk/negative_random_seed_42/results.csv",
        )
    ]
    required = [resume_sensitive_path]
    if not args.resume_only:
        required.append(discrim_sensitive_path)
    missing = [path for path in required if not os.path.isfile(path)]
    missing.extend(
        path
        for path in resume_random_paths
        if not os.path.isfile(path)
    )
    if not args.resume_only:
        missing.extend(path for path in discrim_random_paths if not os.path.isfile(path))
    if missing:
        raise FileNotFoundError(
            "Incomplete head ablation outputs: " + ", ".join(missing)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    resume_sensitive_runs = [_mean_by_head_count(resume_sensitive_path, "bias_level")]
    resume_random_runs = [
        _mean_by_head_count(path, "bias_level") for path in resume_random_paths
    ]
    aggregates = {
        "resume": {
            "sensitive": _aggregate(resume_sensitive_runs),
            "random": _aggregate(resume_random_runs),
        }
    }
    if not args.resume_only:
        discrim_sensitive_runs = [
            _mean_by_head_count(discrim_sensitive_path, "mean_p_yes_gap")
        ]
        discrim_random_runs = [
            _mean_by_head_count(path, "mean_p_yes_gap")
            for path in discrim_random_paths
        ]
        aggregates["discrim"] = {
            "sensitive": _aggregate(discrim_sensitive_runs),
            "random": _aggregate(discrim_random_runs),
        }
    summary_csv = os.path.join(args.output_dir, "head_ablation_aggregated.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "intervention", "head_count", "mean_gap", "std_gap", "num_runs"])
        for dataset, by_intervention in aggregates.items():
            for intervention, by_count in by_intervention.items():
                for count, (mean, std, num_runs) in sorted(by_count.items()):
                    writer.writerow([dataset, intervention, count, mean, std, num_runs])

    curve_pdf = os.path.join(args.output_dir, "sensitive_vs_random_head_count.pdf")
    _plot_head_count(
        args.model_name,
        aggregates["resume"]["sensitive"],
        aggregates["resume"]["random"],
        curve_pdf,
        "",
        14,
        "Sensitive heads",
        "Random heads",
    )
    discrim_curve_pdf = None
    qid_pdf = None
    qid_summary = None
    if not args.resume_only:
        discrim_curve_pdf = os.path.join(
            args.output_dir, "discrim_sensitive_vs_random_head_count.pdf"
        )
        _plot_head_count(
            args.model_name,
            aggregates["discrim"]["sensitive"],
            aggregates["discrim"]["random"],
            discrim_curve_pdf,
            "Mean Bias Reduction by Head Count (Discrim-Eval)",
            18,
            "Key Heads",
            "Random Heads",
        )
        discrim_sensitive_qids = _qid_by_head_count(discrim_sensitive_path)
        discrim_random_qids = [
            _qid_by_head_count(path) for path in discrim_random_paths
        ]
        qid_pdf = os.path.join(
            args.output_dir, "discrim_scenarios_sensitive_vs_random.pdf"
        )
        qid_summary = _plot_qids(
            args.model_name, discrim_sensitive_qids, discrim_random_qids, qid_pdf
        )

    baseline_values = [run.get(0, float("nan")) for run in resume_random_runs]
    metadata = {
        "model_name": args.model_name,
        "style_reference_root": "/home/common1/hwluo/project/fairness_llm_new copy",
        "style_references": {
            "resume_head_count": "exp9/plot_intervention_all_models.py",
            "discrim_head_count": "exp10/plot_discrim_eval_head_count.py",
            "discrim_scenarios": "exp10/plot_discrim_eval_head_count.py",
        },
        "resume_only": args.resume_only,
        "resume_root": resume_root,
        "resume_sensitive_path": resume_sensitive_path,
        "resume_random_paths": resume_random_paths,
        "discrim_sensitive_path": None if args.resume_only else discrim_sensitive_path,
        "discrim_random_paths": [] if args.resume_only else discrim_random_paths,
        "random_seeds": [_seed_from_path(path) for path in resume_random_paths],
        "resume_random_baseline_range": float(np.nanmax(baseline_values) - np.nanmin(baseline_values)),
        "qid_plot": qid_summary,
        "summary_csv": summary_csv,
        "curve_pdf": curve_pdf,
        "discrim_curve_pdf": discrim_curve_pdf,
        "qid_pdf": qid_pdf,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
