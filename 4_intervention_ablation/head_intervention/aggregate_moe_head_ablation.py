#!/usr/bin/env python
"""Aggregate sensitive-vs-random head ablations for one MOE model."""

import argparse
import csv
import glob
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


def _plot_curves(
    model_name: str,
    datasets: List[Tuple[str, Dict[int, Tuple[float, float, int]], Dict[int, Tuple[float, float, int]]]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.1))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (label, sensitive, random) in zip(axes, datasets):
        for values, name, color, marker in (
            (sensitive, "Sensitive heads", "#b42318", "o"),
            (random, "Random heads (5 seeds)", "#175cd3", "s"),
        ):
            counts = sorted(values)
            means = np.array([values[count][0] for count in counts])
            stds = np.array([values[count][1] for count in counts])
            ax.plot(counts, means, marker=marker, color=color, linewidth=2, label=name)
            if np.any(stds > 0):
                ax.fill_between(counts, means - stds, means + stds, color=color, alpha=0.16)
        ax.set_title(label)
        ax.set_xlabel("Number of intervened heads")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean fairness violation")
    axes[-1].legend(frameon=False)
    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_qids(
    model_name: str,
    sensitive: Dict[int, Dict[int, float]],
    random_runs: List[Dict[int, Dict[int, float]]],
    output_path: str,
) -> Dict[str, int]:
    max_count = max(sensitive)
    baseline = sensitive.get(0, {})
    sensitive_max = sensitive[max_count]
    random_at_max = [run[max_count] for run in random_runs if max_count in run]
    qids = sorted(baseline, key=baseline.get, reverse=True)
    x = np.arange(len(qids))
    random_matrix = np.array(
        [[run.get(qid, np.nan) for qid in qids] for run in random_at_max],
        dtype=np.float64,
    )
    random_mean = np.nanmean(random_matrix, axis=0)
    random_std = np.nanstd(random_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x, [baseline[qid] for qid in qids], color="#475467", label="Baseline")
    ax.plot(
        x,
        [sensitive_max.get(qid, np.nan) for qid in qids],
        color="#b42318",
        label=f"Sensitive heads (H={max_count})",
    )
    ax.plot(x, random_mean, color="#175cd3", label=f"Random heads (H={max_count}, 5 seeds)")
    ax.fill_between(x, random_mean - random_std, random_mean + random_std, color="#175cd3", alpha=0.16)
    ax.set_title(model_name)
    ax.set_xlabel("Decision scenarios ordered by baseline violation")
    ax.set_ylabel("Mean fairness violation")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {"head_count": max_count, "num_qids": len(qids), "num_random_seeds": len(random_at_max)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    resume_sensitive_path = os.path.join(
        args.ablation_root,
        "head_resume_topk/sensitive/intervention_results_by_head_count.csv",
    )
    resume_random_paths = sorted(
        glob.glob(
            os.path.join(
                args.ablation_root,
                "head_resume_topk/random_seed_*/intervention_results_by_head_count_random.csv",
            )
        )
    )
    discrim_sensitive_path = os.path.join(
        args.ablation_root, "head_discrim_topk/negative_seed_42/results.csv"
    )
    discrim_random_paths = sorted(
        glob.glob(
            os.path.join(
                args.ablation_root, "head_discrim_topk/negative_random_seed_*/results.csv"
            )
        )
    )
    required = [resume_sensitive_path, discrim_sensitive_path]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing or not resume_random_paths or not discrim_random_paths:
        raise FileNotFoundError(
            "Incomplete head ablation outputs: "
            + ", ".join(missing or ["random-seed result directories"])
        )

    os.makedirs(args.output_dir, exist_ok=True)
    resume_sensitive_runs = [_mean_by_head_count(resume_sensitive_path, "bias_level")]
    resume_random_runs = [
        _mean_by_head_count(path, "bias_level") for path in resume_random_paths
    ]
    discrim_sensitive_runs = [
        _mean_by_head_count(discrim_sensitive_path, "mean_p_yes_gap")
    ]
    discrim_random_runs = [
        _mean_by_head_count(path, "mean_p_yes_gap") for path in discrim_random_paths
    ]

    aggregates = {
        "resume": {
            "sensitive": _aggregate(resume_sensitive_runs),
            "random": _aggregate(resume_random_runs),
        },
        "discrim": {
            "sensitive": _aggregate(discrim_sensitive_runs),
            "random": _aggregate(discrim_random_runs),
        },
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
    _plot_curves(
        args.model_name,
        [
            ("Resume top-100", aggregates["resume"]["sensitive"], aggregates["resume"]["random"]),
            ("Discrim-Eval transfer", aggregates["discrim"]["sensitive"], aggregates["discrim"]["random"]),
        ],
        curve_pdf,
    )
    discrim_sensitive_qids = _qid_by_head_count(discrim_sensitive_path)
    discrim_random_qids = [_qid_by_head_count(path) for path in discrim_random_paths]
    qid_pdf = os.path.join(args.output_dir, "discrim_scenarios_sensitive_vs_random.pdf")
    qid_summary = _plot_qids(
        args.model_name, discrim_sensitive_qids, discrim_random_qids, qid_pdf
    )

    baseline_values = [run.get(0, float("nan")) for run in resume_random_runs]
    metadata = {
        "model_name": args.model_name,
        "resume_sensitive_path": resume_sensitive_path,
        "resume_random_paths": resume_random_paths,
        "discrim_sensitive_path": discrim_sensitive_path,
        "discrim_random_paths": discrim_random_paths,
        "random_seeds": [_seed_from_path(path) for path in discrim_random_paths],
        "resume_random_baseline_range": float(np.nanmax(baseline_values) - np.nanmin(baseline_values)),
        "qid_plot": qid_summary,
        "summary_csv": summary_csv,
        "curve_pdf": curve_pdf,
        "qid_pdf": qid_pdf,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
