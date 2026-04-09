#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot mean |fact_p_yes - cf_p_yes| on Resume top-100 for multiple models.

X-axis: models (e.g., Qwen3-1.7B, Qwen3-4B, ...)
Each model has 4 bars: baseline / CE / KL / ours(exp4)
Additionally, save a summary CSV table with the mean bias per model/mode.
"""

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_bias_from_csv(csv_path: str) -> float:
    """Return mean |fact_p_yes - cf_p_yes| from a per-sample CSV.

    CSV columns: index, fact_p_yes, cf_p_yes, fact_race, cf_race
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    diffs: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fp = float(row.get("fact_p_yes", "nan"))
                cp = float(row.get("cf_p_yes", "nan"))
            except ValueError:
                continue
            if np.isnan(fp) or np.isnan(cp):
                continue
            diffs.append(abs(fp - cp))

    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp18_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp18",
        help="Root directory of exp18 outputs.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=[
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
            "Meta-Llama-3-8B-Instruct",
        ],
        help="Model names to include on x-axis.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp18/resume_top100_fairness_comparison.png",
        help="Where to save the figure.",
    )
    parser.add_argument(
        "--table_output_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp18/resume_top100_fairness_table.csv",
        help="Where to save the summary CSV table.",
    )
    args = parser.parse_args()

    modes = ["baseline", "exp5_CE", "exp5_KL", "exp4"]
    # Label order: baseline / CE / KL / ours
    mode_labels = {
        "baseline": "baseline",
        "exp5_CE": "CE",
        "exp5_KL": "KL",
        "exp4": "ours",
    }
    mode_dir_map = {
        "baseline": "baseline",
        "exp5_CE": "exp5_CE",
        "exp5_KL": "exp5_KL",
        "exp4": "exp4",
    }

    model_names: List[str] = []
    biases_per_mode: Dict[str, List[float]] = {m: [] for m in modes}

    # Also store raw dict for table writing
    table_rows: List[Dict[str, float]] = []

    for model in args.models:
        # We will only keep models where at least one mode exists
        has_any = False
        per_mode_values: Dict[str, float] = {}

        for mode in modes:
            subdir = mode_dir_map[mode]
            csv_path = os.path.join(args.exp18_root, subdir, model, "resume_top100.csv")
            if not os.path.exists(csv_path):
                per_mode_values[mode] = float("nan")
                continue
            try:
                val = _load_bias_from_csv(csv_path)
                per_mode_values[mode] = val
                has_any = True
            except Exception:
                per_mode_values[mode] = float("nan")

        if not has_any:
            continue

        model_names.append(model)
        row: Dict[str, float] = {"model": model}  # type: ignore[assignment]
        for mode in modes:
            val = per_mode_values.get(mode, float("nan"))
            biases_per_mode[mode].append(val)
            row[mode_labels[mode]] = val
        table_rows.append(row)

    if not model_names:
        print("No models with available CSVs found. Nothing to plot or save.")
        return

    # 1) Plot figure
    x = np.arange(len(model_names))
    bar_width = 0.18

    plt.figure(figsize=(max(8, len(model_names) * 1.5), 6))

    offsets = {
        "baseline": -1.5 * bar_width,
        "exp5_CE": -0.5 * bar_width,
        "exp5_KL": 0.5 * bar_width,
        "exp4": 1.5 * bar_width,
    }

    colors = {
        "baseline": "#4C72B0",
        "exp5_CE": "#55A868",
        "exp5_KL": "#C44E52",
        "exp4": "#8172B2",
    }

    for mode in modes:
        vals = biases_per_mode[mode]
        plt.bar(
            x + offsets[mode],
            vals,
            width=bar_width,
            label=mode_labels[mode],
            color=colors.get(mode, None),
        )

    plt.xticks(x, model_names, rotation=20)
    plt.ylabel("Mean |p_yes - p_cf_yes| (top-100 resume)")
    plt.xlabel("Model")
    plt.title("Resume Top-100 Fairness Gap Comparison")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plt.savefig(args.output_path)
    print(f"Saved figure to {args.output_path}")

    # 2) Save summary table
    os.makedirs(os.path.dirname(args.table_output_path) or ".", exist_ok=True)
    with open(args.table_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "baseline", "CE", "KL", "ours"]
        writer.writerow(header)
        for row in table_rows:
            baseline_val = row.get("baseline", float("nan"))
            ce_val = row.get("CE", float("nan"))
            kl_val = row.get("KL", float("nan"))
            ours_val = row.get("ours", float("nan"))

            def _fmt(v: float) -> str:
                return f"{v:.6f}" if not np.isnan(v) else "NaN"

            writer.writerow([
                row["model"],
                _fmt(baseline_val),
                _fmt(ce_val),
                _fmt(kl_val),
                _fmt(ours_val),
            ])
    print(f"Saved summary table to {args.table_output_path}")


if __name__ == "__main__":
    main()
