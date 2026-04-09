#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_bias_from_csv(csv_path: str) -> float:
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
        help="Root directory of exp18 outputs (baseline).",
    )
    parser.add_argument(
        "--exp25_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp25",
        help="Root directory of exp25 outputs (interventions).",
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
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp25/resume_top100_fairness_exp25.png",
    )
    parser.add_argument(
        "--table_output_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp25/resume_top100_fairness_exp25_table.csv",
    )
    args = parser.parse_args()

    modes = ["baseline", "partial", "all"]
    mode_labels = {"baseline": "baseline", "partial": "partial(exp17)", "all": "all(exp25)"}

    model_names: List[str] = []
    biases_per_mode: Dict[str, List[float]] = {m: [] for m in modes}
    table_rows: List[Dict[str, float]] = []

    for model in args.models:
        per_mode: Dict[str, float] = {}
        has_any = False

        # baseline: reuse exp18 baseline csv path layout
        baseline_csv = os.path.join(args.exp18_root, "baseline", model, "resume_top100.csv")
        if os.path.exists(baseline_csv):
            per_mode["baseline"] = _load_bias_from_csv(baseline_csv)
            has_any = True
        else:
            per_mode["baseline"] = float("nan")

        # partial/all: exp25 layout
        partial_csv = os.path.join(args.exp25_root, f"results_{model}", "resume_top100_partial.csv")
        all_csv = os.path.join(args.exp25_root, f"results_{model}", "resume_top100_all.csv")

        per_mode["partial"] = _load_bias_from_csv(partial_csv) if os.path.exists(partial_csv) else float("nan")
        per_mode["all"] = _load_bias_from_csv(all_csv) if os.path.exists(all_csv) else float("nan")
        if os.path.exists(partial_csv) or os.path.exists(all_csv):
            has_any = True

        if not has_any:
            continue

        model_names.append(model)
        row: Dict[str, float] = {"model": model}  # type: ignore[assignment]
        for m in modes:
            biases_per_mode[m].append(per_mode[m])
            row[mode_labels[m]] = per_mode[m]
        table_rows.append(row)

    if not model_names:
        print("No models with available CSVs found.")
        return

    x = np.arange(len(model_names))
    bar_width = 0.22

    plt.figure(figsize=(max(8, len(model_names) * 1.5), 6))

    offsets = {"baseline": -bar_width, "partial": 0.0, "all": bar_width}
    colors = {"baseline": "#4C72B0", "partial": "#8172B2", "all": "#DD8452"}

    for m in modes:
        plt.bar(
            x + offsets[m],
            biases_per_mode[m],
            width=bar_width,
            label=mode_labels[m],
            color=colors.get(m, None),
        )

    plt.xticks(x, model_names, rotation=20)
    plt.ylabel("Mean |p_yes - p_cf_yes| (top-100 resume)")
    plt.xlabel("Model")
    plt.title("Resume Top-100 Fairness Gap: Baseline vs Projection Intervention")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plt.savefig(args.output_path)
    print(f"Saved figure to {args.output_path}")

    os.makedirs(os.path.dirname(args.table_output_path) or ".", exist_ok=True)
    with open(args.table_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "baseline", "partial(exp17)", "all(exp25)"])
        for row in table_rows:
            def _fmt(v: float) -> str:
                return f"{v:.6f}" if not np.isnan(v) else "NaN"

            writer.writerow([
                row["model"],
                _fmt(float(row.get("baseline", float("nan")))),
                _fmt(float(row.get("partial(exp17)", float("nan")))),
                _fmt(float(row.get("all(exp25)", float("nan")))),
            ])

    print(f"Saved table to {args.table_output_path}")


if __name__ == "__main__":
    main()
