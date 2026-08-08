#!/usr/bin/env python
"""Audit and summarize nine-model Adult race baseline and top-100 results."""

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Sequence


MODELS = (
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "DeepSeek-V2-Lite-Chat",
    "JetMoE-8B-Chat",
    "OLMoE-1B-7B-0924-Instruct",
)
CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")


def _json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_model(results_root: str, model: str, top_k: int) -> Dict[str, Any]:
    base_dir = os.path.join(
        results_root, model, "intervention_ablation", "adult_race_yesno_full_baseline_seed_42"
    )
    top_dir = os.path.join(
        results_root,
        model,
        "intervention_ablation",
        f"adult_race_yesno_high_gap_top{top_k}_seed_42",
    )
    eval_dir = os.path.join(top_dir, "evaluation")
    base_metadata = _json(os.path.join(base_dir, "metadata.json"))
    selection = _json(os.path.join(top_dir, "selection_metadata.json"))
    evaluation = _json(os.path.join(eval_dir, "metadata.json"))
    if base_metadata.get("status") != "complete" or evaluation.get("status") != "complete":
        raise ValueError(f"{model}: incomplete metadata.")
    if base_metadata.get("evaluation_protocol") != "yes_no_income_gt_50k_v1":
        raise ValueError(f"{model}: baseline uses a non-Yes/No protocol.")
    if evaluation.get("evaluation_protocol") != "yes_no_income_gt_50k_v1":
        raise ValueError(f"{model}: intervention uses a non-Yes/No protocol.")
    if int(base_metadata["pairs"]) != 46447:
        raise ValueError(f"{model}: expected 46447 baseline pairs.")
    if int(selection["selected_pairs"]) != top_k or int(evaluation["pairs"]) != top_k:
        raise ValueError(f"{model}: invalid top-K counts.")
    if selection.get("intervention_outputs_used_for_selection") is not False:
        raise ValueError(f"{model}: selection was not baseline-only.")

    summary_rows = {row["condition"]: row for row in _csv(os.path.join(eval_dir, "summary.csv"))}
    if set(summary_rows) != set(CONDITIONS):
        raise ValueError(f"{model}: incomplete intervention conditions.")
    expected_pair_ids = selection["selected_pair_ids"]
    for condition in CONDITIONS:
        rows = _csv(os.path.join(eval_dir, f"per_pair_{condition}.csv"))
        if [int(row["pair_id"]) for row in rows] != expected_pair_ids:
            raise ValueError(f"{model}: pair order mismatch for {condition}.")
        if not all(
            math.isfinite(float(row[column]))
            for row in rows
            for column in ("white_p_yes", "black_p_yes", "fairness_violation")
        ):
            raise ValueError(f"{model}: non-finite probability for {condition}.")
    key_heads = {tuple(item) for item in evaluation["key_heads"]}
    random_heads = {tuple(item) for item in evaluation["random_heads"]}
    if len(key_heads) != len(random_heads) or key_heads & random_heads:
        raise ValueError(f"{model}: invalid random-head control.")
    if evaluation.get("head_sweep_complete") is not True:
        raise ValueError(f"{model}: head sweep is incomplete.")

    row: Dict[str, Any] = {
        "model": model,
        "full_pairs": int(base_metadata["pairs"]),
        "selected_pairs": top_k,
        "selected_heads": len(key_heads),
        "selected_mlps": len(evaluation["key_mlps"]),
        "head_count_grid": " ".join(str(value) for value in evaluation["head_count_grid"]),
    }
    baseline = float(summary_rows["base"]["fairness_violation"])
    for condition in CONDITIONS:
        value = float(summary_rows[condition]["fairness_violation"])
        row[f"{condition}_fairness_violation"] = value
        row[f"{condition}_relative_reduction"] = (baseline - value) / baseline if baseline else 0.0
    row["key_head_advantage_vs_random"] = (
        row["key_heads_relative_reduction"] - row["random_heads_relative_reduction"]
    )
    return row


def build_head_curve(results_root: str, models: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
    output = []
    for model in models:
        path = os.path.join(
            results_root,
            model,
            "intervention_ablation",
            f"adult_race_yesno_high_gap_top{top_k}_seed_42",
            "evaluation",
            "head_count_summary.csv",
        )
        for row in _csv(path):
            output.append({"model": model, **row})
    return output


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_csv", default="results/adult_race_yesno_top100_all_models.csv")
    parser.add_argument(
        "--curve_csv", default="results/adult_race_yesno_top100_head_count_all_models.csv"
    )
    parser.add_argument("--audit_json", default="results/adult_race_yesno_top100_all_models.audit.json")
    args = parser.parse_args()
    rows = [audit_model(args.results_root, model, args.top_k) for model in args.models]
    curves = build_head_curve(args.results_root, args.models, args.top_k)
    _write(args.output_csv, rows)
    _write(args.curve_csv, curves)
    audit = {
        "status": "complete",
        "dataset": "Adult White/Black race counterfactual",
        "metric": "mean(abs(p_yes(black) - p_yes(white)))",
        "evaluation_protocol": "yes_no_income_gt_50k_v1",
        "models": list(args.models),
        "model_count": len(rows),
        "full_pairs_per_model": 46447,
        "selected_pairs_per_model": args.top_k,
        "checks": [
            "complete_metadata",
            "baseline_only_ranking",
            "strict_top_k_pair_alignment",
            "finite_probabilities",
            "equal_count_disjoint_random_heads",
            "complete_head_count_sweeps",
        ],
        "output_csv": os.path.abspath(args.output_csv),
        "curve_csv": os.path.abspath(args.curve_csv),
    }
    with open(args.audit_json, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


if __name__ == "__main__":
    main()
