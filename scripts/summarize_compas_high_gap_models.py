#!/usr/bin/env python
"""Audit and compare full COMPAS results with base-selected high-gap subsets."""

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Sequence


CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")
DEFAULT_MODELS = (
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
DOCUMENTED_STALE_FULL_INTERVENTION_MODELS = {"Llama-3.2-3B-Instruct"}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_by_condition(path: str) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    summary = {row["condition"]: row for row in rows}
    if set(summary) != set(CONDITIONS):
        raise ValueError(f"Incomplete conditions in {path}")
    return summary


def audit_pair_outputs(
    evaluation_dir: str,
    expected_pair_ids: Sequence[int],
) -> None:
    expected = list(expected_pair_ids)
    for condition in CONDITIONS:
        path = os.path.join(evaluation_dir, f"per_pair_{condition}.csv")
        rows = read_csv(path)
        pair_ids = [int(row["pair_id"]) for row in rows]
        if pair_ids != expected:
            raise ValueError(f"Pair IDs/order differ for {condition}: {path}")
        if any(row["condition"] != condition for row in rows):
            raise ValueError(f"Condition column mismatch: {path}")
        for row in rows:
            for column in (
                "white_p_yes",
                "black_p_yes",
                "black_minus_white_gap",
                "fairness_violation",
            ):
                if not math.isfinite(float(row[column])):
                    raise ValueError(f"Non-finite {column}: {path}")


def audit_selected_dataset(path: str, expected_pair_ids: Sequence[int]) -> None:
    records = read_json(path)
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["pair_id"]), []).append(record)
    if set(grouped) != set(expected_pair_ids) or len(records) != 2 * len(grouped):
        raise ValueError(f"Selected dataset does not match selected pair IDs: {path}")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {str(row["race"]).lower() for row in pair} != {"white", "black"}:
            raise ValueError(f"Invalid white/black pair {pair_id}: {path}")
        first, second = pair
        if int(first["matched_id"]) != int(second["id"]) or int(second["matched_id"]) != int(first["id"]):
            raise ValueError(f"Non-reciprocal matched_id for pair {pair_id}: {path}")


def audit_model(results_root: str, model: str, experiment_name: str) -> Dict[str, Any]:
    model_root = os.path.join(results_root, model, "intervention_ablation")
    full_dir = os.path.join(model_root, "compas_full_seed_42")
    high_gap_dir = os.path.join(model_root, experiment_name)
    evaluation_dir = os.path.join(high_gap_dir, "evaluation")

    full_metadata = read_json(os.path.join(full_dir, "metadata.json"))
    selection_metadata = read_json(os.path.join(high_gap_dir, "selection_metadata.json"))
    evaluation_metadata = read_json(os.path.join(evaluation_dir, "metadata.json"))
    for label, metadata in (
        ("full", full_metadata),
        ("selection", selection_metadata),
        ("evaluation", evaluation_metadata),
    ):
        if metadata.get("status") != "complete":
            raise ValueError(f"{model}: {label} metadata is not complete")

    if selection_metadata.get("selection_condition") != "base":
        raise ValueError(f"{model}: high-gap selection is not based on base outputs")
    if selection_metadata.get("intervention_outputs_used_for_selection") is not False:
        raise ValueError(f"{model}: intervention outputs leaked into sample selection")
    if int(evaluation_metadata["seed"]) != 42 or int(evaluation_metadata["batch_size"]) != 1:
        raise ValueError(f"{model}: expected seed 42 and batch size 1")

    key_heads = {tuple(item) for item in evaluation_metadata["key_heads"]}
    random_heads = {tuple(item) for item in evaluation_metadata["random_heads"]}
    if len(key_heads) != len(random_heads) or key_heads & random_heads:
        raise ValueError(f"{model}: random heads must be equal-count and disjoint")
    if key_heads != {tuple(item) for item in full_metadata["key_heads"]}:
        raise ValueError(f"{model}: high-gap key heads differ from full-run metadata")
    if list(evaluation_metadata["key_mlps"]) != list(full_metadata["key_mlps"]):
        raise ValueError(f"{model}: high-gap key MLPs differ from full-run metadata")

    selected_pair_ids = [int(value) for value in selection_metadata["selected_pair_ids"]]
    if len(selected_pair_ids) != int(selection_metadata["selected_pairs"]):
        raise ValueError(f"{model}: selected pair count mismatch")
    if int(evaluation_metadata["pairs"]) != len(selected_pair_ids):
        raise ValueError(f"{model}: evaluation pair count mismatch")
    audit_selected_dataset(selection_metadata["selected_dataset_path"], selected_pair_ids)
    audit_pair_outputs(evaluation_dir, selected_pair_ids)

    full_summary = summary_by_condition(os.path.join(full_dir, "summary.csv"))
    high_gap_summary = summary_by_condition(os.path.join(evaluation_dir, "summary.csv"))
    predicted_rows = {
        row["condition"]: row
        for row in read_csv(os.path.join(high_gap_dir, "top_k_intervention_summary.csv"))
        if int(row["top_k"]) == len(selected_pair_ids)
    }
    posthoc_differences = {}
    for condition in CONDITIONS:
        predicted = float(predicted_rows[condition]["fairness_violation"])
        actual = float(high_gap_summary[condition]["fairness_violation"])
        posthoc_differences[condition] = abs(predicted - actual)
    rerun_matches_posthoc = all(value <= 1e-12 for value in posthoc_differences.values())
    if not rerun_matches_posthoc and model not in DOCUMENTED_STALE_FULL_INTERVENTION_MODELS:
        raise ValueError(f"{model}: rerun differs from post-hoc full-result slice")

    ranking = read_csv(os.path.join(high_gap_dir, "base_gap_ranking.csv"))
    selected_gaps = [float(row["base_fairness_violation"]) for row in ranking[:len(selected_pair_ids)]]
    full_base = float(full_summary["base"]["fairness_violation"])
    high_gap_base = float(high_gap_summary["base"]["fairness_violation"])
    output: Dict[str, Any] = {
        "model": model,
        "reported_result_source": "actual_high_gap_rerun",
        "rerun_matches_posthoc_full_output": rerun_matches_posthoc,
        "posthoc_max_abs_difference": max(posthoc_differences.values()),
        "full_pairs": int(full_metadata["pairs"]),
        "selected_pairs": len(selected_pair_ids),
        "selected_fraction": len(selected_pair_ids) / int(full_metadata["pairs"]),
        "selected_gap_min": min(selected_gaps),
        "selected_gap_max": max(selected_gaps),
        "base_gap_enrichment": high_gap_base / full_base,
        "key_head_count": len(key_heads),
        "random_head_count": len(random_heads),
        "key_mlp_count": len(evaluation_metadata["key_mlps"]),
        "seed": int(evaluation_metadata["seed"]),
        "batch_size": int(evaluation_metadata["batch_size"]),
    }
    for condition in CONDITIONS:
        full_row = full_summary[condition]
        high_gap_row = high_gap_summary[condition]
        output[f"full_{condition}_fairness_violation"] = float(full_row["fairness_violation"])
        output[f"full_{condition}_relative_reduction"] = float(full_row["relative_reduction_from_base"])
        output[f"high_gap_{condition}_fairness_violation"] = float(high_gap_row["fairness_violation"])
        output[f"high_gap_{condition}_relative_reduction"] = float(high_gap_row["relative_reduction_from_base"])
    output["high_gap_key_head_advantage_vs_random"] = (
        output["high_gap_random_heads_fairness_violation"]
        - output["high_gap_key_heads_fairness_violation"]
    )
    output["high_gap_key_head_reduction_advantage_vs_random"] = (
        output["high_gap_key_heads_relative_reduction"]
        - output["high_gap_random_heads_relative_reduction"]
    )
    return output


def build_curve_rows(
    results_root: str,
    models: Sequence[str],
    experiment_name: str,
) -> List[Dict[str, Any]]:
    output = []
    expected_k = None
    for model in models:
        path = os.path.join(
            results_root,
            model,
            "intervention_ablation",
            experiment_name,
            "top_k_intervention_summary.csv",
        )
        grouped: Dict[int, Dict[str, Dict[str, str]]] = {}
        for row in read_csv(path):
            grouped.setdefault(int(row["top_k"]), {})[row["condition"]] = row
        if expected_k is None:
            expected_k = sorted(grouped)
        elif sorted(grouped) != expected_k:
            raise ValueError(f"{model}: top-k curve grid differs from other models")
        for top_k in sorted(grouped):
            conditions = grouped[top_k]
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"{model}: incomplete top-k={top_k} conditions")
            curve_row: Dict[str, Any] = {"model": model, "top_k": top_k}
            for condition in CONDITIONS:
                curve_row[f"{condition}_fairness_violation"] = float(
                    conditions[condition]["fairness_violation"]
                )
                curve_row[f"{condition}_relative_reduction"] = float(
                    conditions[condition]["relative_reduction_from_base"]
                )
            curve_row["key_head_reduction_advantage_vs_random"] = (
                curve_row["key_heads_relative_reduction"]
                - curve_row["random_heads_relative_reduction"]
            )
            output.append(curve_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--experiment_name",
        default="compas_high_gap_top100_seed_42",
    )
    parser.add_argument(
        "--output_csv",
        default="results/compas_high_gap_top100_all_models.csv",
    )
    parser.add_argument(
        "--audit_json",
        default="results/compas_high_gap_top100_all_models.audit.json",
    )
    parser.add_argument(
        "--curve_csv",
        default="results/compas_high_gap_top_k_all_models.csv",
    )
    args = parser.parse_args()

    rows = [audit_model(args.results_root, model, args.experiment_name) for model in args.models]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    curve_models = [
        row["model"] for row in rows if row["rerun_matches_posthoc_full_output"]
    ]
    curve_rows = build_curve_rows(args.results_root, curve_models, args.experiment_name)
    with open(args.curve_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    audit = {
        "status": "complete",
        "experiment_name": args.experiment_name,
        "selection_protocol": "exp2_style_base_gap_ranking",
        "selection_condition": "base",
        "intervention_outputs_used_for_selection": False,
        "models": list(args.models),
        "model_count": len(rows),
        "rerun_matches_posthoc_models": curve_models,
        "documented_stale_full_intervention_models": [
            row["model"] for row in rows if not row["rerun_matches_posthoc_full_output"]
        ],
        "curve_models": curve_models,
        "curve_k": sorted({row["top_k"] for row in curve_rows}),
        "checks": [
            "complete_metadata",
            "strict_white_black_pairs",
            "reciprocal_matched_ids",
            "identical_pair_ids_and_order_across_conditions",
            "finite_probabilities",
            "equal_count_disjoint_random_heads",
            "base_only_sample_selection",
            "actual_rerun_uses_full_metadata_key_components",
            "post_hoc_match_or_documented_stale_full_intervention_output",
        ],
        "output_csv": os.path.abspath(args.output_csv),
        "curve_csv": os.path.abspath(args.curve_csv),
    }
    with open(args.audit_json, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


if __name__ == "__main__":
    main()
