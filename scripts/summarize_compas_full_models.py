#!/usr/bin/env python
"""Audit and summarize complete multi-model COMPAS intervention results."""

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List


CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")
DEFAULT_MODELS = (
    "Llama-3.2-1B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "DeepSeek-V2-Lite-Chat",
    "JetMoE-8B-Chat",
    "OLMoE-1B-7B-0924-Instruct",
)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_model(result_dir: str, model: str) -> Dict[str, Any]:
    metadata_path = os.path.join(result_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError(f"{model}: metadata status is not complete")
    if metadata.get("completed_conditions") != list(CONDITIONS):
        raise ValueError(f"{model}: unexpected completed_conditions")
    if int(metadata.get("batch_size", 0)) != 1:
        raise ValueError(f"{model}: formal result must use batch_size=1")

    pair_ids = None
    for condition in CONDITIONS:
        rows = read_csv(os.path.join(result_dir, f"per_pair_{condition}.csv"))
        current_pair_ids = [int(row["pair_id"]) for row in rows]
        if len(current_pair_ids) != int(metadata["pairs"]):
            raise ValueError(f"{model}: pair count mismatch for {condition}")
        if pair_ids is None:
            pair_ids = current_pair_ids
        elif current_pair_ids != pair_ids:
            raise ValueError(f"{model}: pair order mismatch for {condition}")
        if any(row.get("condition") != condition for row in rows):
            raise ValueError(f"{model}: condition column mismatch for {condition}")
        for row in rows:
            for column in (
                "white_p_yes",
                "black_p_yes",
                "black_minus_white_gap",
                "fairness_violation",
            ):
                if not math.isfinite(float(row[column])):
                    raise ValueError(f"{model}: non-finite {column} for {condition}")

    key_heads = {tuple(item) for item in metadata["key_heads"]}
    random_heads = {tuple(item) for item in metadata["random_heads"]}
    if key_heads & random_heads:
        raise ValueError(f"{model}: random heads overlap key heads")
    if len(key_heads) != len(random_heads):
        raise ValueError(f"{model}: random/key head counts differ")

    with open(metadata["selected_heads_path"], "r", encoding="utf-8") as handle:
        selected_head_records = json.load(handle)
    selected_heads = {
        (int(item["layer"]), int(item["head"])) for item in selected_head_records
    }
    if selected_heads != key_heads:
        raise ValueError(f"{model}: metadata key heads differ from elbow selection")
    with open(metadata["selected_mlp_path"], "r", encoding="utf-8") as handle:
        selected_mlp_records = json.load(handle)
    selected_mlps = [int(item["layer"]) for item in selected_mlp_records]
    if selected_mlps != [int(layer) for layer in metadata["key_mlps"]]:
        raise ValueError(f"{model}: metadata key MLPs differ from elbow selection")

    summary_rows = read_csv(os.path.join(result_dir, "summary.csv"))
    summary = {row["condition"]: row for row in summary_rows}
    if set(summary) != set(CONDITIONS):
        raise ValueError(f"{model}: incomplete summary conditions")

    output: Dict[str, Any] = {
        "model": model,
        "pairs": int(metadata["pairs"]),
        "key_head_count": len(key_heads),
        "random_head_count": len(random_heads),
        "key_mlp_count": len(metadata["key_mlps"]),
        "key_mlp_layers": ";".join(str(layer) for layer in metadata["key_mlps"]),
        "seed": int(metadata["seed"]),
        "batch_size": int(metadata["batch_size"]),
    }
    for condition in CONDITIONS:
        row = summary[condition]
        output[f"{condition}_fairness_violation"] = float(row["fairness_violation"])
        output[f"{condition}_signed_gap"] = float(row["black_minus_white_gap_mean"])
        output[f"{condition}_relative_reduction"] = float(
            row["relative_reduction_from_base"]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--experiment_name", default="compas_full_seed_42")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--output_csv",
        default="results/compas_full_seed_42_all_models.csv",
    )
    parser.add_argument(
        "--audit_json",
        default="results/compas_full_seed_42_all_models.audit.json",
    )
    args = parser.parse_args()

    rows = []
    result_dirs = {}
    for model in args.models:
        result_dir = os.path.join(
            args.results_root,
            model,
            "intervention_ablation",
            args.experiment_name,
        )
        rows.append(audit_model(result_dir, model))
        result_dirs[model] = os.path.abspath(result_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "complete",
        "experiment_name": args.experiment_name,
        "metric": "mean(abs(p_yes(black) - p_yes(white)))",
        "models": list(args.models),
        "model_count": len(rows),
        "pairs_per_condition": sorted({row["pairs"] for row in rows}),
        "seed": sorted({row["seed"] for row in rows}),
        "batch_size": sorted({row["batch_size"] for row in rows}),
        "checks": [
            "complete_metadata",
            "condition_row_count",
            "pair_id_order",
            "finite_probabilities",
            "equal_random_key_head_count",
            "zero_random_key_overlap",
            "heads_match_elbow_selection",
            "mlps_match_elbow_selection",
        ],
        "result_directories": result_dirs,
    }
    with open(args.audit_json, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


if __name__ == "__main__":
    main()
