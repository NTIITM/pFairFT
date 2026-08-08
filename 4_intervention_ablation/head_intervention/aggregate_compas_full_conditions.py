#!/usr/bin/env python
"""Combine independently evaluated COMPAS conditions into one audited result directory."""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
from typing import Any, Dict, List


CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")
SHARED_METADATA_KEYS = (
    "dataset_path",
    "model_path",
    "checkpoint_type",
    "model_type",
    "adapter_family",
    "head_activation_kind",
    "mlp_surface",
    "metric",
    "seed",
    "records",
    "pairs",
    "batch_size",
    "key_heads",
    "random_heads",
    "key_mlps",
    "selected_heads_path",
    "head_results_path",
    "selected_mlp_path",
    "mlp_embeddings_path",
)


def load_rows(path: str, condition: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("condition") != condition for row in rows):
        raise ValueError(f"Invalid condition rows in {path}")
    for row in rows:
        for column in (
            "white_p_yes",
            "black_p_yes",
            "black_minus_white_gap",
            "fairness_violation",
        ):
            if not math.isfinite(float(row[column])):
                raise ValueError(f"Non-finite {column} in {path}")
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    violations = [float(row["fairness_violation"]) for row in rows]
    signed = [float(row["black_minus_white_gap"]) for row in rows]
    return {
        "condition": rows[0]["condition"],
        "valid_pairs": len(rows),
        "fairness_violation": statistics.fmean(violations),
        "black_minus_white_gap_mean": statistics.fmean(signed),
        "fairness_violation_median": statistics.median(violations),
        "fairness_violation_max": max(violations),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final_dir", required=True)
    parser.add_argument(
        "--condition_dir",
        action="append",
        required=True,
        metavar="CONDITION=PATH",
    )
    args = parser.parse_args()

    sources = {}
    for value in args.condition_dir:
        condition, separator, path = value.partition("=")
        if not separator or condition not in CONDITIONS:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        sources[condition] = os.path.abspath(path)
    missing = [condition for condition in CONDITIONS if condition not in sources]
    if missing:
        raise ValueError(f"Missing condition directories: {missing}")

    metadata_by_condition = {}
    rows_by_condition = {}
    pair_ids = None
    reference_metadata = None
    for condition in CONDITIONS:
        source = sources[condition]
        with open(os.path.join(source, "metadata.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if condition not in metadata.get("completed_conditions", []):
            raise ValueError(f"Condition {condition} missing from completed_conditions: {source}")
        rows = load_rows(os.path.join(source, f"per_pair_{condition}.csv"), condition)
        current_pair_ids = [int(row["pair_id"]) for row in rows]
        if len(current_pair_ids) != int(metadata["pairs"]):
            raise ValueError(f"Pair count mismatch for {condition}")
        if pair_ids is None:
            pair_ids = current_pair_ids
            reference_metadata = metadata
        elif current_pair_ids != pair_ids:
            raise ValueError(f"Pair order mismatch for {condition}")
        for key in SHARED_METADATA_KEYS:
            if metadata.get(key) != reference_metadata.get(key):
                raise ValueError(f"Metadata mismatch for {condition}: {key}")
        metadata_by_condition[condition] = metadata
        rows_by_condition[condition] = rows

    summaries = [summarize(rows_by_condition[condition]) for condition in CONDITIONS]
    baseline = float(summaries[0]["fairness_violation"])
    for row in summaries:
        reduction = baseline - float(row["fairness_violation"])
        row["absolute_reduction_from_base"] = reduction
        row["relative_reduction_from_base"] = reduction / baseline if baseline else 0.0

    os.makedirs(args.final_dir, exist_ok=True)
    for condition in CONDITIONS:
        source_csv = os.path.join(sources[condition], f"per_pair_{condition}.csv")
        target_csv = os.path.join(args.final_dir, f"per_pair_{condition}.csv")
        if os.path.abspath(source_csv) != os.path.abspath(target_csv):
            shutil.copy2(source_csv, target_csv)
    write_csv(os.path.join(args.final_dir, "summary.csv"), summaries)

    final_metadata = dict(reference_metadata)
    final_metadata["status"] = "complete"
    final_metadata["conditions"] = list(CONDITIONS)
    final_metadata["completed_conditions"] = list(CONDITIONS)
    final_metadata["condition_sources"] = sources
    with open(os.path.join(args.final_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(final_metadata, handle, indent=2)


if __name__ == "__main__":
    main()
