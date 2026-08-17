#!/usr/bin/env python3
"""Validate and materialize the current Meta-Llama-3-8B Figure 5 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_adapter_dir(path: Path) -> Path:
    """Normalize a checkpoint root or its final_model directory."""
    path = path.expanduser().resolve()
    if (path / "adapter_model.safetensors").is_file():
        return path
    final_model = path / "final_model"
    if (final_model / "adapter_model.safetensors").is_file():
        return final_model
    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors under checkpoint {path}"
    )


def validate_discrim_csv(path: Path, expected_prompt: str | None) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2520:
        raise ValueError(f"{path}: expected 2520 rows, found {len(rows)}")
    ids = [int(row["sample_id"]) for row in rows]
    if len(set(ids)) != 2520:
        raise ValueError(f"{path}: sample_id values are not unique")
    by_id = {int(row["sample_id"]): row for row in rows}
    pairs = set()
    qid_counts: dict[int, int] = {}
    for sample_id, row in by_id.items():
        matched_id = int(row["matched_id"])
        if matched_id not in by_id:
            raise ValueError(f"{path}: missing matched_id={matched_id}")
        pairs.add(tuple(sorted((sample_id, matched_id))))
        qid = int(row["decision_question_id"])
        qid_counts[qid] = qid_counts.get(qid, 0) + 1
        if expected_prompt is not None and row.get("prompt_type") != expected_prompt:
            raise ValueError(
                f"{path}: expected prompt_type={expected_prompt}, found {row.get('prompt_type')!r}"
            )
    if len(pairs) != 1260 or len(qid_counts) != 70 or set(qid_counts.values()) != {36}:
        raise ValueError(
            f"{path}: expected 1260 pairs and 70 QIDs x 36 rows; "
            f"found {len(pairs)} pairs and {len(qid_counts)} QIDs"
        )
    return {"rows": len(rows), "pairs": len(pairs), "qids": len(qid_counts)}


def copy_with_record(source: Path, destination: Path, records: list[dict], role: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")
    records.append(
        {
            "role": role,
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "sha256": source_hash,
            "bytes": source.stat().st_size,
        }
    )


def validate_activation_geometry(path: Path, selected_heads: set[tuple[int, int]]) -> dict:
    metadata_path = path / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected = metadata["selected_head"]
    head = (int(selected["layer"]), int(selected["head"]))
    if head not in selected_heads:
        raise ValueError(f"Activation-geometry head {head} is not in the current selected-head set")
    if metadata.get("head_selection_domain") != "resume":
        raise ValueError("Panel-h head must be selected on Resume data")
    if int(metadata.get("geometry_schema_version", 0)) < 2:
        raise ValueError("Panel-h geometry uses the obsolete sensitive-residual/PC1 schema")
    direction = np.asarray(metadata["sensitive_direction"], dtype=np.float64)
    if direction.ndim != 1 or not np.isfinite(direction).all():
        raise ValueError("Panel-h sensitive direction is invalid")
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6):
        raise ValueError("Panel-h sensitive direction is not unit normalized")

    expected = {
        "resume": {"rows": 400, "per_condition": 200, "per_group": 100},
        "discrim": {"rows": 5040, "per_condition": 2520, "per_group": 1260},
    }
    heads_dir = Path(metadata["heads_dir"]).expanduser().resolve()
    head_results = heads_dir / "results.pkl"
    pfairft_adapter = Path(metadata["pfairft_adapter"]).expanduser().resolve()
    adapter_model = pfairft_adapter / "adapter_model.safetensors"
    if not head_results.is_file() or not adapter_model.is_file():
        raise FileNotFoundError("Panel-h provenance inputs are missing")
    validation = {
        "selected_head": {"layer": head[0], "head": head[1]},
        "head_results": str(head_results),
        "head_results_sha256": sha256(head_results),
        "pfairft_adapter": str(pfairft_adapter),
        "pfairft_adapter_model_sha256": sha256(adapter_model),
    }
    candidate_path = path / "resume_candidate_activations.npz"
    with np.load(candidate_path) as candidate:
        if candidate["base"].shape != (200, len(selected_heads), direction.size):
            raise ValueError(f"{candidate_path}: unexpected Base candidate shape")
        if candidate["pfairft"].shape != candidate["base"].shape:
            raise ValueError(f"{candidate_path}: PFairFT candidate shape differs")
        if not np.isfinite(candidate["base"]).all() or not np.isfinite(candidate["pfairft"]).all():
            raise ValueError(f"{candidate_path}: non-finite candidate activations")
    for domain, counts in expected.items():
        csv_path = path / f"{domain}_geometry.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != counts["rows"]:
            raise ValueError(f"{csv_path}: expected {counts['rows']} rows, found {len(rows)}")
        for condition in ("base", "pfairft"):
            condition_rows = [row for row in rows if row["condition"] == condition]
            if len(condition_rows) != counts["per_condition"]:
                raise ValueError(f"{csv_path}: incomplete {condition} rows")
            group_counts = {
                group: sum(row["group"] == group for row in condition_rows)
                for group in ("Black", "White")
            }
            if set(group_counts.values()) != {counts["per_group"]}:
                raise ValueError(f"{csv_path}: unexpected {condition} group counts {group_counts}")
            coordinates = np.asarray(
                [
                    [float(row["orthogonal_pc1"]), float(row["orthogonal_pc2"])]
                    for row in condition_rows
                ],
                dtype=np.float64,
            )
            if not np.isfinite(coordinates).all():
                raise ValueError(f"{csv_path}: non-finite projected coordinates")
        geometry = metadata["geometry"][domain]
        for key in ("direction_pc1_dot", "direction_pc2_dot", "pc1_pc2_dot"):
            if abs(float(geometry[key])) > 1e-6:
                raise ValueError(f"{domain}: invalid orthogonal PCA basis ({key})")
        ratios = np.asarray(geometry["explained_variance_ratio"], dtype=np.float64)
        if ratios.shape != (2,) or not np.isfinite(ratios).all() or np.any(ratios < 0):
            raise ValueError(f"{domain}: invalid PC1/PC2 explained-variance ratios")
        raw_path = path / f"{domain}_selected_activations.npz"
        with np.load(raw_path) as raw:
            if raw["base"].shape != raw["pfairft"].shape:
                raise ValueError(f"{raw_path}: Base/PFairFT activation shapes differ")
            if raw["base"].shape[0] != counts["per_condition"]:
                raise ValueError(f"{raw_path}: unexpected selected-activation row count")
            if not np.isfinite(raw["base"]).all() or not np.isfinite(raw["pfairft"]).all():
                raise ValueError(f"{raw_path}: non-finite selected activations")
            if raw["sample_id"].shape != (counts["per_condition"],):
                raise ValueError(f"{raw_path}: record identifiers do not align")
        validation[domain] = counts
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        default=str(ROOT.parents[1]),
    )
    parser.add_argument(
        "--model_name",
        default="Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--result_root",
        default=None,
        help="Explicit isolated Figure 5 result root.",
    )
    parser.add_argument("--output_dir", default=str(ROOT / "data" / "current"))
    parser.add_argument(
        "--global_csv",
        default=None,
        help="Fresh Discrim-Eval CSV generated from the Global checkpoint.",
    )
    parser.add_argument(
        "--head_analysis_dir",
        default=None,
        help="Head-analysis directory corresponding to the selected Global checkpoint.",
    )
    parser.add_argument(
        "--global_adapter",
        default=None,
        help="Global LoRA checkpoint root or final_model directory.",
    )
    parser.add_argument(
        "--activation_geometry_dir",
        default=None,
        help="Validated Base/PFairFT activation geometry for panel h.",
    )
    parser.add_argument("--context_results", default=None)
    parser.add_argument("--selected_heads", default=None)
    parser.add_argument("--heads_results", default=None)
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    isolated_layout = args.result_root is not None
    result_root = (
        Path(args.result_root).expanduser().resolve()
        if args.result_root
        else project / "results" / args.model_name
    )
    downstream = result_root / ("downstream" if isolated_layout else "downstream_evaluation")
    global_csv = Path(args.global_csv).expanduser().resolve() if args.global_csv else (
        downstream / "discrim_global_oldtarget_raw_summary_full_3epoch.csv"
    )
    head_root = (
        Path(args.head_analysis_dir).expanduser().resolve()
        if args.head_analysis_dir
        else result_root / "figure5_analysis" / "head_conditions_global_oldtarget_raw"
    )
    global_adapter = resolve_adapter_dir(
        Path(args.global_adapter)
        if args.global_adapter
        else result_root / "global_lora_oldtarget_raw_summary_full_3epoch"
    )
    output = Path(args.output_dir).resolve()
    activation_geometry = (
        Path(args.activation_geometry_dir).expanduser().resolve()
        if args.activation_geometry_dir
        else result_root / "figure5_analysis" / "activation_geometry"
    )
    records: list[dict] = []

    csv_sources = {
        "original": (
            downstream
            / (
                "discrim_baseline.csv"
                if isolated_layout
                else "discrim_baseline_pkfair_3epoch_fresh.csv"
            ),
            "prompt",
        ),
        "debiased_prompt": (
            downstream
            / (
                "discrim_debiased_prompt.csv"
                if isolated_layout
                else "discrim_baseline_debiased_prompt_figure5_fresh.csv"
            ),
            "debiased_prompt",
        ),
        "global": (global_csv, "prompt"),
        "pfairft": (
            downstream / ("discrim_pfairft.csv" if isolated_layout else "discrim_pkfair_kl_pkfair_3epoch_fresh.csv"),
            "prompt",
        ),
        "pfairft_kl": (
            downstream / ("discrim_pfairft_kl.csv" if isolated_layout else "discrim_pkfair_pkfair_3epoch_fresh.csv"),
            "prompt",
        ),
        "pfairft_ce": (
            downstream / ("discrim_pfairft_ce.csv" if isolated_layout else "discrim_pkfair_ce_pkfair_3epoch_fresh.csv"),
            "prompt",
        ),
        "inference_time": (
            result_root
            / ("inference_time" if isolated_layout else "inference_time_figure5")
            / "discrim_partial.csv",
            None,
        ),
    }
    validation = {}
    for label, (source, prompt_type) in csv_sources.items():
        validation[label] = validate_discrim_csv(source, prompt_type)
        destination = output / "downstream" / f"{label}.csv"
        copy_with_record(source, destination, records, f"downstream:{label}")
        metadata_source = Path(str(source) + ".metadata.json")
        if not metadata_source.exists() and label == "inference_time":
            metadata_source = source.parent / "discrim_metadata.json"
        if metadata_source.exists():
            if label == "global":
                with metadata_source.open("r", encoding="utf-8") as metadata_handle:
                    global_metadata = json.load(metadata_handle)
                recorded_adapter = global_metadata.get("adapter_path")
                if not recorded_adapter:
                    raise ValueError(f"{metadata_source}: missing adapter_path")
                if Path(recorded_adapter).expanduser().resolve() != global_adapter:
                    raise ValueError(
                        f"{source}: metadata adapter {recorded_adapter!r} does not match "
                        f"the requested Global checkpoint {global_adapter}"
                    )
            copy_with_record(
                metadata_source,
                output / "downstream" / f"{label}.metadata.json",
                records,
                f"downstream_metadata:{label}",
            )

    selected_source = (
        Path(args.selected_heads).expanduser().resolve()
        if args.selected_heads
        else result_root
        / "sensitive_heads_moefreeze_top100_summary_only_current_ranking"
        / "selected_heads_elbow.json"
    )
    with selected_source.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    selected_count = len(selected)
    if selected_count <= 0:
        raise ValueError("No current sensitive heads were selected")
    selected_head_set = {(int(row["layer"]), int(row["head"])) for row in selected}
    copy_with_record(
        selected_source,
        output / "heads" / "selected_heads_elbow.json",
        records,
        "selected_heads",
    )

    with (head_root / "metadata.json").open("r", encoding="utf-8") as handle:
        head_metadata = json.load(handle)
    recorded_head_global = Path(
        head_metadata["conditions"]["global"]["adapter_path"]
    ).expanduser().resolve()
    if recorded_head_global != global_adapter:
        raise ValueError(
            f"{head_root}: head analysis uses {recorded_head_global}, "
            f"expected {global_adapter}"
        )
    selected_qid = int(head_metadata["selected_comparison_qid"])
    if int(head_metadata["num_selected_heads"]) != selected_count:
        raise ValueError("Head-analysis metadata does not use the current selected-head set.")
    if int(head_metadata["batch_size"]) != 1:
        raise ValueError("Head-analysis metadata must report batch_size=1.")
    candidate_count = int(head_metadata["candidate_count"])
    if candidate_count <= 0 or len(head_metadata["candidate_qids"]) != candidate_count:
        raise ValueError("Head-analysis metadata must cover every eligible QID candidate.")

    array_sources = {
        "panel_b/debiased_prompt_md.npy": head_root
        / "panel_b_qid90"
        / "debiased_prompt_md.npy",
        "panel_b/debiased_prompt_with_context_md.npy": head_root
        / "panel_b_qid90"
        / "debiased_prompt_with_context_md.npy",
        "panel_e_f/original_md.npy": head_root
        / "candidates"
        / f"qid{selected_qid}"
        / "original_md.npy",
        "panel_e_f/global_md.npy": head_root
        / "candidates"
        / f"qid{selected_qid}"
        / "global_md.npy",
        "panel_e_f/pfairft_md.npy": head_root
        / "candidates"
        / f"qid{selected_qid}"
        / "pfairft_md.npy",
    }
    for relative, source in array_sources.items():
        values = np.load(source)
        if values.shape != (32, 32) or not np.isfinite(values).all():
            raise ValueError(f"{source}: expected finite (32, 32) array, found {values.shape}")
        copy_with_record(source, output / "heads" / relative, records, f"head_array:{relative}")

    for name in ("metadata.json", "qid_selection.csv", "output_level_candidates.csv"):
        copy_with_record(
            head_root / name,
            output / "heads" / name,
            records,
            f"head_analysis:{name}",
        )

    context_source = (
        Path(args.context_results).expanduser().resolve()
        if args.context_results
        else ROOT / "data" / "exp1" / "context" / "context_results_Meta-Llama-3-8B-Instruct.json"
    )
    copy_with_record(
        context_source,
        output / "context" / "panel_c_context_results.json",
        records,
        "panel_c_current_context",
    )

    activation_validation = validate_activation_geometry(
        activation_geometry, selected_head_set
    )
    for name in (
        "metadata.json",
        "head_scores.csv",
        "resume_geometry.csv",
        "discrim_geometry.csv",
        "resume_selected_activations.npz",
        "discrim_selected_activations.npz",
        "resume_candidate_activations.npz",
    ):
        copy_with_record(
            activation_geometry / name,
            output / "activation_geometry" / name,
            records,
            f"activation_geometry:{name}",
        )
    head_results_source = Path(activation_validation["head_results"])
    if args.heads_results:
        requested_head_results = Path(args.heads_results).expanduser().resolve()
        if head_results_source.resolve() != requested_head_results:
            raise ValueError(
                f"Activation geometry uses {head_results_source}, expected {requested_head_results}"
            )
    copy_with_record(
        head_results_source,
        output / "activation_geometry" / "head_results.pkl",
        records,
        "activation_geometry:sensitive_direction_source",
    )

    training_timing_path = global_adapter.parent / "training_timing.json"
    training_timing = None
    if training_timing_path.is_file():
        with training_timing_path.open("r", encoding="utf-8") as handle:
            training_timing = json.load(handle)
    manifest = {
        "model_name": args.model_name,
        "results_root": str(result_root),
        "global_csv": str(global_csv),
        "head_analysis_dir": str(head_root),
        "global_checkpoint": {
            "path": str(global_adapter),
            "adapter_model": str(global_adapter / "adapter_model.safetensors"),
            "adapter_model_sha256": sha256(global_adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(global_adapter / "adapter_config.json")
            if (global_adapter / "adapter_config.json").is_file()
            else None,
            "training_timing": training_timing,
        },
        "selected_comparison_qid": selected_qid,
        "strict_global_balance_found": bool(head_metadata["strict_global_balance_found"]),
        "qid_selection_mode": head_metadata["selection_mode"],
        "panel_b_qid": 90,
        "method_mapping": {
            "PFairFT": "fairness_kl",
            "PFairFT-KL": "fairness_kl_ce",
            "PFairFT-CE": "fairness_ce",
        },
        "panel_c_policy": "recomputed from the current ModelScope checkpoint",
        "panel_h": {
            "main_domain": "resume",
            "transfer_reference_domain": "discrim",
            "head_selection_domain": "resume",
            **activation_validation,
        },
        "validation": validation,
        "artifacts": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Materialized {len(records)} Figure 5 inputs under {output}")
    print(f"Selected comparison QID: {selected_qid}")


if __name__ == "__main__":
    main()
