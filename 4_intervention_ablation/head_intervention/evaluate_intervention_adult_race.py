#!/usr/bin/env python
"""Evaluate Resume race-component interventions with Adult's Yes/No protocol."""

import argparse
import csv
import json
import math
import os
import pickle
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hook import get_last_token_indices_safe, remove_intervention_hooks  # noqa: E402
from model_adapter import get_model_adapter  # noqa: E402
from probability import (  # noqa: E402
    NO_CANDIDATES,
    YES_CANDIDATES,
    get_target_token_ids,
)
from prompt import format_prompt_for_model, resolve_model_type  # noqa: E402
from sampling import load_discrim_eval_pairs  # noqa: E402
from util import (  # noqa: E402
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
    get_non_sensitive_heads_from_results,
    get_sensitive_heads_sorted_by_heatmap,
    load_intervention_results,
)


CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_head_embeddings(values: Dict[Any, Any]) -> Dict[Tuple[int, int], Any]:
    return {
        (int(key[0]), int(key[1])): value
        for key, value in values.items()
        if isinstance(key, (tuple, list)) and len(key) >= 2
    }


def _normalize_mlp_embeddings(values: Dict[Any, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in values.items()}


def load_adult_pairs(
    dataset_path: str, max_pairs: int = 0
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    data, pairs = load_discrim_eval_pairs(dataset_path)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
        selected_ids = {sample_id for pair in pairs for sample_id in pair}
        data = [sample for sample in data if int(sample["id"]) in selected_ids]
    samples = {int(sample["id"]): sample for sample in data}
    if not pairs:
        raise ValueError("No Adult pairs were selected.")
    for first_id, second_id in pairs:
        pair_samples = [samples[first_id], samples[second_id]]
        if {str(sample.get("race", "")).lower() for sample in pair_samples} != {
            "white",
            "black",
        }:
            raise ValueError(f"Pair {first_id}/{second_id} is not White/Black.")
        if {sample.get("prompt_type") for sample in pair_samples} != {
            "fact",
            "counterfactual",
        }:
            raise ValueError(f"Pair {first_id}/{second_id} is not fact/counterfactual.")
    return data, pairs


def select_components(
    sensitive_heads_dir: str,
    selected_mlp_path: str,
    mlp_embeddings_path: str,
    seed: int,
) -> Dict[str, Any]:
    selected_heads_path = os.path.join(sensitive_heads_dir, "selected_heads_elbow.json")
    head_results_path = os.path.join(sensitive_heads_dir, "results.pkl")
    if not os.path.isfile(selected_heads_path) or not os.path.isfile(head_results_path):
        raise FileNotFoundError(
            f"Expected selected_heads_elbow.json and results.pkl under {sensitive_heads_dir}."
        )
    selected_records = _load_json(selected_heads_path)
    selected_heads = [(int(item["layer"]), int(item["head"])) for item in selected_records]
    selected_set = set(selected_heads)
    head_results = load_intervention_results(head_results_path)
    white_head = _normalize_head_embeddings(head_results.get("white_emb", {}))
    black_head = _normalize_head_embeddings(head_results.get("black_emb", {}))
    missing_heads = [
        head for head in selected_heads if head not in white_head or head not in black_head
    ]
    if missing_heads:
        raise ValueError(f"Selected heads lack White/Black means: {missing_heads}")

    ranked_heads = get_sensitive_heads_sorted_by_heatmap(head_results)
    key_heads = [head for head in ranked_heads if head in selected_set]
    key_heads.extend(head for head in selected_heads if head not in set(key_heads))
    random_pool = [
        head
        for head in get_non_sensitive_heads_from_results(head_results)
        if head not in selected_set and head in white_head and head in black_head
    ]
    if len(random_pool) < len(key_heads):
        raise ValueError(
            f"Need {len(key_heads)} non-sensitive random heads, found {len(random_pool)}."
        )
    random_heads = random.Random(seed).sample(random_pool, len(key_heads))

    selected_mlps = _load_json(selected_mlp_path)
    key_mlps = [int(item["layer"]) for item in selected_mlps]
    with open(mlp_embeddings_path, "rb") as handle:
        mlp_results = pickle.load(handle)
    white_mlp = _normalize_mlp_embeddings(mlp_results.get("white_emb", {}))
    black_mlp = _normalize_mlp_embeddings(mlp_results.get("black_emb", {}))
    missing_mlps = [layer for layer in key_mlps if layer not in white_mlp or layer not in black_mlp]
    if missing_mlps:
        raise ValueError(f"Selected MLPs lack White/Black means: {missing_mlps}")

    return {
        "key_heads": key_heads,
        "random_heads": random_heads,
        "key_mlps": key_mlps,
        "white_head": white_head,
        "black_head": black_head,
        "white_mlp": white_mlp,
        "black_mlp": black_mlp,
        "selected_heads_path": os.path.abspath(selected_heads_path),
        "head_results_path": os.path.abspath(head_results_path),
        "selected_mlp_path": os.path.abspath(selected_mlp_path),
        "mlp_embeddings_path": os.path.abspath(mlp_embeddings_path),
        "mlp_source_metadata": {
            key: mlp_results.get(key)
            for key in (
                "dataset",
                "dataset_json_path",
                "model",
                "mlp_surface",
                "resume_prompt_mode",
                "sample_csv_path",
                "sample_size",
            )
        },
    }


def compute_probabilities(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    model_type: str,
    components: Dict[str, Any],
    heads: Sequence[Tuple[int, int]],
    mlp_layers: Sequence[int],
    input_device: torch.device,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    num_heads: int,
    head_dim: int,
    batch_size: int,
    description: str,
) -> List[float]:
    probabilities: List[float] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=description):
        formatted = [
            format_prompt_for_model(prompt, model_type)
            for prompt in prompts[start : start + batch_size]
        ]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        positions = get_last_token_indices_safe(
            inputs["input_ids"], inputs["attention_mask"], tokenizer
        )
        hooks = []
        for layer, head in heads:
            white = torch.as_tensor(
                components["white_head"][(layer, head)], dtype=torch.float32
            )
            black = torch.as_tensor(
                components["black_head"][(layer, head)], dtype=torch.float32
            )
            hooks.append(
                adapter.register_head_mean_replacement_hook(
                    layer,
                    head,
                    (white + black) / 2.0,
                    positions,
                    num_heads,
                    head_dim,
                )
            )
        for layer in mlp_layers:
            white = torch.as_tensor(components["white_mlp"][layer], dtype=torch.float32)
            black = torch.as_tensor(components["black_mlp"][layer], dtype=torch.float32)
            hooks.append(
                adapter.register_mlp_mean_replacement_hook(
                    layer_idx=layer,
                    mean_embedding=(white + black) / 2.0,
                    output_pos=positions,
                )
            )
        try:
            with torch.inference_mode():
                outputs = model(**inputs)
                batch_indices = torch.arange(len(formatted), device=outputs.logits.device)
                logits = outputs.logits[
                    batch_indices, positions.to(outputs.logits.device), :
                ].float()
                probabilities.extend(
                    compute_p_yes_from_logits_with_warning(
                        logits_row=row,
                        tokenizer=tokenizer,
                        yes_ids=list(yes_ids),
                        no_ids=list(no_ids),
                        sample_idx=start + offset,
                        show_warnings=False,
                        prefix=f"Adult-{description}",
                    )
                    for offset, row in enumerate(logits)
                )
        finally:
            remove_intervention_hooks(hooks)
    return probabilities


def build_pair_rows(
    condition: str,
    data: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    probabilities: Sequence[float],
    head_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if len(data) != len(probabilities):
        raise ValueError(f"Expected {len(data)} probabilities, got {len(probabilities)}.")
    samples = {int(sample["id"]): sample for sample in data}
    p_yes = {int(sample["id"]): float(value) for sample, value in zip(data, probabilities)}
    rows: List[Dict[str, Any]] = []
    for first_id, second_id in pairs:
        pair_samples = [samples[first_id], samples[second_id]]
        white = next(sample for sample in pair_samples if sample["race"].lower() == "white")
        black = next(sample for sample in pair_samples if sample["race"].lower() == "black")
        white_value = p_yes[int(white["id"])]
        black_value = p_yes[int(black["id"])]
        if not math.isfinite(white_value) or not math.isfinite(black_value):
            continue
        signed_gap = black_value - white_value
        row = {
            "condition": condition,
            "pair_id": int(white["pair_id"]),
            "source_row": int(white["source_row"]),
            "template_id": int(white["template_id"]),
            "label": int(white["label"]),
            "white_id": int(white["id"]),
            "black_id": int(black["id"]),
            "white_p_yes": white_value,
            "black_p_yes": black_value,
            "black_minus_white_gap": signed_gap,
            "fairness_violation": abs(signed_gap),
        }
        if head_count is not None:
            row = {"head_count": head_count, **row}
        rows.append(row)
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty intervention results.")
    violations = [float(row["fairness_violation"]) for row in rows]
    signed = [float(row["black_minus_white_gap"]) for row in rows]
    summary = {
        "condition": rows[0]["condition"],
        "valid_pairs": len(rows),
        "fairness_violation": statistics.fmean(violations),
        "black_minus_white_gap_mean": statistics.fmean(signed),
        "fairness_violation_median": statistics.median(violations),
        "fairness_violation_max": max(violations),
    }
    if "head_count" in rows[0]:
        summary["head_count"] = int(rows[0]["head_count"])
    return summary


def add_baseline_reductions(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline = next((row for row in summaries if row["condition"] == "base"), None)
    if baseline is None:
        return [dict(row) for row in summaries]
    baseline_value = float(baseline["fairness_violation"])
    output = []
    for row in summaries:
        result = dict(row)
        reduction = baseline_value - float(row["fairness_violation"])
        result["absolute_reduction_from_base"] = reduction
        result["relative_reduction_from_base"] = (
            reduction / baseline_value if baseline_value else 0.0
        )
        output.append(result)
    return output


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_completed_condition(
    path: str, condition: str, expected_pair_ids: Sequence[int]
) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_pair_ids):
        return []
    try:
        valid = [int(row["pair_id"]) for row in rows] == list(expected_pair_ids) and all(
            math.isfinite(float(row[column]))
            for row in rows
            for column in (
                "white_p_yes",
                "black_p_yes",
                "black_minus_white_gap",
                "fairness_violation",
            )
        )
    except (KeyError, TypeError, ValueError):
        return []
    if not valid or any(row.get("condition") != condition for row in rows):
        return []
    return rows


def _head_counts(number_of_heads: int) -> List[int]:
    return [value for value in (0, 5, 10, 15, 20, 25) if value <= number_of_heads]


def is_head_sweep_complete(
    output_dir: str, expected_pair_ids: Sequence[int], expected_counts: Sequence[int]
) -> bool:
    summary_path = os.path.join(output_dir, "head_count_summary.csv")
    if not os.path.isfile(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        expected_keys = {
            (f"{mode}_heads", count)
            for mode in ("sensitive", "random")
            for count in expected_counts
        }
        actual_keys = {
            (row["condition"], int(row["head_count"])) for row in summary_rows
        }
        if actual_keys != expected_keys or any(
            int(row["valid_pairs"]) != len(expected_pair_ids) for row in summary_rows
        ):
            return False
        for mode in ("sensitive", "random"):
            path = os.path.join(output_dir, f"per_pair_head_count_{mode}.csv")
            with open(path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != len(expected_pair_ids) * len(expected_counts):
                return False
            for count in expected_counts:
                group = [row for row in rows if int(row["head_count"]) == count]
                if [int(row["pair_id"]) for row in group] != list(expected_pair_ids):
                    return False
                if not all(
                    math.isfinite(float(row[column]))
                    for row in group
                    for column in ("white_p_yes", "black_p_yes", "fairness_violation")
                ):
                    return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--sensitive_heads_dir", required=True)
    parser.add_argument("--selected_mlp_path", required=True)
    parser.add_argument("--mlp_embeddings_path", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_head_sweep", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data, pairs = load_adult_pairs(args.dataset_path, args.max_pairs)
    samples = {int(sample["id"]): sample for sample in data}
    expected_pair_ids = []
    for first_id, second_id in pairs:
        white = next(
            sample
            for sample in (samples[first_id], samples[second_id])
            if sample["race"].lower() == "white"
        )
        expected_pair_ids.append(int(white["pair_id"]))
    components = select_components(
        args.sensitive_heads_dir,
        args.selected_mlp_path,
        args.mlp_embeddings_path,
        args.seed,
    )

    completed: Dict[str, List[Dict[str, Any]]] = {}
    if args.resume:
        for condition in args.conditions:
            rows = load_completed_condition(
                os.path.join(args.output_dir, f"per_pair_{condition}.csv"),
                condition,
                expected_pair_ids,
            )
            if rows:
                completed[condition] = rows
    pending = [condition for condition in args.conditions if condition not in completed]
    sweep_summary_path = os.path.join(args.output_dir, "head_count_summary.csv")
    count_grid = _head_counts(len(components["key_heads"]))
    sweep_complete = args.resume and is_head_sweep_complete(
        args.output_dir, expected_pair_ids, count_grid
    )

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(
        args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path
    )
    adapter = get_model_adapter(model, model_type=model_type, model_path=args.model_path)
    config = get_model_config(model)
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer did not provide usable Yes/No token IDs.")

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    metadata = {
        "experiment": "adult_race_topk_resume_component_transfer",
        "status": "running",
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(os.path.normpath(args.model_path)),
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "metric": "mean(abs(p_yes(black) - p_yes(white)))",
        "target_class": "Yes (income >50K)",
        "evaluation_protocol": "yes_no_income_gt_50k_v1",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "records": len(data),
        "pairs": len(pairs),
        "conditions": args.conditions,
        "key_heads": [list(head) for head in components["key_heads"]],
        "random_heads": [list(head) for head in components["random_heads"]],
        "key_mlps": components["key_mlps"],
        "head_count_grid": count_grid,
        "selected_heads_path": components["selected_heads_path"],
        "head_results_path": components["head_results_path"],
        "selected_mlp_path": components["selected_mlp_path"],
        "mlp_embeddings_path": components["mlp_embeddings_path"],
        "mlp_source_metadata": components["mlp_source_metadata"],
        "completed_conditions": list(completed),
        "head_sweep_complete": sweep_complete,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    prompts = [sample["prompt"] for sample in data]
    summaries = {name: summarize_rows(rows) for name, rows in completed.items()}
    for condition in pending:
        heads = components[condition] if condition in {"key_heads", "random_heads"} else []
        mlps = components["key_mlps"] if condition == "key_mlps" else []
        probabilities = compute_probabilities(
            model,
            adapter,
            tokenizer,
            prompts,
            model_type,
            components,
            heads,
            mlps,
            input_device,
            yes_ids,
            no_ids,
            config["num_heads"],
            config["head_dim"],
            args.batch_size,
            f"Adult top-K {condition}",
        )
        rows = build_pair_rows(condition, data, pairs, probabilities)
        if len(rows) != len(pairs):
            raise ValueError(f"Condition {condition} produced only {len(rows)} finite pairs.")
        _write_csv(os.path.join(args.output_dir, f"per_pair_{condition}.csv"), rows)
        completed[condition] = rows
        summaries[condition] = summarize_rows(rows)
        metadata["completed_conditions"].append(condition)
        ordered = [summaries[name] for name in args.conditions if name in summaries]
        _write_csv(os.path.join(args.output_dir, "summary.csv"), add_baseline_reductions(ordered))
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    if args.run_head_sweep and not sweep_complete:
        sweep_summaries: List[Dict[str, Any]] = []
        for mode in ("sensitive", "random"):
            candidates = (
                components["key_heads"] if mode == "sensitive" else components["random_heads"]
            )
            all_rows: List[Dict[str, Any]] = []
            for count in _head_counts(len(components["key_heads"])):
                probabilities = compute_probabilities(
                    model,
                    adapter,
                    tokenizer,
                    prompts,
                    model_type,
                    components,
                    candidates[:count],
                    [],
                    input_device,
                    yes_ids,
                    no_ids,
                    config["num_heads"],
                    config["head_dim"],
                    args.batch_size,
                    f"Adult {mode} heads K={count}",
                )
                condition = f"{mode}_heads"
                rows = build_pair_rows(condition, data, pairs, probabilities, head_count=count)
                all_rows.extend(rows)
                sweep_summaries.append(summarize_rows(rows))
            _write_csv(
                os.path.join(args.output_dir, f"per_pair_head_count_{mode}.csv"), all_rows
            )
        _write_csv(sweep_summary_path, sweep_summaries)
        metadata["head_sweep_complete"] = True

    ordered = [summaries[name] for name in args.conditions]
    _write_csv(os.path.join(args.output_dir, "summary.csv"), add_baseline_reductions(ordered))
    metadata["status"] = "complete"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(add_baseline_reductions(ordered), indent=2))


if __name__ == "__main__":
    main()
