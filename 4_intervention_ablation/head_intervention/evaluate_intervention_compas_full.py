#!/usr/bin/env python
"""Run full COMPAS base/key-head/random-head/key-MLP intervention evaluation."""

import argparse
import csv
import json
import math
import os
import pickle
import random
import statistics
import sys
from typing import Any, Dict, List, Sequence, Tuple

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
from probability import NO_CANDIDATES, YES_CANDIDATES, get_target_token_ids  # noqa: E402
from prompt import format_prompt_for_model, resolve_model_type  # noqa: E402
from sampling import load_discrim_eval_pairs  # noqa: E402
from util import (  # noqa: E402
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
    get_non_sensitive_heads_from_results,
    load_intervention_results,
)


CONDITIONS = ("base", "key_heads", "random_heads", "key_mlps")


def _normalize_head_embeddings(values: Dict[Any, Any]) -> Dict[Tuple[int, int], Any]:
    return {
        (int(key[0]), int(key[1])): value
        for key, value in values.items()
        if isinstance(key, (tuple, list)) and len(key) >= 2
    }


def _normalize_mlp_embeddings(values: Dict[Any, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in values.items()}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_compas(dataset_path: str, max_pairs: int):
    data, pairs = load_discrim_eval_pairs(dataset_path)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
        selected_ids = {sample_id for pair in pairs for sample_id in pair}
        data = [sample for sample in data if int(sample["id"]) in selected_ids]
    if not pairs:
        raise ValueError("No COMPAS pairs selected.")
    return data, pairs


def _select_components(args: argparse.Namespace) -> Dict[str, Any]:
    selected_heads_path = os.path.join(args.sensitive_heads_dir, "selected_heads_elbow.json")
    head_results_path = os.path.join(args.sensitive_heads_dir, "results.pkl")
    if not os.path.isfile(selected_heads_path) or not os.path.isfile(head_results_path):
        raise FileNotFoundError(
            f"Expected selected_heads_elbow.json and results.pkl under {args.sensitive_heads_dir}."
        )
    selected_head_records = _load_json(selected_heads_path)
    key_heads = [(int(item["layer"]), int(item["head"])) for item in selected_head_records]
    head_results = load_intervention_results(head_results_path)
    white_head = _normalize_head_embeddings(head_results.get("white_emb", {}))
    black_head = _normalize_head_embeddings(head_results.get("black_emb", {}))
    missing_heads = [head for head in key_heads if head not in white_head or head not in black_head]
    if missing_heads:
        raise ValueError(f"Key heads are missing white/black embeddings: {missing_heads}")

    random_pool = get_non_sensitive_heads_from_results(head_results)
    key_head_set = set(key_heads)
    random_pool = [head for head in random_pool if head not in key_head_set]
    if len(random_pool) < len(key_heads):
        raise ValueError(
            f"Need {len(key_heads)} random non-sensitive heads, found {len(random_pool)}."
        )
    random_heads = random.Random(args.seed).sample(random_pool, len(key_heads))
    if set(random_heads) & key_head_set:
        raise RuntimeError("Random-head control overlaps key heads.")

    selected_mlp_records = _load_json(args.selected_mlp_path)
    key_mlps = [int(item["layer"]) for item in selected_mlp_records]
    with open(args.mlp_embeddings_path, "rb") as handle:
        mlp_results = pickle.load(handle)
    white_mlp = _normalize_mlp_embeddings(mlp_results.get("white_emb", {}))
    black_mlp = _normalize_mlp_embeddings(mlp_results.get("black_emb", {}))
    missing_mlps = [layer for layer in key_mlps if layer not in white_mlp or layer not in black_mlp]
    if missing_mlps:
        raise ValueError(f"Key MLP layers are missing white/black embeddings: {missing_mlps}")

    return {
        "key_heads": key_heads,
        "random_heads": random_heads,
        "key_mlps": key_mlps,
        "white_head": white_head,
        "black_head": black_head,
        "white_mlp": white_mlp,
        "black_mlp": black_mlp,
        "head_results_path": head_results_path,
        "selected_heads_path": selected_heads_path,
        "mlp_metadata": {
            key: mlp_results.get(key)
            for key in (
                "dataset", "dataset_json_path", "model", "mlp_surface",
                "resume_prompt_mode", "sample_csv_path", "sample_size",
            )
        },
    }


def compute_condition_probabilities(
    condition: str,
    model: Any,
    adapter: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    model_type: str,
    components: Dict[str, Any],
    input_device: torch.device,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    num_heads: int,
    head_dim: int,
    batch_size: int,
) -> List[float]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    heads = components.get(condition, []) if condition in {"key_heads", "random_heads"} else []
    mlp_layers = components["key_mlps"] if condition == "key_mlps" else []
    probabilities: List[float] = []

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"COMPAS {condition}, batch={batch_size}",
    ):
        batch_prompts = [
            format_prompt_for_model(prompt, model_type)
            for prompt in prompts[start : start + batch_size]
        ]
        inputs = tokenizer(
            batch_prompts,
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
            white = torch.as_tensor(components["white_head"][(layer, head)], dtype=torch.float32)
            black = torch.as_tensor(components["black_head"][(layer, head)], dtype=torch.float32)
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
                batch_indices = torch.arange(len(batch_prompts), device=outputs.logits.device)
                logits = outputs.logits[
                    batch_indices, positions.to(outputs.logits.device), :
                ].float()
                for offset, logits_row in enumerate(logits):
                    probabilities.append(
                        float(
                            compute_p_yes_from_logits_with_warning(
                                logits_row=logits_row,
                                tokenizer=tokenizer,
                                yes_ids=list(yes_ids),
                                no_ids=list(no_ids),
                                sample_idx=start + offset,
                                show_warnings=False,
                                prefix=f"COMPAS-{condition}",
                            )
                        )
                    )
        finally:
            remove_intervention_hooks(hooks)
    return probabilities


def build_pair_rows(
    condition: str,
    data: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    probabilities: Sequence[float],
) -> List[Dict[str, Any]]:
    if len(data) != len(probabilities):
        raise ValueError(f"Expected {len(data)} probabilities, got {len(probabilities)}.")
    samples = {int(sample["id"]): sample for sample in data}
    p_yes = {
        int(sample["id"]): float(value)
        for sample, value in zip(data, probabilities)
    }
    rows = []
    for first_id, second_id in pairs:
        pair_samples = [samples[first_id], samples[second_id]]
        white = next(sample for sample in pair_samples if sample["race"].lower() == "white")
        black = next(sample for sample in pair_samples if sample["race"].lower() == "black")
        white_value = p_yes[int(white["id"])]
        black_value = p_yes[int(black["id"])]
        if not math.isfinite(white_value) or not math.isfinite(black_value):
            continue
        signed_gap = black_value - white_value
        rows.append({
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
        })
    return rows


def summarize_condition(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty condition results.")
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


def add_baseline_reductions(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_row = next((row for row in summary_rows if row["condition"] == "base"), None)
    if baseline_row is None:
        return [dict(row) for row in summary_rows]
    baseline = float(baseline_row["fairness_violation"])
    output = []
    for row in summary_rows:
        result = dict(row)
        reduction = baseline - float(row["fairness_violation"])
        result["absolute_reduction_from_base"] = reduction
        result["relative_reduction_from_base"] = reduction / baseline if baseline else 0.0
        output.append(result)
    return output


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_completed_condition(
    path: str,
    condition: str,
    expected_pair_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    """Load a complete condition CSV, or return an empty list for a stale partial file."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_pair_ids):
        return []
    try:
        pair_ids = [int(row["pair_id"]) for row in rows]
        finite = all(
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
    if pair_ids != list(expected_pair_ids) or not finite:
        return []
    if any(row.get("condition") != condition for row in rows):
        return []
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        default=os.path.join(REPO_ROOT, "data", "compas", "compas_paired.json"),
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument("--sensitive_heads_dir", required=True)
    parser.add_argument("--selected_mlp_path", required=True)
    parser.add_argument("--mlp_embeddings_path", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, finite, pair-aligned per-condition CSV files.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    components = _select_components(args)
    data, pairs = _load_compas(args.dataset_path, args.max_pairs)
    samples = {int(sample["id"]): sample for sample in data}
    expected_pair_ids = []
    for first_id, second_id in pairs:
        pair_samples = [samples[first_id], samples[second_id]]
        white = next(sample for sample in pair_samples if sample["race"].lower() == "white")
        expected_pair_ids.append(int(white["pair_id"]))

    completed_rows: Dict[str, List[Dict[str, Any]]] = {}
    if args.resume:
        for condition in args.conditions:
            path = os.path.join(args.output_dir, f"per_pair_{condition}.csv")
            rows = load_completed_condition(path, condition, expected_pair_ids)
            if rows:
                completed_rows[condition] = rows
                print(f"Resume: reusing complete condition {condition} ({len(rows)} pairs).")
            elif os.path.exists(path):
                print(f"Resume: stale or partial condition will be recomputed: {condition}.")

    pending_conditions = [
        condition for condition in args.conditions if condition not in completed_rows
    ]
    print(
        f"COMPAS records={len(data)}, pairs={len(pairs)}, key_heads={len(components['key_heads'])}, "
        f"random_heads={len(components['random_heads'])}, key_mlps={len(components['key_mlps'])}, "
        f"pending_conditions={pending_conditions}"
    )

    if not pending_conditions:
        summaries = [summarize_condition(completed_rows[condition]) for condition in args.conditions]
        _write_csv(os.path.join(args.output_dir, "summary.csv"), add_baseline_reductions(summaries))
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            metadata = _load_json(metadata_path)
            metadata["status"] = "complete"
            metadata["completed_conditions"] = list(args.conditions)
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        print("All requested conditions are already complete.")
        return

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
        args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    adapter = get_model_adapter(model, model_type=model_type, model_path=args.model_path)
    config = get_model_config(model)
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer did not provide usable Yes/No token IDs.")

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    metadata = {
        "experiment": "compas_full_component_intervention",
        "status": "running",
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_path": args.model_path,
        "checkpoint_type": "pre_pfairft_base_checkpoint",
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "mlp_surface": (
            "routed_moe_block_output"
            if adapter.family in {"deepseek", "olmoe", "jetmoe", "qwen_moe"}
            else "mlp_block_output"
        ),
        "metric": "mean(abs(p_yes(black) - p_yes(white)))",
        "seed": args.seed,
        "conditions": args.conditions,
        "records": len(data),
        "pairs": len(pairs),
        "batch_size": args.batch_size,
        "key_heads": [list(head) for head in components["key_heads"]],
        "random_heads": [list(head) for head in components["random_heads"]],
        "key_mlps": components["key_mlps"],
        "selected_heads_path": components["selected_heads_path"],
        "head_results_path": components["head_results_path"],
        "selected_mlp_path": os.path.abspath(args.selected_mlp_path),
        "mlp_embeddings_path": os.path.abspath(args.mlp_embeddings_path),
        "mlp_source_metadata": components["mlp_metadata"],
        "completed_conditions": [
            condition for condition in args.conditions if condition in completed_rows
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    prompts = [sample["prompt"] for sample in data]
    summaries = {
        condition: summarize_condition(rows)
        for condition, rows in completed_rows.items()
    }
    for condition in pending_conditions:
        probabilities = compute_condition_probabilities(
            condition=condition,
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            prompts=prompts,
            model_type=model_type,
            components=components,
            input_device=input_device,
            yes_ids=yes_ids,
            no_ids=no_ids,
            num_heads=config["num_heads"],
            head_dim=config["head_dim"],
            batch_size=args.batch_size,
        )
        pair_rows = build_pair_rows(condition, data, pairs, probabilities)
        _write_csv(os.path.join(args.output_dir, f"per_pair_{condition}.csv"), pair_rows)
        summaries[condition] = summarize_condition(pair_rows)
        ordered_summaries = [
            summaries[name] for name in args.conditions if name in summaries
        ]
        summaries_with_reductions = add_baseline_reductions(ordered_summaries)
        _write_csv(os.path.join(args.output_dir, "summary.csv"), summaries_with_reductions)
        metadata["completed_conditions"].append(condition)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(json.dumps(summaries_with_reductions[-1], indent=2))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata["status"] = "complete"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    ordered_summaries = [summaries[name] for name in args.conditions]
    print(json.dumps(add_baseline_reductions(ordered_summaries), indent=2))


if __name__ == "__main__":
    main()
