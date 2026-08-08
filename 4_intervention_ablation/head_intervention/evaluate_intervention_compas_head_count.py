#!/usr/bin/env python
"""Measure COMPAS pair fairness before and after sensitive-head mean ablation."""

import argparse
import csv
import json
import math
import os
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
    get_sensitive_heads_sorted_by_heatmap,
    load_intervention_results,
)


def _normalize_embeddings(values: Dict[Any, Any]) -> Dict[Tuple[int, int], Any]:
    return {
        (int(key[0]), int(key[1])): value
        for key, value in values.items()
        if isinstance(key, (tuple, list)) and len(key) >= 2
    }


def load_compas_pairs(dataset_path: str, max_pairs: int = 0):
    data, pairs = load_discrim_eval_pairs(dataset_path)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
        selected_ids = {sample_id for pair in pairs for sample_id in pair}
        data = [sample for sample in data if int(sample["id"]) in selected_ids]
    if not pairs:
        raise ValueError("No COMPAS fact/counterfactual pairs were selected.")
    return data, pairs


def compute_p_yes_with_intervention(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    model_type: str,
    heads: Sequence[Tuple[int, int]],
    white_embeddings: Dict[Tuple[int, int], Any],
    black_embeddings: Dict[Tuple[int, int], Any],
    input_device: torch.device,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    num_heads: int,
    head_dim: int,
) -> List[float]:
    """Run one prompt at a time so MOE routing cannot depend on batch composition."""
    probabilities: List[float] = []
    for sample_index, prompt in enumerate(
        tqdm(prompts, desc=f"COMPAS p(yes), K={len(heads)}")
    ):
        formatted = format_prompt_for_model(prompt, model_type)
        input_ids = tokenizer.encode(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(input_device)
        attention_mask = torch.ones_like(input_ids)
        output_position = int(
            get_last_token_indices_safe(input_ids, attention_mask, tokenizer)[0].item()
        )

        hooks = []
        for layer, head in heads:
            key = (layer, head)
            if key not in white_embeddings or key not in black_embeddings:
                raise ValueError(f"Missing white/black embedding for head {key}.")
            white = torch.as_tensor(white_embeddings[key], dtype=torch.float32)
            black = torch.as_tensor(black_embeddings[key], dtype=torch.float32)
            mean_embedding = (white + black) / 2.0
            hooks.append(
                adapter.register_head_mean_replacement_hook(
                    layer,
                    head,
                    mean_embedding,
                    output_position,
                    num_heads,
                    head_dim,
                )
            )
        try:
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, output_position, :].float()
                probability = compute_p_yes_from_logits_with_warning(
                    logits_row=logits,
                    tokenizer=tokenizer,
                    yes_ids=list(yes_ids),
                    no_ids=list(no_ids),
                    sample_idx=sample_index,
                    show_warnings=False,
                    prefix=f"COMPAS-K{len(heads)}",
                )
                probabilities.append(float(probability))
        finally:
            remove_intervention_hooks(hooks)
    return probabilities


def build_pair_rows(
    data: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[int, int]],
    probabilities: Sequence[float],
    head_count: int,
    intervention_mode: str,
) -> List[Dict[str, Any]]:
    if len(data) != len(probabilities):
        raise ValueError(f"Expected {len(data)} probabilities, got {len(probabilities)}.")
    samples = {int(sample["id"]): sample for sample in data}
    p_yes = {
        int(sample["id"]): float(probability)
        for sample, probability in zip(data, probabilities)
    }
    rows = []
    for first_id, second_id in pairs:
        pair_samples = [samples[first_id], samples[second_id]]
        white = next(sample for sample in pair_samples if sample["race"].lower() == "white")
        black = next(sample for sample in pair_samples if sample["race"].lower() == "black")
        white_p_yes = p_yes[int(white["id"])]
        black_p_yes = p_yes[int(black["id"])]
        if not math.isfinite(white_p_yes) or not math.isfinite(black_p_yes):
            continue
        signed_gap = black_p_yes - white_p_yes
        rows.append({
            "intervention_mode": intervention_mode,
            "head_count": head_count,
            "pair_id": int(white["pair_id"]),
            "source_row": int(white["source_row"]),
            "template_id": int(white["template_id"]),
            "label": int(white["label"]),
            "white_id": int(white["id"]),
            "black_id": int(black["id"]),
            "white_p_yes": white_p_yes,
            "black_p_yes": black_p_yes,
            "black_minus_white_gap": signed_gap,
            "fairness_violation": abs(signed_gap),
        })
    return rows


def summarize_head_counts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_count: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_count.setdefault(int(row["head_count"]), []).append(row)
    if 0 not in by_count:
        raise ValueError("head_count=0 is required to calculate intervention reduction.")

    baseline = statistics.fmean(
        float(row["fairness_violation"]) for row in by_count[0]
    )
    summary = []
    for head_count in sorted(by_count):
        group = by_count[head_count]
        violations = [float(row["fairness_violation"]) for row in group]
        signed_gaps = [float(row["black_minus_white_gap"]) for row in group]
        value = statistics.fmean(violations)
        reduction = baseline - value
        summary.append({
            "intervention_mode": group[0]["intervention_mode"],
            "head_count": head_count,
            "valid_pairs": len(group),
            "fairness_violation": value,
            "black_minus_white_gap_mean": statistics.fmean(signed_gaps),
            "fairness_violation_median": statistics.median(violations),
            "fairness_violation_max": max(violations),
            "absolute_reduction_from_baseline": reduction,
            "relative_reduction_from_baseline": reduction / baseline if baseline else 0.0,
        })
    return summary


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare COMPAS fairness violation before/after sensitive-head intervention."
    )
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
    parser.add_argument(
        "--intervention_mode",
        default="sensitive",
        choices=["sensitive", "random"],
    )
    parser.add_argument("--head_counts", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if 0 not in args.head_counts:
        raise ValueError("--head_counts must include 0 as the no-intervention baseline.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results_path = os.path.join(args.sensitive_heads_dir, "results.pkl")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Sensitive-head embeddings not found: {results_path}")
    results = load_intervention_results(results_path)
    white_embeddings = _normalize_embeddings(results.get("white_emb", {}))
    black_embeddings = _normalize_embeddings(results.get("black_emb", {}))
    sensitive_heads = get_sensitive_heads_sorted_by_heatmap(results)
    random_heads = get_non_sensitive_heads_from_results(results)
    if args.intervention_mode == "sensitive":
        candidates = sensitive_heads
    else:
        candidates = list(random_heads)
        random.Random(args.seed).shuffle(candidates)
    if not candidates:
        raise ValueError(f"No {args.intervention_mode} head candidates found in {results_path}.")

    requested_counts = sorted(set(args.head_counts))
    unavailable = [count for count in requested_counts if count > len(candidates)]
    if unavailable:
        raise ValueError(
            f"Requested head counts {unavailable} exceed the {len(candidates)} available "
            f"{args.intervention_mode} heads."
        )

    data, pairs = load_compas_pairs(args.dataset_path, max_pairs=args.max_pairs)
    print(f"Loaded {len(data)} COMPAS records in {len(pairs)} pairs.")
    print(f"Loading base checkpoint: {args.model_path}")
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

    all_rows: List[Dict[str, Any]] = []
    selected_heads_by_count: Dict[str, List[List[int]]] = {}
    prompts = [sample["prompt"] for sample in data]
    for head_count in requested_counts:
        selected = candidates[:head_count]
        selected_heads_by_count[str(head_count)] = [list(head) for head in selected]
        probabilities = compute_p_yes_with_intervention(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            prompts=prompts,
            model_type=model_type,
            heads=selected,
            white_embeddings=white_embeddings,
            black_embeddings=black_embeddings,
            input_device=input_device,
            yes_ids=yes_ids,
            no_ids=no_ids,
            num_heads=config["num_heads"],
            head_dim=config["head_dim"],
        )
        all_rows.extend(
            build_pair_rows(
                data,
                pairs,
                probabilities,
                head_count=head_count,
                intervention_mode=args.intervention_mode,
            )
        )

    summary = summarize_head_counts(all_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    per_pair_path = os.path.join(args.output_dir, "per_pair.csv")
    summary_path = os.path.join(args.output_dir, "summary_by_head_count.csv")
    _write_csv(per_pair_path, all_rows)
    _write_csv(summary_path, summary)
    metadata = {
        "experiment": "compas_sensitive_head_intervention",
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_path": args.model_path,
        "checkpoint_type": "pre_pfairft_base_checkpoint",
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "sensitive_heads_dir": os.path.abspath(args.sensitive_heads_dir),
        "results_path": os.path.abspath(results_path),
        "intervention": "last-decision-token head mean replacement",
        "intervention_mode": args.intervention_mode,
        "metric": "mean(abs(p_yes(black) - p_yes(white)))",
        "label_source": "two_year_recid",
        "label_note": "The fairness metric is pair-based; two_year_recid is retained for stratification.",
        "seed": args.seed,
        "head_counts": requested_counts,
        "selected_heads_by_count": selected_heads_by_count,
        "records": len(data),
        "pairs": len(pairs),
        "per_pair_csv": os.path.abspath(per_pair_path),
        "summary_csv": os.path.abspath(summary_path),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
