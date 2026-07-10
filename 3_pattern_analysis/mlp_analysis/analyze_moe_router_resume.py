#!/usr/bin/env python
"""Analyze MOE routing on Resume top-ranked factual/counterfactual pairs."""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from hook import get_last_token_indices_safe, remove_intervention_hooks  # type: ignore  # noqa: E402
from model_adapter import get_model_adapter  # type: ignore  # noqa: E402
from probability import NO_CANDIDATES, YES_CANDIDATES, get_target_token_ids  # type: ignore  # noqa: E402
from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type  # type: ignore  # noqa: E402
from sampling import load_samples_by_csv_indices  # type: ignore  # noqa: E402
from util import create_counterfactual_by_race, get_model_config  # type: ignore  # noqa: E402


RouterKey = Tuple[int, str]
RouterValue = Any


class ResumePairs(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        fact = {
            "query": item.get("summary", ""),
            "race": item.get("race", ""),
            "ID": item.get("_orig_index", index),
        }
        cf = create_counterfactual_by_race(fact)
        return {
            "index": index,
            "fact_prompt": add_yes_no_instruction(fact["query"]),
            "cf_prompt": add_yes_no_instruction(cf["query"]),
        }


def _reshape_router_output(output: RouterValue, batch_size: int, seq_len: int) -> RouterValue:
    if torch.is_tensor(output):
        return output.detach().view(batch_size, seq_len, -1).cpu()
    if isinstance(output, tuple) and len(output) >= 2:
        if torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
            return (
                output[0].detach().view(batch_size, seq_len, -1).cpu(),
                output[1].detach().view(batch_size, seq_len, -1).cpu(),
            )
    raise TypeError(f"Unsupported router output type: {type(output)}")


def _last_router_value(value: RouterValue, positions: torch.Tensor) -> RouterValue:
    rows = torch.arange(len(positions))
    positions = positions.cpu()
    if torch.is_tensor(value):
        return value[rows, positions]
    return value[0][rows, positions], value[1][rows, positions]


def _router_metrics(first: RouterValue, second: RouterValue, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if torch.is_tensor(first) and torch.is_tensor(second):
        p = torch.softmax(first.float(), dim=-1)
        q = torch.softmax(second.float(), dim=-1)
        midpoint = 0.5 * (p + q)
        eps = 1e-10
        js = 0.5 * (
            torch.sum(p * (torch.log(p + eps) - torch.log(midpoint + eps)), dim=-1)
            + torch.sum(q * (torch.log(q + eps) - torch.log(midpoint + eps)), dim=-1)
        )
        k = min(top_k, p.shape[-1])
        first_top = torch.topk(p, k=k, dim=-1).indices
        second_top = torch.topk(q, k=k, dim=-1).indices
    elif isinstance(first, tuple) and isinstance(second, tuple):
        first_top, first_weights = first
        second_top, second_weights = second
        max_expert = int(torch.max(torch.cat([first_top.flatten(), second_top.flatten()])).item()) + 1
        p = torch.zeros(first_top.shape[0], max_expert, dtype=torch.float32)
        q = torch.zeros_like(p)
        p.scatter_add_(1, first_top.long(), first_weights.float())
        q.scatter_add_(1, second_top.long(), second_weights.float())
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        midpoint = 0.5 * (p + q)
        eps = 1e-10
        js = 0.5 * (
            torch.sum(p * (torch.log(p + eps) - torch.log(midpoint + eps)), dim=-1)
            + torch.sum(q * (torch.log(q + eps) - torch.log(midpoint + eps)), dim=-1)
        )
        first_top = first_top[:, :top_k]
        second_top = second_top[:, :top_k]
    else:
        raise TypeError("Router output representation changed between forwards")

    overlaps = []
    for left, right in zip(first_top.tolist(), second_top.tolist()):
        left_set, right_set = set(left), set(right)
        overlaps.append(len(left_set & right_set) / max(len(left_set | right_set), 1))
    return js.numpy(), np.asarray(overlaps, dtype=np.float64)


def _make_capture_hook(
    key: RouterKey,
    buffer: Dict[RouterKey, RouterValue],
    batch_size: int,
    seq_len: int,
):
    def hook(module, inputs, output):
        buffer[key] = _reshape_router_output(output, batch_size, seq_len)
        return output

    return hook


def _make_last_token_force_hook(
    key: RouterKey,
    fact_router: Dict[RouterKey, RouterValue],
    fact_positions: torch.Tensor,
    cf_positions: torch.Tensor,
    batch_size: int,
    seq_len: int,
):
    def hook(module, inputs, output):
        cached = fact_router.get(key)
        if cached is None:
            return output
        rows = torch.arange(batch_size, device=inputs[0].device)
        fact_pos = fact_positions.cpu()
        cf_pos = cf_positions.to(rows.device)
        if torch.is_tensor(output) and torch.is_tensor(cached):
            reshaped = output.view(batch_size, seq_len, -1).clone()
            values = cached[torch.arange(batch_size), fact_pos].to(
                device=reshaped.device, dtype=reshaped.dtype
            )
            reshaped[rows, cf_pos] = values
            return reshaped.view_as(output)
        if isinstance(output, tuple) and isinstance(cached, tuple):
            indices = output[0].view(batch_size, seq_len, -1).clone()
            weights = output[1].view(batch_size, seq_len, -1).clone()
            cpu_rows = torch.arange(batch_size)
            indices[rows, cf_pos] = cached[0][cpu_rows, fact_pos].to(indices.device, indices.dtype)
            weights[rows, cf_pos] = cached[1][cpu_rows, fact_pos].to(weights.device, weights.dtype)
            replacement = (indices.view_as(output[0]), weights.view_as(output[1]))
            return replacement + output[2:]
        return output

    return hook


def _load_head_means(selected_path: str, embeddings_path: str) -> Dict[Tuple[int, int], torch.Tensor]:
    with open(selected_path, "r", encoding="utf-8") as f:
        selected = [(int(row["layer"]), int(row["head"])) for row in json.load(f)]
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    white = embeddings.get("white_emb", {})
    black = embeddings.get("black_emb", {})
    means = {}
    for key in selected:
        if key in white and key in black:
            means[key] = torch.from_numpy(
                (np.asarray(white[key]) + np.asarray(black[key])) / 2.0
            ).float()
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="auto")
    parser.add_argument("--dataset_json_path", required=True)
    parser.add_argument("--sample_csv_path", required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--sensitive_heads_path", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        source = json.load(f)
    records, used_indices, _ = load_samples_by_csv_indices(
        source, args.sample_csv_path, args.sample_size
    )
    for item, index in zip(records, used_indices):
        item["_orig_index"] = int(index)
    dataloader = DataLoader(ResumePairs(records), batch_size=args.batch_size, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model_type = resolve_model_type(
        args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path
    )
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    routers = adapter.router_modules_for_freeze()
    if not routers:
        raise ValueError(f"No router modules found for adapter family {adapter.family}")
    config = get_model_config(model)
    head_means = _load_head_means(args.sensitive_heads_path, args.embeddings_path)
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    candidate_ids = list(dict.fromkeys(yes_ids + no_ids))
    yes_mask = torch.tensor([token in set(yes_ids) for token in candidate_ids])
    input_device = adapter.get_input_embedding_module().weight.device

    metric_values: Dict[RouterKey, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    final_gaps: Dict[str, List[float]] = defaultdict(list)

    def run_forward(
        prompts: List[str],
        head_intervention: bool = False,
        forced_router: Optional[Dict[RouterKey, RouterValue]] = None,
        fact_positions: Optional[torch.Tensor] = None,
    ):
        formatted = [format_prompt_for_model(prompt, model_type) for prompt in prompts]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        positions = get_last_token_indices_safe(
            inputs["input_ids"], inputs.get("attention_mask"), tokenizer
        )
        bsz, seq_len = inputs["input_ids"].shape
        captured: Dict[RouterKey, RouterValue] = {}
        hooks = []
        if forced_router is not None:
            if fact_positions is None:
                raise ValueError("fact_positions are required for frozen_fact mode")
            for key, module in routers:
                hooks.append(
                    module.register_forward_hook(
                        _make_last_token_force_hook(
                            key,
                            forced_router,
                            fact_positions,
                            positions,
                            bsz,
                            seq_len,
                        )
                    )
                )
        for key, module in routers:
            hooks.append(
                module.register_forward_hook(
                    _make_capture_hook(key, captured, bsz, seq_len)
                )
            )
        if head_intervention:
            for (layer, head), mean in head_means.items():
                hooks.append(
                    adapter.register_head_mean_replacement_hook(
                        layer,
                        head,
                        mean,
                        positions,
                        config["num_heads"],
                        config["head_dim"],
                    )
                )
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        finally:
            remove_intervention_hooks(hooks)
        rows = torch.arange(bsz, device=outputs.logits.device)
        logits = outputs.logits[rows, positions.to(outputs.logits.device)].float()
        probs = torch.softmax(logits[:, candidate_ids], dim=-1).cpu()
        p_yes = probs[:, yes_mask].sum(dim=-1).numpy()
        last = {key: _last_router_value(value, positions) for key, value in captured.items()}
        return p_yes, captured, last, positions

    for batch in tqdm(dataloader, desc="MOE router analysis"):
        fact_prompts = list(batch["fact_prompt"])
        cf_prompts = list(batch["cf_prompt"])
        fact_p, fact_raw, fact_last, fact_positions = run_forward(fact_prompts)
        cf_p, _, cf_last, _ = run_forward(cf_prompts)
        frozen_cf_p, _, _, _ = run_forward(
            cf_prompts, forced_router=fact_raw, fact_positions=fact_positions
        )
        fact_head_p, _, fact_head_last, _ = run_forward(
            fact_prompts, head_intervention=True
        )
        cf_head_p, _, cf_head_last, _ = run_forward(
            cf_prompts, head_intervention=True
        )
        final_gaps["native_router"].extend(np.abs(fact_p - cf_p).tolist())
        final_gaps["frozen_fact"].extend(np.abs(fact_p - frozen_cf_p).tolist())
        final_gaps["head_intervention"].extend(
            np.abs(fact_head_p - cf_head_p).tolist()
        )
        for key in fact_last:
            comparisons = {
                "native_fact_cf": (fact_last[key], cf_last[key]),
                "fact_head_change": (fact_last[key], fact_head_last[key]),
                "cf_head_change": (cf_last[key], cf_head_last[key]),
            }
            for name, (first, second) in comparisons.items():
                js, overlap = _router_metrics(first, second, args.top_k)
                metric_values[key][f"{name}_js"].extend(js.tolist())
                metric_values[key][f"{name}_topk_overlap"].extend(overlap.tolist())

    csv_path = os.path.join(args.output_dir, "router_metrics_by_layer.csv")
    metric_names = sorted(next(iter(metric_values.values())).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "router_kind"] + metric_names)
        for (layer, kind), metrics in sorted(metric_values.items()):
            writer.writerow(
                [layer, kind]
                + [float(np.mean(metrics[name])) for name in metric_names]
            )

    metadata = {
        "model_path": args.model_path,
        "model_type": model_type,
        "adapter_family": adapter.family,
        "dataset_json_path": args.dataset_json_path,
        "sample_csv_path": args.sample_csv_path,
        "sample_size": len(records),
        "sample_indices": [int(index) for index in used_indices],
        "resume_prompt_mode": "summary_only",
        "router_modes": ["native_router", "frozen_fact", "head_intervention"],
        "frozen_fact_scope": "last_decision_token_at_each_router",
        "num_router_modules": len(routers),
        "router_keys": [[layer, kind] for (layer, kind), _ in routers],
        "selected_heads_path": args.sensitive_heads_path,
        "embeddings_path": args.embeddings_path,
        "num_selected_heads": len(head_means),
        "top_k": args.top_k,
        "final_mean_abs_fact_cf_p_yes_gap": {
            name: float(np.mean(values)) for name, values in final_gaps.items()
        },
        "metrics_csv": csv_path,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
