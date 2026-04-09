#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


@dataclass
class ModelSpec:
    name: str
    model_path: str
    heads_analysis_dir: str


def iter_model_specs(
    llm_research_dir: str,
    qwen_dir: str,
    exp2_dir: str,
    restrict_to: Optional[Set[str]] = None,
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    def _scan(root: str) -> None:
        if not os.path.isdir(root):
            return
        for entry in sorted(os.listdir(root)):
            model_dir = os.path.join(root, entry)
            if not os.path.isdir(model_dir):
                continue
            model_name = os.path.basename(model_dir)
            if restrict_to is not None and model_name not in restrict_to:
                continue
            heads_dir = os.path.join(exp2_dir, f"sensitive_heads_{model_name}_top100")
            results_path = os.path.join(heads_dir, "results.pkl")
            if not os.path.isfile(results_path):
                continue
            specs.append(ModelSpec(name=model_name, model_path=model_dir, heads_analysis_dir=heads_dir))

    _scan(llm_research_dir)
    _scan(qwen_dir)
    return specs


def load_selected_layers_from_results_pkl(heads_analysis_dir: str) -> Set[int]:
    results_path = os.path.join(heads_analysis_dir, "results.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    selected_heads = results.get("selected_heads", [])
    layers: Set[int] = set()
    for h in selected_heads:
        if not isinstance(h, dict):
            continue
        if "layer" in h:
            try:
                layers.add(int(h["layer"]))
            except Exception:
                pass
    return layers


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_num_layers(model: torch.nn.Module) -> int:
    for attr_path in [
        ["model", "layers"],
        ["base_model", "model", "model", "layers"],
        ["base_model", "model", "layers"],
        ["base_model", "layers"],
    ]:
        cur = model
        ok = True
        for a in attr_path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            return len(cur)
    raise ValueError("Cannot infer number of layers from model structure")


def estimate_lora_params_for_linear(weight: torch.Tensor, r: int, bias: str = "none") -> int:
    out_features, in_features = weight.shape
    base = r * (in_features + out_features)
    if bias != "none":
        base += out_features
    return base


def estimate_lora_params_full_model(
    base_model: torch.nn.Module,
    target_modules: Sequence[str],
    r: int,
    bias: str = "none",
) -> int:
    total = 0
    for name, module in base_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight is None:
            continue
        leaf = name.split(".")[-1]
        if leaf in set(target_modules):
            total += estimate_lora_params_for_linear(module.weight, r=r, bias=bias)
    return total


def estimate_lora_params_selected_layers(
    base_model: torch.nn.Module,
    selected_layers: Set[int],
    target_modules: Sequence[str],
    r: int,
    bias: str = "none",
) -> int:
    total = 0
    target_set = set(target_modules)

    for name, module in base_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight is None:
            continue
        leaf = name.split(".")[-1]
        if leaf not in target_set:
            continue

        layer_idx: Optional[int] = None
        parts = name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] == "layers":
                try:
                    layer_idx = int(parts[i + 1])
                except Exception:
                    layer_idx = None
                break

        if layer_idx is None:
            continue
        if layer_idx not in selected_layers:
            continue

        total += estimate_lora_params_for_linear(module.weight, r=r, bias=bias)

    return total


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_research_dir", type=str, default="/mnt/nfs/huggingface/LLM-Research")
    parser.add_argument("--qwen_dir", type=str, default="/mnt/nfs/huggingface/Qwen")
    parser.add_argument("--exp2_dir", type=str, default="/home/common1/hwluo/project/pFairFT/exp2_old")
    parser.add_argument("--output_csv", type=str, default="/home/common1/hwluo/project/pFairFT/exp4/param_counts_lora_vs_precision.csv")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--restrict_models",
        type=str,
        default="Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Llama-3.2-1B-Instruct,Llama-3.2-3B-Instruct,Meta-Llama-3-8B-Instruct",
        help="Comma-separated model directory basenames. Empty means all models with exp2 results.pkl.",
    )
    parser.add_argument(
        "--precision_mode",
        type=str,
        default="selected_layers",
        choices=["as_is", "selected_layers"],
        help=(
            "as_is: use finetune_precision_fairness.py current behavior (LoRA on all target_modules). "
            "selected_layers: treat 'precision' as LoRA only on layers that contain selected_heads."
        ),
    )
    args = parser.parse_args()

    restrict_to: Optional[Set[str]]
    if args.restrict_models.strip() == "":
        restrict_to = None
    else:
        restrict_to = set([x.strip() for x in args.restrict_models.split(",") if x.strip()])

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    specs = iter_model_specs(args.llm_research_dir, args.qwen_dir, args.exp2_dir, restrict_to=restrict_to)
    if len(specs) == 0:
        raise SystemExit("No models found (missing exp2 results.pkl or restrict_models filtered everything).")

    rows: List[Dict[str, object]] = []

    for spec in specs:
        selected_layers = load_selected_layers_from_results_pkl(spec.heads_analysis_dir)

        base_model = AutoModelForCausalLM.from_pretrained(
            spec.model_path,
            device_map="cpu",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True, trust_remote_code=True
        )

        num_layers = infer_num_layers(base_model)
        selected_layer_count = len([l for l in selected_layers if 0 <= l < num_layers])

        full_lora_params_est = estimate_lora_params_full_model(
            base_model=base_model,
            target_modules=target_modules,
            r=args.lora_rank,
            bias=args.bias,
        )

        if args.precision_mode == "as_is":
            precision_params_est = full_lora_params_est
        else:
            precision_params_est = estimate_lora_params_selected_layers(
                base_model=base_model,
                selected_layers=set([l for l in selected_layers if 0 <= l < num_layers]),
                target_modules=target_modules,
                r=args.lora_rank,
                bias=args.bias,
            )

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        peft_model = get_peft_model(base_model, lora_config)
        trainable_params_actual = count_trainable_params(peft_model)

        rows.append(
            {
                "model_name": spec.name,
                "model_path": spec.model_path,
                "heads_analysis_dir": spec.heads_analysis_dir,
                "num_layers": num_layers,
                "selected_layers": selected_layer_count,
                "lora_rank": args.lora_rank,
                "target_modules": "+".join(target_modules),
                "full_lora_trainable_params_est": full_lora_params_est,
                "precision_trainable_params_est": precision_params_est,
                "full_lora_trainable_params_actual": trainable_params_actual,
                "precision_mode": args.precision_mode,
            }
        )

        del peft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", args.output_csv)
    print("\nSummary:")
    for r in rows:
        print(
            f"- {r['model_name']}: full_lora_actual={fmt_int(int(r['full_lora_trainable_params_actual']))}, "
            f"full_lora_est={fmt_int(int(r['full_lora_trainable_params_est']))}, "
            f"precision_est={fmt_int(int(r['precision_trainable_params_est']))} "
            f"(selected_layers={r['selected_layers']}/{r['num_layers']}, mode={r['precision_mode']})"
        )


if __name__ == "__main__":
    main()
