#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp13

在不同层 MLP 上捕获 MLP_in 与 MLP_out（last token），并与敏感属性向量 W_U[s]
计算余弦相似度，输出“偏见传递曲线”（含控制组：移除敏感词）。

读取与 prompt 逻辑参考 exp2/analyze_race_sensitive_MLPs.py。
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NOTE on environments:
# Some setups (e.g. Python 3.12 + NumPy 2.x) may have incompatible SciPy/sklearn wheels.
# Recent transformers may import sklearn in generation utilities even if we never call `generate`.
# To keep this script runnable in such environments, we optionally stub `sklearn` before importing transformers.
# If your environment has a working sklearn/scipy, you can disable this by:
#   FAIRNESSLLM_STUB_SKLEARN=0 python ...
# ---------------------------------------------------------------------------
if os.environ.get("FAIRNESSLLM_STUB_SKLEARN", "1") == "1":
    import sys as _sys
    import types as _types
    import importlib.machinery as _machinery

    if "sklearn" not in _sys.modules:
        _sk = _types.ModuleType("sklearn")
        _metrics = _types.ModuleType("sklearn.metrics")
        _sk.__spec__ = _machinery.ModuleSpec(name="sklearn", loader=None)
        _metrics.__spec__ = _machinery.ModuleSpec(name="sklearn.metrics", loader=None)

        def _roc_curve(*args, **kwargs):  # pragma: no cover
            raise ImportError("sklearn is stubbed (FAIRNESSLLM_STUB_SKLEARN=1). Install sklearn to use roc_curve.")

        _metrics.roc_curve = _roc_curve
        _sk.metrics = _metrics
        _sys.modules["sklearn"] = _sk
        _sys.modules["sklearn.metrics"] = _metrics

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from hook import get_last_token_indices_safe  # type: ignore  # noqa: E402
from prompt import (  # type: ignore  # noqa: E402
    add_yes_no_instruction,
    format_prompt_for_model,
    resolve_model_type,
)
from sampling import sample_resume_data_by_race  # type: ignore  # noqa: E402


def _get_model_num_layers_hidden_size(model: Any) -> Tuple[int, int]:
    """
    Lightweight model config extractor (avoid importing util.py which depends on pandas/scipy).
    Works for common Llama/Qwen-style `model.model.layers`.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        raise ValueError("Cannot find model layers to infer num_layers.")

    if hasattr(model, "config") and model.config is not None:
        cfg = model.config
        if hasattr(cfg, "hidden_size"):
            hidden_size = int(cfg.hidden_size)
        elif hasattr(cfg, "d_model"):
            hidden_size = int(cfg.d_model)
        else:
            hidden_size = int(getattr(cfg, "n_embd"))
    else:
        # fallback: try output embedding dimension
        out_emb = model.get_output_embeddings()
        if out_emb is None or not hasattr(out_emb, "weight"):
            raise ValueError("Cannot infer hidden_size (no model.config and no output embeddings).")
        hidden_size = int(out_emb.weight.shape[1])

    return int(num_layers), int(hidden_size)


def _get_input_device(model: Any, requested_device: str) -> torch.device:
    """
    For device_map="auto" models, inputs must be placed on the input embedding device.
    Otherwise, use requested_device.
    """
    device = torch.device(requested_device)
    use_auto_device = hasattr(model, "hf_device_map") and model.hf_device_map is not None
    if use_auto_device:
        try:
            emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
            if emb is not None and hasattr(emb, "weight"):
                device = emb.weight.device
            else:
                device = next(model.parameters()).device
        except Exception:
            device = next(model.parameters()).device
    else:
        # single-device: move model
        model.to(device)
    return device


def _neutralize_terms(text: str, terms: Sequence[str]) -> str:
    """Remove sensitive terms using word-boundary regex (case-insensitive)."""
    if not text:
        return text
    out = text
    for t in terms:
        if not t:
            continue
        # replace whole word occurrences; keep spacing clean
        out = re.sub(rf"\b{re.escape(t)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _pick_single_token_id(tokenizer: Any, token_str: str) -> int:
    """
    Best-effort pick a single token id for a concept string.
    Tries with/without leading space; requires encoding into exactly one token.
    """
    candidates = [token_str, " " + token_str]
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    # fallback: first token of plain encode
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if not ids:
        raise ValueError(f"attribute_token produced no token ids: {token_str!r}")
    return int(ids[0])


def _get_wu_attribute_vector(model: Any, token_id: int) -> torch.Tensor:
    """
    Return W_U[s] vector for token_id from output embedding / lm_head weight.
    Shape: [hidden_size]
    """
    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        # common fallback
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            weight = model.lm_head.weight
        else:
            raise ValueError("Cannot locate output embedding weight (W_U).")
    else:
        weight = out_emb.weight
    return weight[token_id].detach().float()


@dataclass
class BatchOut:
    mlp_in: Dict[int, torch.Tensor]   # layer -> [B, Seq, H]
    mlp_out: Dict[int, torch.Tensor]  # layer -> [B, Seq, H]
    last_indices: torch.Tensor        # [B]


def _register_mlp_hooks(model: Any, num_layers: int, buf_in: Dict[int, torch.Tensor], buf_out: Dict[int, torch.Tensor]):
    hooks = []
    for l in range(num_layers):
        if not hasattr(model.model.layers[l], "mlp"):
            raise ValueError(f"Layer {l} has no mlp module")

        mlp = model.model.layers[l].mlp

        def _make_hook(layer_idx: int):
            def hook(module, inputs, output):
                # inputs[0], output: [B, Seq, H]
                if inputs and torch.is_tensor(inputs[0]):
                    buf_in[layer_idx] = inputs[0].detach()
                else:
                    raise ValueError("Unexpected MLP hook inputs")
                buf_out[layer_idx] = output.detach() if torch.is_tensor(output) else output[0].detach()
            return hook

        hooks.append(mlp.register_forward_hook(_make_hook(l)))
    return hooks


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"index": idx, "prompt": self.prompts[idx]}


def _compute_layer_cosine_means(
    *,
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    model_type: str,
    attribute_vec: torch.Tensor,
    num_layers: int,
    batch_size: int,
    device: torch.device,
    max_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cos_in_mean:  [num_layers]
      cos_out_mean: [num_layers]
    """
    ds = PromptDataset(prompts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # accumulate sum and count
    sum_in = np.zeros(num_layers, dtype=np.float64)
    sum_out = np.zeros(num_layers, dtype=np.float64)
    count = 0

    buf_in: Dict[int, torch.Tensor] = {}
    buf_out: Dict[int, torch.Tensor] = {}
    hooks = _register_mlp_hooks(model, num_layers, buf_in, buf_out)

    # Keep attribute vector on CPU; move to each layer's device right before computing cosine.
    # This is important when using device_map="auto" (layers may live on different GPUs).
    attr_cpu = attribute_vec.detach().float().cpu()

    try:
        for batch in tqdm(dl, desc="Forward+Hook (MLP_in/out)", leave=False):
            batch_prompts = [add_yes_no_instruction(p) for p in batch["prompt"]]
            batch_prompts = [format_prompt_for_model(p, model_type) for p in batch_prompts]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length is not None else False,
                max_length=max_length,
                add_special_tokens=False,
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            last_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)

            buf_in.clear()
            buf_out.clear()

            with torch.no_grad():
                _ = model(**inputs)

            bsz = input_ids.shape[0]
            count += bsz

            for l in range(num_layers):
                if l not in buf_in or l not in buf_out:
                    raise RuntimeError(f"Missing hook buffer for layer {l}.")

                x_in = buf_in[l]   # [B, Seq, H]
                x_out = buf_out[l] # [B, Seq, H]

                dev = x_in.device
                last_idx_dev = last_indices.to(dev)
                batch_idx = torch.arange(bsz, device=dev)

                v_in = x_in[batch_idx, last_idx_dev, :].float()
                v_out = x_out[batch_idx, last_idx_dev, :].float()

                # cosine similarity to W_U[s]
                attr_dev = attr_cpu.to(dev)
                attr_norm = torch.norm(attr_dev) + 1e-12
                v_in_norm = torch.norm(v_in, dim=1) + 1e-12
                v_out_norm = torch.norm(v_out, dim=1) + 1e-12
                cos_in = (v_in @ attr_dev) / (v_in_norm * attr_norm)
                cos_out = (v_out @ attr_dev) / (v_out_norm * attr_norm)

                # Accumulate sums for correct sample-weighted mean across variable batch sizes.
                sum_in[l] += float(cos_in.sum().detach().cpu())
                sum_out[l] += float(cos_out.sum().detach().cpu())
    finally:
        for h in hooks:
            h.remove()

    denom = max(count, 1)
    return (sum_in / denom).astype(np.float32), (sum_out / denom).astype(np.float32)


def _plot_curves(
    out_path: str,
    cos_in: np.ndarray,
    cos_out: np.ndarray,
    cos_in_ctrl: np.ndarray,
    cos_out_ctrl: np.ndarray,
    title: str,
):
    import matplotlib.pyplot as plt

    layers = np.arange(len(cos_in))
    plt.figure(figsize=(10, 5))
    plt.plot(layers, cos_in, label="MLP_in (original)", linewidth=2)
    plt.plot(layers, cos_out, label="MLP_out (original)", linewidth=2)
    plt.plot(layers, cos_in_ctrl, label="MLP_in (control)", linewidth=2, linestyle="--")
    plt.plot(layers, cos_out_ctrl, label="MLP_out (control)", linewidth=2, linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity to W_U[s]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument("--output_dir", type=str, default="pFairFT/3_pattern_analysis/mlp_analysis/mlp_in_out_similarity")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Tokenizer max_length for truncation (helps avoid very long prompts).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
    )

    # sensitive concept
    parser.add_argument(
        "--attribute_token",
        type=str,
        required=True,
        help="Concept token string for W_U[s], e.g. 'female' or 'White'.",
    )
    parser.add_argument(
        "--sensitive_terms",
        nargs="+",
        default=[],
        help="Terms to neutralize for control prompts (removed by regex word boundary).",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="neutralize",
        choices=["neutralize", "none"],
        help="Control group: neutralize sensitive terms or disable control.",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        default=False,
        help="Use random sampling instead of sequential sampling (resume sampling).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling by race (resume dataset).",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if args.device.startswith("cuda", trust_remote_code=True) and torch.cuda.is_available() else None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    num_layers, hidden_size = _get_model_num_layers_hidden_size(model)
    print(f"Model layers={num_layers}, hidden_size={hidden_size}")

    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    print(f"Using model_type={model_type}")

    # If model is sharded, use the input embedding device for tokenized inputs.
    device = _get_input_device(model, args.device)

    print(f"Loading dataset: {args.dataset_json_path}")
    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of records.")

    sampled = sample_resume_data_by_race(
        data_records=dataset,
        max_samples=args.max_samples,
        balanced=args.balanced,
        random_sampling=args.random_sampling,
        seed=args.seed,
    )

    # Build prompts (reuse exp2 style: use summary as query)
    prompts: List[str] = []
    for item in sampled:
        summary = item.get("summary", "")
        if not summary:
            continue
        prompts.append(summary)

    if not prompts:
        raise ValueError("No prompts found after sampling.")

    if args.control_mode == "neutralize":
        prompts_ctrl = [_neutralize_terms(p, args.sensitive_terms) for p in prompts]
    else:
        prompts_ctrl = list(prompts)

    print(f"Prompts: {len(prompts)} (control_mode={args.control_mode})")

    token_id = _pick_single_token_id(tokenizer, args.attribute_token)
    attr_vec = _get_wu_attribute_vector(model, token_id)
    print(f"Attribute token={args.attribute_token!r}, token_id={token_id}")

    print("Computing cosine similarity curves (original)...")
    cos_in, cos_out = _compute_layer_cosine_means(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        model_type=model_type,
        attribute_vec=attr_vec,
        num_layers=num_layers,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    print("Computing cosine similarity curves (control)...")
    cos_in_ctrl, cos_out_ctrl = _compute_layer_cosine_means(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_ctrl,
        model_type=model_type,
        attribute_vec=attr_vec,
        num_layers=num_layers,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    # Save
    results = {
        "model_path": args.model_path,
        "dataset_json_path": args.dataset_json_path,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": len(prompts),
        "attribute_token": args.attribute_token,
        "attribute_token_id": token_id,
        "sensitive_terms": list(args.sensitive_terms),
        "control_mode": args.control_mode,
        "cosine_similarity": {
            "mlp_in_original": cos_in.tolist(),
            "mlp_out_original": cos_out.tolist(),
            "mlp_in_control": cos_in_ctrl.tolist(),
            "mlp_out_control": cos_out_ctrl.tolist(),
        },
    }

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fig_path = os.path.join(args.output_dir, "curve_mlp_in_out.png")
    _plot_curves(
        fig_path,
        cos_in=cos_in,
        cos_out=cos_out,
        cos_in_ctrl=cos_in_ctrl,
        cos_out_ctrl=cos_out_ctrl,
        title=f"MLP_in/out Cosine Similarity to W_U[{args.attribute_token}]",
    )
    print(f"Saved: {os.path.join(args.output_dir, 'results.json')}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

