#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP20: Head-level KL and mean-diff on Resume top-100 under MLP intervention.

MLP intervention logic is aligned with exp15/evaluate_intervention_MLP_resume.py:
- Load sensitive MLP layers from exp15/mlp_elbow_<MODEL_NAME>/selected_mlp_layers_elbow.json
- For each prompt, find last token position and on those sensitive layers,
  register a forward hook on layer.mlp that replaces the output at that token
  position with the input hidden state (identity), i.e. "mlp_negative".

Head metrics (same definition as exp20/analyze_head_kl_resume.py):
- For each sample, compute head logits (o_proj slice + lm_head) for fact and cf
  (both forwards run under the MLP intervention).
- Restrict logits to YES ∪ NO candidates, softmax over this candidate set,
  sum YES candidates to get P_yes; P_no = 1 - P_yes.
- KL per sample: KL([P_yes_fact,P_no_fact] || [P_yes_cf,P_no_cf])
- Head-level KL = mean over samples.
- mean_diff = mean(P_yes_fact) - mean(P_yes_cf).

Outputs in output_dir:
- kl_p_yes_mlp.npy
- mean_diff_p_yes_mlp.npy
- selected_heads_mlp_elbow.json  (new elbow selection on kl_p_yes_mlp)
- summary_mlp.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import (
    build_category_prompt,
    format_prompt_for_model,
    resolve_model_type,
    add_yes_no_instruction,
)
from probability import YES_CANDIDATES, NO_CANDIDATES, get_target_token_ids
from util import extract_race_from_query, create_counterfactual_by_race, get_model_config, compute_elbow_point
from sampling import load_samples_by_csv_indices
from hook import (
    get_last_token_indices_safe,
    remove_intervention_hooks,
)
from model_adapter import get_model_adapter


class ResumeInterventionDataset(Dataset):
    def __init__(self, fact_data: List[dict], cf_data: List[dict]):
        self.fact_data = fact_data
        self.cf_data = cf_data

    def __len__(self) -> int:
        return len(self.fact_data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        fact = self.fact_data[idx]
        cf = self.cf_data[idx]
        race = extract_race_from_query(fact["query"])
        return {
            "index": idx,
            "fact_prompt": add_yes_no_instruction(fact["query"]),
            "cf_prompt": add_yes_no_instruction(cf["query"]),
            "race": race if race else "Unknown",
        }


def _load_resume_topk_by_csv(dataset_json_path: str, csv_path: str, sample_size: int) -> List[dict]:
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(dataset_json_path)
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Resume dataset must be a list of records.")

    sampled_data, used_indices, _ = load_samples_by_csv_indices(
        dataset=dataset,
        csv_path=csv_path,
        sample_size=sample_size,
    )
    for rec, idx in zip(sampled_data, used_indices):
        rec["_orig_index"] = int(idx)
    return sampled_data


def _kl_pq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def _load_sensitive_mlp_layers(selected_mlp_json: str, num_layers: int) -> List[int]:
    if not os.path.exists(selected_mlp_json):
        raise FileNotFoundError(selected_mlp_json)
    with open(selected_mlp_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    layers: List[int] = []
    for item in data:
        if "layer" not in item:
            continue
        try:
            l = int(item["layer"])
        except Exception:
            continue
        if 0 <= l < num_layers:
            layers.append(l)
    # dedup keep order
    out: List[int] = []
    seen: Set[int] = set()
    for l in layers:
        if l not in seen:
            out.append(l)
            seen.add(l)
    return out


def make_mlp_input_to_output_replacement_hook(layer_idx: int, output_pos):
    """Same as exp15/evaluate_intervention_MLP_resume.py."""

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise ValueError(
                f"MLP hook at layer {layer_idx}: expected Tensor output, got {type(hidden)}"
            )
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError(
                f"MLP hook at layer {layer_idx}: expected Tensor input hidden states in inputs[0]"
            )

        inp = inputs[0]
        out = hidden.clone()

        if inp.ndim != 3 or out.ndim != 3:
            raise ValueError(
                f"MLP hook at layer {layer_idx}: expected 3D tensors, got inp={inp.shape}, out={out.shape}"
            )

        if torch.is_tensor(output_pos):
            rows = torch.arange(out.shape[0], device=out.device)
            positions = output_pos.to(out.device).clamp(min=0, max=out.shape[1] - 1)
            inp_vec = inp[rows.to(inp.device), positions.to(inp.device), :].to(
                dtype=out.dtype, device=out.device
            )
            out[rows, positions, :] = inp_vec
        elif int(output_pos) < out.shape[1]:
            inp_vec = inp[:, int(output_pos), :].to(dtype=out.dtype, device=out.device)
            out[:, int(output_pos), :] = inp_vec
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    return hook


def _run_with_mlp_intervention_collect_acts(
    model,
    adapter,
    tokenizer,
    model_type: str,
    device: torch.device,
    dataloader: DataLoader,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    sensitive_layers: List[int],
    prompt_key: str,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Run forward pass with MLP intervention and collect last-token head activations.

    Returns dict (layer, head) -> [N, head_dim] numpy
    """
    batch_activations_buffer: Dict[int, torch.Tensor] = {}

    hooks = []
    for l in range(num_layers):
        hooks.append(
            adapter.register_head_activation_hook(
                l, num_heads, head_dim, batch_activations_buffer
            )
        )

    input_device = adapter.get_input_embedding_module().weight.device

    acts: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for batch in tqdm(dataloader, desc=f"MLP-intervention activations ({prompt_key})"):
        prompts = [format_prompt_for_model(p, model_type) for p in batch[prompt_key]]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
        last_token_indices = get_last_token_indices_safe(inputs["input_ids"], attention_mask, tokenizer)
        batch_range = torch.arange(inputs["input_ids"].shape[0], device=input_device)

        prompt_hooks = []
        for l in sensitive_layers:
            target_module = adapter.get_mlp_module(l)
            hook_fn = make_mlp_input_to_output_replacement_hook(
                layer_idx=l, output_pos=last_token_indices
            )
            prompt_hooks.append(target_module.register_forward_hook(hook_fn))

        batch_activations_buffer.clear()
        try:
            with torch.no_grad():
                _ = model(**inputs)
        finally:
            remove_intervention_hooks(prompt_hooks)

        for l in range(num_layers):
            if l in batch_activations_buffer:
                act = batch_activations_buffer[l]  # [B, Seq, H, D]
                act_device = act.device
                last_act = act[
                    batch_range.to(act_device),
                    last_token_indices.to(act_device),
                    :,
                    :,
                ]  # [B, H, D]
                last_act_np = last_act.detach().cpu().float().numpy()
                for h in range(num_heads):
                    key = (l, h)
                    if key not in acts:
                        acts[key] = []
                    acts[key].append(last_act_np[:, h, :])
                del batch_activations_buffer[l]

    for hk in hooks:
        hk.remove()

    out: Dict[Tuple[int, int], np.ndarray] = {}
    for key, chunks in acts.items():
        out[key] = np.concatenate(chunks, axis=0)
    return out


def _elbow_select_heads(heatmap: np.ndarray) -> Tuple[float, List[Dict[str, int]]]:
    flat = heatmap.flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return 0.0, []
    sorted_scores = np.sort(flat)[::-1]
    elbow_idx, elbow_score = compute_elbow_point(sorted_scores)
    selected = []
    L, H = heatmap.shape
    for l in range(L):
        for h in range(H):
            if np.isfinite(heatmap[l, h]) and heatmap[l, h] >= elbow_score:
                selected.append({"layer": int(l), "head": int(h)})
    return float(elbow_score), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument("--biased_csv_path", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument(
        "--mlp_selected_path",
        type=str,
        required=True,
        help="Path to exp15/mlp_elbow_<MODEL_NAME>/selected_mlp_layers_elbow.json",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map = "auto" if args.device == "cuda" and torch.cuda.is_available() else None
    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    hidden_size = config["hidden_size"]
    print(f"Layers: {num_layers}, Heads: {num_heads}, Head Dim: {head_dim}, Hidden Size: {hidden_size}")

    model_type = resolve_model_type(
        args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path
    )
    print(f"Using model_type: {model_type}")
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    device = torch.device(args.device)

    # Candidate token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not resolve yes/no candidate token IDs.")

    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))
    yes_ids_set = set(yes_ids)

    # Detect actual head config if needed
    print("Detecting actual head config via a single forward...")
    temp_buf: Dict[str, int] = {}
    temp_hook = adapter.register_config_detection_hook(temp_buf)

    sampled_data = _load_resume_topk_by_csv(
        args.dataset_json_path, args.biased_csv_path, args.sample_size
    )

    fact_data: List[dict] = []
    cf_data: List[dict] = []
    for item in sampled_data:
        summary = item.get("summary", "")
        category = item.get("category", "")
        race = item.get("race", "")
        if not summary or not category:
            continue
        base_query = build_category_prompt(summary, category)
        extracted_race = extract_race_from_query(base_query) or race or "Unknown"
        fact_item = {
            "query": base_query,
            "summary": summary,
            "category": category,
            "race": extracted_race,
            "ID": item.get("_orig_index", item.get("ID", 0)),
        }
        cf_item = create_counterfactual_by_race(fact_item)
        fact_data.append(fact_item)
        cf_data.append(cf_item)

    min_len = min(len(fact_data), len(cf_data))
    fact_data = fact_data[:min_len]
    cf_data = cf_data[:min_len]
    print(f"Effective fact/cf pairs: {len(fact_data)}")

    # Trigger detection
    if fact_data:
        test_prompt = format_prompt_for_model(fact_data[0]["query"], model_type)
        test_inputs = tokenizer(
            [test_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            _ = model(**test_inputs)
    temp_hook.remove()

    if "num_heads" in temp_buf and temp_buf["num_heads"] is not None:
        detected_num_heads = temp_buf["num_heads"]
        detected_head_dim = temp_buf["head_dim"]
        if detected_num_heads != num_heads or detected_head_dim != head_dim:
            print("Detected config mismatch, updating:")
            print(f"  Initial: num_heads={num_heads}, head_dim={head_dim}")
            print(f"  Actual:  num_heads={detected_num_heads}, head_dim={detected_head_dim}")
            num_heads = detected_num_heads
            head_dim = detected_head_dim

    sensitive_layers = _load_sensitive_mlp_layers(args.mlp_selected_path, num_layers)
    print(f"Sensitive MLP layers (count={len(sensitive_layers)}): {sensitive_layers}")

    dataset_obj = ResumeInterventionDataset(fact_data, cf_data)
    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

    # Collect head activations under intervention, for fact and cf
    fact_acts = _run_with_mlp_intervention_collect_acts(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        device=device,
        dataloader=dataloader,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        sensitive_layers=sensitive_layers,
        prompt_key="fact_prompt",
    )
    cf_acts = _run_with_mlp_intervention_collect_acts(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        device=device,
        dataloader=dataloader,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        sensitive_layers=sensitive_layers,
        prompt_key="cf_prompt",
    )

    num_samples = next(iter(fact_acts.values())).shape[0] if fact_acts else 0
    print(f"Collected intervention activations. samples per head: {num_samples}")

    # Compute head metrics
    kl_p_yes_mlp = np.zeros((num_layers, num_heads), dtype=np.float64)
    mean_diff_p_yes_mlp = np.zeros((num_layers, num_heads), dtype=np.float64)

    for l in range(num_layers):
        for h in range(num_heads):
            key = (l, h)
            if key not in fact_acts or key not in cf_acts:
                continue

            fact_hd = torch.from_numpy(fact_acts[key])
            cf_hd = torch.from_numpy(cf_acts[key])
            fact_logits = adapter.project_head_activations_to_logits(
                l, h, fact_hd, num_heads, head_dim
            )
            cf_logits = adapter.project_head_activations_to_logits(
                l, h, cf_hd, num_heads, head_dim
            )
            fact_cand_ids = torch.tensor(
                cand_ids_list, dtype=torch.long, device=fact_logits.device
            )
            cf_cand_ids = fact_cand_ids.to(cf_logits.device)
            fact_yes_mask = torch.tensor(
                [int(tok_id in yes_ids_set) for tok_id in cand_ids_list],
                dtype=torch.bool,
                device=fact_logits.device,
            )
            cf_yes_mask = fact_yes_mask.to(cf_logits.device)

            fact_cand_probs = torch.softmax(
                fact_logits[:, fact_cand_ids].float(), dim=-1
            )
            cf_cand_probs = torch.softmax(
                cf_logits[:, cf_cand_ids].float(), dim=-1
            )

            p_yes_fact = fact_cand_probs[:, fact_yes_mask].sum(dim=-1)
            p_yes_cf = cf_cand_probs[:, cf_yes_mask].sum(dim=-1)

            p_no_fact = 1.0 - p_yes_fact
            p_no_cf = 1.0 - p_yes_cf

            mean_diff_p_yes_mlp[l, h] = float(p_yes_fact.mean().item() - p_yes_cf.mean().item())

            P_fact = torch.stack([p_yes_fact, p_no_fact], dim=-1)
            P_cf = torch.stack([p_yes_cf, p_no_cf], dim=-1)

            kl_vec = _kl_pq(P_fact, P_cf).detach().cpu().numpy()
            kl_p_yes_mlp[l, h] = float(np.mean(kl_vec))

    np.save(os.path.join(args.output_dir, "kl_p_yes_mlp.npy"), kl_p_yes_mlp)
    np.save(os.path.join(args.output_dir, "mean_diff_p_yes_mlp.npy"), mean_diff_p_yes_mlp)

    elbow_score, selected_heads = _elbow_select_heads(kl_p_yes_mlp)
    with open(os.path.join(args.output_dir, "selected_heads_mlp_elbow.json"), "w", encoding="utf-8") as f:
        json.dump(selected_heads, f, indent=2)

    summary = {
        "model_path": args.model_path,
        "dataset_json_path": args.dataset_json_path,
        "biased_csv_path": args.biased_csv_path,
        "sample_size": args.sample_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "sensitive_mlp_path": args.mlp_selected_path,
        "sensitive_layers": sensitive_layers,
        "elbow_score": elbow_score,
        "cand_ids_count": len(cand_ids_list),
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
    }
    with open(os.path.join(args.output_dir, "summary_mlp.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved MLP-intervention results to {args.output_dir}")


if __name__ == "__main__":
    main()
