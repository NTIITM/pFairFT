#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP20: For each head, compare fact vs counterfactual p(yes) on Resume top-100.

Definition (per head):
- For each sample, take head output -> o_proj slice -> lm_head to get logits.
- Restrict logits to YES_CANDIDATES ∪ NO_CANDIDATES and apply softmax over this
  candidate set (yes+no together) to get a distribution over candidates.
- Let P_yes_fact be the sum of probabilities of YES_CANDIDATES under fact,
  P_yes_cf under counterfactual; P_no = 1 - P_yes.
- Define a 2-class distribution: [P_yes, P_no] for fact and cf.
- Head-level KL = mean_s KL( [P_yes_fact_s, P_no_fact_s] || [P_yes_cf_s, P_no_cf_s] )
- Head-level mean_diff = mean_s(P_yes_fact_s) - mean_s(P_yes_cf_s).

This script outputs, per model:
- kl_p_yes.npy:  [num_layers, num_heads]
- mean_diff_p_yes.npy: [num_layers, num_heads]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import (
    build_category_prompt,
    format_prompt_for_model,
    resolve_model_type,
    add_yes_no_instruction,
)
from probability import YES_CANDIDATES, NO_CANDIDATES, get_target_token_ids
from util import extract_race_from_query, create_counterfactual_by_race, get_model_config
from sampling import load_samples_by_csv_indices
from hook import (
    get_last_token_indices_safe,
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


def _load_resume_topk_by_csv(
    dataset_json_path: str,
    csv_path: str,
    sample_size: int,
) -> List[dict]:
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
    """Compute KL(p || q) along last dimension for distributions.

    p, q: [..., K]
    returns: [...] (sum over last dim)
    """
    eps = 1e-10
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


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
    print(f"Yes IDs: {yes_ids}")
    print(f"No IDs: {no_ids}")

    # Union of yes+no candidates, used for candidate-level softmax
    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))  # preserve order, remove dups
    print(f"Total candidate tokens (yes+no): {len(cand_ids_list)}")

    # Detect actual num_heads/head_dim if needed
    print("Detecting actual head config via a single forward...")
    temp_buf: Dict[str, int] = {}
    temp_hook = adapter.register_config_detection_hook(temp_buf)

    # Prepare data: load top-K resume samples by CSV index
    print(f"Loading resume top-{args.sample_size} samples by {args.biased_csv_path}")
    sampled_data = _load_resume_topk_by_csv(
        args.dataset_json_path, args.biased_csv_path, args.sample_size
    )
    print(f"Loaded {len(sampled_data)} samples")

    # Build fact/cf records using category+summary (方式A)
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

    # Run one sample to trigger config detection
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

    # Dataset & dataloader
    dataset_obj = ResumeInterventionDataset(fact_data, cf_data)
    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

    # Collect activations for fact and cf at last token: [N, H, D]
    print("Collecting per-head activations for fact and cf...")
    batch_activations_buffer: Dict[int, torch.Tensor] = {}

    hooks = []
    for l in range(num_layers):
        hooks.append(
            adapter.register_head_activation_hook(
                l, num_heads, head_dim, batch_activations_buffer
            )
        )

    # We'll store per-layer/head activations as lists of [B, D] and concat later
    fact_acts: Dict[Tuple[int, int], List[np.ndarray]] = {}
    cf_acts: Dict[Tuple[int, int], List[np.ndarray]] = {}

    # Fact pass
    print("Fact forward pass...")
    for batch in tqdm(dataloader, desc="Fact Activations"):
        fact_prompts = [format_prompt_for_model(p, model_type) for p in batch["fact_prompt"]]
        fact_inputs = tokenizer(
            fact_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        attention_mask = fact_inputs.get(
            "attention_mask", torch.ones_like(fact_inputs["input_ids"])
        )
        last_token_indices = get_last_token_indices_safe(
            fact_inputs["input_ids"], attention_mask, tokenizer
        )
        batch_range = torch.arange(fact_inputs["input_ids"].shape[0], device=device)

        batch_activations_buffer.clear()
        with torch.no_grad():
            _ = model(**fact_inputs)

        for l in range(num_layers):
            if l in batch_activations_buffer:
                act = batch_activations_buffer[l]  # [B, Seq, H, D]
                act_device = act.device
                batch_range_on_device = batch_range.to(act_device)
                last_token_indices_on_device = last_token_indices.to(act_device)
                last_act = act[
                    batch_range_on_device,
                    last_token_indices_on_device,
                    :,
                    :,
                ]  # [B, H, D]
                last_act_np = last_act.detach().cpu().float().numpy()
                for h in range(num_heads):
                    key = (l, h)
                    if key not in fact_acts:
                        fact_acts[key] = []
                    fact_acts[key].append(last_act_np[:, h, :])  # [B, D]
                del batch_activations_buffer[l]

    # CF pass
    print("Counterfactual forward pass...")
    for batch in tqdm(dataloader, desc="CF Activations"):
        cf_prompts = [format_prompt_for_model(p, model_type) for p in batch["cf_prompt"]]
        cf_inputs = tokenizer(
            cf_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        attention_mask = cf_inputs.get(
            "attention_mask", torch.ones_like(cf_inputs["input_ids"])
        )
        last_token_indices = get_last_token_indices_safe(
            cf_inputs["input_ids"], attention_mask, tokenizer
        )
        batch_range = torch.arange(cf_inputs["input_ids"].shape[0], device=device)

        batch_activations_buffer.clear()
        with torch.no_grad():
            _ = model(**cf_inputs)

        for l in range(num_layers):
            if l in batch_activations_buffer:
                act = batch_activations_buffer[l]  # [B, Seq, H, D]
                act_device = act.device
                batch_range_on_device = batch_range.to(act_device)
                last_token_indices_on_device = last_token_indices.to(act_device)
                last_act = act[
                    batch_range_on_device,
                    last_token_indices_on_device,
                    :,
                    :,
                ]  # [B, H, D]
                last_act_np = last_act.detach().cpu().float().numpy()
                for h in range(num_heads):
                    key = (l, h)
                    if key not in cf_acts:
                        cf_acts[key] = []
                    cf_acts[key].append(last_act_np[:, h, :])  # [B, D]
                del batch_activations_buffer[l]

    for h in hooks:
        h.remove()

    # Concatenate activations to [N, D]
    for key in fact_acts:
        fact_acts[key] = np.concatenate(fact_acts[key], axis=0)  # [N, D]
    for key in cf_acts:
        cf_acts[key] = np.concatenate(cf_acts[key], axis=0)  # [N, D]

    num_samples = next(iter(fact_acts.values())).shape[0] if fact_acts else 0
    print(f"Collected activations for {len(fact_acts)} heads, samples per head: {num_samples}")

    # Compute KL_p_yes and mean_diff_p_yes for each layer/head
    print("Computing KL_p_yes and mean_diff_p_yes per head (fact vs cf)...")

    kl_p_yes = np.zeros((num_layers, num_heads), dtype=np.float64)
    mean_diff_p_yes = np.zeros((num_layers, num_heads), dtype=np.float64)

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
            cand_ids = torch.tensor(cand_ids_list, dtype=torch.long, device=fact_logits.device)
            yes_ids_set = set(yes_ids)
            yes_mask = torch.tensor(
                [int(tok_id in yes_ids_set) for tok_id in cand_ids_list],
                dtype=torch.bool,
                device=fact_logits.device,
            )

            # Restrict to candidate set (yes+no) and softmax over candidates
            fact_cand_logits = fact_logits[:, cand_ids]  # [N, K_cand]
            cf_cand_logits = cf_logits[:, cand_ids]      # [N, K_cand]

            fact_cand_probs = torch.softmax(fact_cand_logits.float(), dim=-1)
            cf_cand_probs = torch.softmax(cf_cand_logits.float(), dim=-1)

            # P_yes and P_no per sample
            p_yes_fact = fact_cand_probs[:, yes_mask].sum(dim=-1)  # [N]
            p_yes_cf = cf_cand_probs[:, yes_mask].sum(dim=-1)      # [N]

            p_no_fact = 1.0 - p_yes_fact
            p_no_cf = 1.0 - p_yes_cf

            # Mean difference of P_yes
            mean_diff = float(p_yes_fact.mean().item() - p_yes_cf.mean().item())
            mean_diff_p_yes[l, h] = mean_diff

            # Build 2-class distributions and compute KL per sample, then mean
            P_fact = torch.stack([p_yes_fact, p_no_fact], dim=-1)  # [N, 2]
            P_cf = torch.stack([p_yes_cf, p_no_cf], dim=-1)        # [N, 2]

            kl_vec = _kl_pq(P_fact, P_cf).detach().cpu().numpy()   # [N]
            kl_p_yes[l, h] = float(np.mean(kl_vec))

    # Save outputs
    np.save(os.path.join(args.output_dir, "kl_p_yes.npy"), kl_p_yes)
    np.save(os.path.join(args.output_dir, "mean_diff_p_yes.npy"), mean_diff_p_yes)

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
        "yes_ids": yes_ids,
        "no_ids": no_ids,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved kl_p_yes and mean_diff_p_yes to {args.output_dir}")


if __name__ == "__main__":
    main()
