#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP21: Analyze head patterns for Qwen3-4B on decision_question_id 12.
Compare Prompt vs Debiased Prompt conditions.
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
    format_prompt_for_model,
    resolve_model_type,
    add_yes_no_instruction,
)
from probability import YES_CANDIDATES, NO_CANDIDATES, get_target_token_ids
from util import get_model_config
from hook import (
    get_last_token_indices_safe,
)
from sampling import load_discrim_eval_pairs
from model_adapter import get_model_adapter

class DiscrimEvalPairedDataset(Dataset):
    def __init__(self, pairs: List[Tuple[dict, dict]], prompt_key: str):
        self.pairs = pairs
        self.prompt_key = prompt_key

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        fact, cf = self.pairs[idx]
        return {
            "index": idx,
            "fact_prompt": add_yes_no_instruction(fact[self.prompt_key]),
            "cf_prompt": add_yes_no_instruction(cf[self.prompt_key]),
        }

def _kl_pq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)

def run_analysis(model, adapter, tokenizer, dataloader, model_type, output_prefix):
    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))

    batch_activations_buffer = {}
    hooks = []
    for l in range(num_layers):
        hooks.append(
            adapter.register_head_activation_hook(
                l, num_heads, head_dim, batch_activations_buffer
            )
        )

    input_device = adapter.get_input_embedding_module().weight.device

    def collect_acts(is_fact=True):
        all_acts = {}
        key = "fact_prompt" if is_fact else "cf_prompt"
        for batch in tqdm(dataloader, desc=f"Collecting {'Fact' if is_fact else 'CF'} Acts"):
            prompts = [format_prompt_for_model(p, model_type) for p in batch[key]]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(input_device)
            last_token_indices = get_last_token_indices_safe(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                tokenizer,
            )
            batch_range = torch.arange(inputs["input_ids"].shape[0], device=input_device)
            batch_activations_buffer.clear()
            with torch.no_grad():
                _ = model(**inputs)
            for l in range(num_layers):
                buf = batch_activations_buffer[l]
                # buf can live on a different device when using device_map="auto"
                idx_device = buf.device
                act = buf[
                    batch_range.to(idx_device),
                    last_token_indices.to(idx_device),
                    :, :
                ]  # [B, H, D]
                act_np = act.detach().cpu().float().numpy()
                for h in range(num_heads):
                    if (l, h) not in all_acts: all_acts[(l, h)] = []
                    all_acts[(l, h)].append(act_np[:, h, :])
        return {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}

    fact_acts = collect_acts(is_fact=True)
    cf_acts = collect_acts(is_fact=False)
    for h in hooks: h.remove()

    kl_p_yes = np.zeros((num_layers, num_heads))
    mean_diff_p_yes = np.zeros((num_layers, num_heads))

    for l in range(num_layers):
        for h in range(num_heads):
            f_hd = torch.from_numpy(fact_acts[(l, h)])
            c_hd = torch.from_numpy(cf_acts[(l, h)])
            f_logits = adapter.project_head_activations_to_logits(
                l, h, f_hd, num_heads, head_dim
            )
            c_logits = adapter.project_head_activations_to_logits(
                l, h, c_hd, num_heads, head_dim
            )
            cand_ids = torch.tensor(cand_ids_list, device=f_logits.device)
            yes_mask = torch.tensor(
                [int(tid in set(yes_ids)) for tid in cand_ids_list],
                dtype=torch.bool,
                device=f_logits.device,
            )
            
            f_probs = torch.softmax(f_logits[:, cand_ids].float(), dim=-1)
            c_probs = torch.softmax(c_logits[:, cand_ids].float(), dim=-1)
            
            p_y_f = f_probs[:, yes_mask].sum(dim=-1)
            p_y_c = c_probs[:, yes_mask].sum(dim=-1)
            
            mean_diff_p_yes[l, h] = (p_y_f - p_y_c).abs().mean().item()
            P_f = torch.stack([p_y_f, 1-p_y_f], dim=-1)
            P_c = torch.stack([p_y_c, 1-p_y_c], dim=-1)
            kl_p_yes[l, h] = _kl_pq(P_f, P_c).mean().item()

    np.save(f"{output_prefix}_kl.npy", kl_p_yes)
    np.save(f"{output_prefix}_md.npy", mean_diff_p_yes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--qid", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    
    model_type = resolve_model_type("auto", model=model, tokenizer=tokenizer, model_path=args.model_path)
    adapter = get_model_adapter(model, model_type="auto", model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    id_to_sample = {item["id"]: item for item in data}
    target_pairs = []
    for a, b in pairs:
        if id_to_sample[a]["decision_question_id"] == args.qid:
            target_pairs.append((id_to_sample[a], id_to_sample[b]))
    print(f"Found {len(target_pairs)} pairs for QID {args.qid}")

    for p_type in ["prompt", "debiased_prompt"]:
        print(f"Analyzing {p_type}...")
        ds = DiscrimEvalPairedDataset(target_pairs, p_type)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        run_analysis(model, adapter, tokenizer, dl, model_type, os.path.join(args.output_dir, p_type))

    metadata = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "qid": args.qid,
        "num_pairs": len(target_pairs),
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "conditions": ["prompt", "debiased_prompt"],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
