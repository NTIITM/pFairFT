#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze per-layer MLP output contribution under head mean-replacement intervention on Resume top-100.

- Load sensitive heads list.
- Run forward pass with head mean-replacement intervention at last token.
- Collect last-token MLP outputs for each layer.
- Project through lm_head.weight and compute KL(p_yes) and mean_diff(p_yes).
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
    remove_intervention_hooks,
)
from model_adapter import get_model_adapter
from residual_probe import collect_next_mlp_inputs


class ResumeDataset(Dataset):
    def __init__(self, fact_data: List[dict], cf_data: List[dict]):
        self.fact_data = fact_data
        self.cf_data = cf_data

    def __len__(self) -> int:
        return len(self.fact_data)

    def __getitem__(self, idx: int):
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


import pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument("--biased_csv_path", type=str, required=True)
    parser.add_argument("--sensitive_heads_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_type", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    device = torch.device(args.device)
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))
    cand_ids = torch.tensor(cand_ids_list, dtype=torch.long, device=device)
    yes_mask = torch.tensor([int(tok in set(yes_ids)) for tok in cand_ids_list], dtype=torch.bool, device=device)

    # Load sensitive heads and embeddings
    with open(args.sensitive_heads_path, "r", encoding="utf-8") as f:
        selected_heads_data = json.load(f)
    sensitive_heads = [(h["layer"], h["head"]) for h in selected_heads_data]

    with open(args.embeddings_path, "rb") as f:
        embeddings_data = pickle.load(f)
    white_embeddings_raw = embeddings_data.get("white_emb", {})
    black_embeddings_raw = embeddings_data.get("black_emb", {})

    def _normalize_head_key(k):
        if isinstance(k, tuple) and len(k) == 2:
            return (int(k[0]), int(k[1]))
        if isinstance(k, list) and len(k) == 2:
            return (int(k[0]), int(k[1]))
        return None

    white_embeddings: Dict[Tuple[int, int], np.ndarray] = {}
    black_embeddings: Dict[Tuple[int, int], np.ndarray] = {}
    for k, v in (white_embeddings_raw or {}).items():
        nk = _normalize_head_key(k)
        if nk is not None:
            white_embeddings[nk] = v
    for k, v in (black_embeddings_raw or {}).items():
        nk = _normalize_head_key(k)
        if nk is not None:
            black_embeddings[nk] = v

    head_mean_emb: Dict[Tuple[int, int], torch.Tensor] = {}
    valid_sensitive_heads: List[Tuple[int, int]] = []
    for l, h in sensitive_heads:
        k = (int(l), int(h))
        if k in white_embeddings and k in black_embeddings:
            w_e = torch.from_numpy(np.asarray(white_embeddings[k])).float()
            b_e = torch.from_numpy(np.asarray(black_embeddings[k])).float()
            head_mean_emb[k] = (w_e + b_e) / 2.0
            valid_sensitive_heads.append(k)

    sensitive_heads = valid_sensitive_heads

    sampled_data = _load_resume_topk_by_csv(
        args.dataset_json_path, args.biased_csv_path, args.sample_size
    )
    fact_data, cf_data = [], []
    for item in sampled_data:
        base_query = build_category_prompt(item["summary"], item["category"])
        fact_item = {"query": base_query, "ID": item.get("_orig_index", item.get("ID", 0))}
        fact_data.append(fact_item)
        cf_data.append(create_counterfactual_by_race(fact_item))

    dataset = ResumeDataset(fact_data, cf_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    def intervention_factory(output_pos: torch.Tensor):
        hooks = []
        for layer, head in sensitive_heads:
            mean_emb = head_mean_emb.get((layer, head))
            if mean_emb is None:
                continue
            hooks.append(
                adapter.register_head_mean_replacement_hook(
                    layer,
                    head,
                    mean_emb,
                    output_pos,
                    num_heads,
                    head_dim,
                )
            )
        return hooks

    fact_outs = collect_next_mlp_inputs(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        dataloader=dataloader,
        num_layers=num_layers,
        prompt_key="fact_prompt",
        intervention_factory=intervention_factory,
    )
    cf_outs = collect_next_mlp_inputs(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        dataloader=dataloader,
        num_layers=num_layers,
        prompt_key="cf_prompt",
        intervention_factory=intervention_factory,
    )
    mlp_kl = np.zeros(num_layers)
    mlp_mean_diff = np.zeros(num_layers)

    for l in range(num_layers):
        if l not in fact_outs or l not in cf_outs:
            continue

        f_hd = torch.from_numpy(fact_outs[l]).to(device)
        c_hd = torch.from_numpy(cf_outs[l]).to(device)

        f_logits = adapter.project_residual_to_logits(
            f_hd, apply_final_norm=False
        ).float()
        c_logits = adapter.project_residual_to_logits(
            c_hd, apply_final_norm=False
        ).float()
        f_probs = torch.softmax(
            f_logits[:, cand_ids.to(f_logits.device)].float(), dim=-1
        )
        c_probs = torch.softmax(
            c_logits[:, cand_ids.to(c_logits.device)].float(), dim=-1
        )
        p_y_f = f_probs[:, yes_mask.to(f_probs.device)].sum(-1)
        p_y_c = c_probs[:, yes_mask.to(c_probs.device)].sum(-1)
        mlp_mean_diff[l] = torch.abs(p_y_f - p_y_c).mean().item()
        mlp_kl[l] = _kl_pq(torch.stack([p_y_f, 1-p_y_f], -1), torch.stack([p_y_c, 1-p_y_c], -1)).mean().item()

    np.save(os.path.join(args.output_dir, "mlp_kl_p_yes_intervened.npy"), mlp_kl)
    np.save(os.path.join(args.output_dir, "mlp_mean_diff_p_yes_intervened.npy"), mlp_mean_diff)
    metadata = {
        "model_path": args.model_path,
        "dataset_json_path": args.dataset_json_path,
        "biased_csv_path": args.biased_csv_path,
        "sensitive_heads_path": args.sensitive_heads_path,
        "embeddings_path": args.embeddings_path,
        "sample_size": args.sample_size,
        "num_layers": num_layers,
        "num_samples": len(dataset),
        "num_selected_heads": len(sensitive_heads),
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "probe_surface": "next_mlp_input_cumulative_residual",
        "probe_norm": "none",
        "semantic_projection": "W_U h",
    }
    with open(os.path.join(args.output_dir, "mlp_summary_intervened.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved intervened MLP metrics to {args.output_dir}")

if __name__ == "__main__":
    main()
