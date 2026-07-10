#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze per-layer MLP output contribution to p(yes) on Resume top-100 (baseline).

For each model and each transformer layer's MLP:
- On fact and counterfactual Resume prompts (top-100 from biased_samples_ranking.csv),
  collect the last-token MLP output (post-MLP, pre-residual add).
- Project this MLP output through lm_head.weight to obtain logits-only-from-MLP.
- Restrict logits to YES ∪ NO candidates, softmax over this candidate set, sum YES
  to get P_yes, set P_no = 1 - P_yes.
- Define per-sample Bernoulli distributions [P_yes,P_no] for fact and cf, and
  compute KL_s = KL( [P_yes_fact,P_no_fact] || [P_yes_cf,P_no_cf] ).
- Per-layer metrics:
  KL_p_yes_mlp_layer[l] = mean_s KL_s
  mean_diff_p_yes_mlp_layer[l] = mean_s(P_yes_fact_s - P_yes_cf_s)

Outputs in output_dir:
- mlp_kl_p_yes.npy           shape [num_layers]
- mlp_mean_diff_p_yes.npy    shape [num_layers]
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# import project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import (  # type: ignore
    build_category_prompt,
    format_prompt_for_model,
    resolve_model_type,
    add_yes_no_instruction,
)
from probability import (
    YES_CANDIDATES,
    NO_CANDIDATES,
    RACE_WHITE_CANDIDATES,
    RACE_BLACK_CANDIDATES,
    get_target_token_ids,
)  # type: ignore
from util import extract_race_from_query, create_counterfactual_by_race, get_model_config  # type: ignore
from sampling import load_samples_by_csv_indices  # type: ignore
from hook import get_last_token_indices_safe  # type: ignore
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
    hidden_size = config["hidden_size"]

    model_type = resolve_model_type(
        args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path
    )
    print(f"Using model_type: {model_type}")
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    device = torch.device(args.device)

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not resolve yes/no candidate token IDs.")

    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))
    cand_ids = torch.tensor(cand_ids_list, dtype=torch.long, device=device)
    yes_ids_set = set(yes_ids)
    yes_mask = torch.tensor(
        [int(tok in yes_ids_set) for tok in cand_ids_list],
        dtype=torch.bool,
        device=device,
    )

    race_white_ids = get_target_token_ids(tokenizer, RACE_WHITE_CANDIDATES)
    race_black_ids = get_target_token_ids(tokenizer, RACE_BLACK_CANDIDATES)
    if not race_white_ids or not race_black_ids:
        raise ValueError("Could not resolve race candidate token IDs (white/black).")
    race_ids_list = list(dict.fromkeys(race_white_ids + race_black_ids))
    race_ids = torch.tensor(race_ids_list, dtype=torch.long, device=device)
    white_ids_set = set(race_white_ids)
    white_mask = torch.tensor(
        [int(tok in white_ids_set) for tok in race_ids_list],
        dtype=torch.bool,
        device=device,
    )

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

    dataset_obj = ResumeDataset(fact_data, cf_data)
    dataloader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)

    fact_layer_outs = collect_next_mlp_inputs(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        dataloader=dataloader,
        num_layers=num_layers,
        prompt_key="fact_prompt",
    )
    cf_layer_outs = collect_next_mlp_inputs(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        model_type=model_type,
        dataloader=dataloader,
        num_layers=num_layers,
        prompt_key="cf_prompt",
    )

    num_samples = next(iter(fact_layer_outs.values())).shape[0] if fact_layer_outs else 0

    w_u = adapter.get_lm_head_weight().to(device=device, dtype=torch.float32)

    mlp_kl = np.zeros((num_layers,), dtype=np.float64)
    mlp_mean_diff = np.zeros((num_layers,), dtype=np.float64)
    
    # Signed cosine similarity between per-layer delta MLP input (fact - cf) and sensitive direction (race) in Wu.
    mlp_input_delta_cos_race_signed = np.full((num_layers,), np.nan, dtype=np.float64)

    # Mean absolute difference in race probability between fact and cf MLP inputs.
    mlp_input_mean_abs_diff_p_race = np.zeros((num_layers,), dtype=np.float64)

    # Build sign array in the exact dataloader order (white=+1, black=-1, other=0)
    sign_chunks: List[np.ndarray] = []
    for batch in dataloader:
        races = [str(r).lower() for r in batch["race"]]
        s = np.zeros((len(races),), dtype=np.float32)
        for i, r in enumerate(races):
            if "white" in r.lower():
                s[i] = 1.0
            elif "black" in r.lower():
                s[i] = -1.0
        sign_chunks.append(s)
    sign_arr = np.concatenate(sign_chunks, axis=0) if sign_chunks else np.zeros((0,), dtype=np.float32)

    for l in range(num_layers):
        if l not in fact_layer_outs or l not in cf_layer_outs:
            continue

        # 直接使用该层的 MLP 输出，不再累加
        fact_hd = torch.from_numpy(fact_layer_outs[l]).to(device=device, dtype=torch.float32)  # [N, Hidden]
        cf_hd = torch.from_numpy(cf_layer_outs[l]).to(device=device, dtype=torch.float32)

        fact_logits = adapter.project_residual_to_logits(
            fact_hd, apply_final_norm=False
        ).float()
        cf_logits = adapter.project_residual_to_logits(
            cf_hd, apply_final_norm=False
        ).float()

        # Signed cosine similarity between delta input (fact - cf) and sensitive direction (race) in Wu
        # Sensitive direction: mean(Wu[white_tokens]) - mean(Wu[black_tokens])
        with torch.no_grad():
            wu_white = w_u[race_white_ids].mean(dim=0)
            wu_black = w_u[race_black_ids].mean(dim=0)
            v_race = (wu_white - wu_black)
            v_race = v_race / (v_race.norm(p=2) + 1e-12)

        delta_hd = (fact_hd - cf_hd)
        delta_hd_norm = delta_hd / (delta_hd.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        cos_sim = (delta_hd_norm * v_race.view(1, -1)).sum(dim=-1)  # [N]

        # Apply sign based on fact race: white -> +1, black -> -1, others ignored
        fact_races = [str(rec.get("race", "")).lower() for rec in fact_data]
        sign_list = []
        for r in fact_races:
            if "white" in r:
                sign_list.append(1.0)
            elif "black" in r:
                sign_list.append(-1.0)
            else:
                sign_list.append(0.0)
        sign = torch.tensor(sign_list, device=cos_sim.device, dtype=cos_sim.dtype)

        signed_cos = cos_sim * sign
        valid = sign != 0
        if torch.any(valid):
            mlp_input_delta_cos_race_signed[l] = float(signed_cos[valid].mean().item())
        else:
            mlp_input_delta_cos_race_signed[l] = float("nan")

        # Calculate p_race difference (mean absolute difference between fact and cf)
        fact_race_vocab_probs = torch.softmax(fact_logits.float(), dim=-1)  # [N, V]
        cf_race_vocab_probs = torch.softmax(cf_logits.float(), dim=-1)      # [N, V]
        
        p_race_fact_vec = fact_race_vocab_probs[:, race_ids].sum(dim=-1) # [N]
        p_race_cf_vec = cf_race_vocab_probs[:, race_ids].sum(dim=-1)     # [N]
        
        mlp_input_mean_abs_diff_p_race[l] = float(torch.abs(p_race_fact_vec - p_race_cf_vec).mean().item())

        fact_cand_probs = torch.softmax(fact_logits[:, cand_ids].float(), dim=-1)
        cf_cand_probs = torch.softmax(cf_logits[:, cand_ids].float(), dim=-1)

        p_yes_fact = fact_cand_probs[:, yes_mask].sum(dim=-1)
        p_yes_cf = cf_cand_probs[:, yes_mask].sum(dim=-1)

        p_no_fact = 1.0 - p_yes_fact
        p_no_cf = 1.0 - p_yes_cf

        # 计算差的绝对值的均值
        mlp_mean_diff[l] = float(torch.abs(p_yes_fact - p_yes_cf).mean().item())

        P_fact = torch.stack([p_yes_fact, p_no_fact], dim=-1)
        P_cf = torch.stack([p_yes_cf, p_no_cf], dim=-1)

        kl_vec = _kl_pq(P_fact, P_cf).detach().cpu().numpy()
        mlp_kl[l] = float(np.mean(kl_vec))

    np.save(os.path.join(args.output_dir, "mlp_kl_p_yes.npy"), mlp_kl)
    np.save(os.path.join(args.output_dir, "mlp_mean_diff_p_yes.npy"), mlp_mean_diff)
    np.save(os.path.join(args.output_dir, "mlp_input_delta_cos_race_signed.npy"), mlp_input_delta_cos_race_signed)
    np.save(os.path.join(args.output_dir, "mlp_input_mean_abs_diff_p_race.npy"), mlp_input_mean_abs_diff_p_race)

    summary = {
        "model_path": args.model_path,
        "dataset_json_path": args.dataset_json_path,
        "biased_csv_path": args.biased_csv_path,
        "sample_size": args.sample_size,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "cand_ids": cand_ids_list,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "probe_surface": "next_mlp_input_cumulative_residual",
        "probe_norm": "none",
    }
    with open(os.path.join(args.output_dir, "mlp_summary_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved baseline MLP output metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
