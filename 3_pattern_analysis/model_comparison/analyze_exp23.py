#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP23: Head-level fairness violation comparison on QID.

We compute per-head metric: mean |Δp_yes| between factual/counterfactual paired samples.
Compare two adapters on a base model. Dense models use o_proj-input head
activations; MOE models use the activation surface defined by src/model_adapter.py.

Outputs under output_dir:
- first_md.npy
- second_md.npy
"""

import argparse
import json
import os

os.environ["TRANSFORMERS_NO_SKLEARN"] = "1"

import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import add_yes_no_instruction, format_prompt_for_model, resolve_model_type
from probability import NO_CANDIDATES, YES_CANDIDATES, get_target_token_ids
from hook import get_last_token_indices_safe
from model_adapter import get_model_adapter
from sampling import load_discrim_eval_pairs


class PairedQidDataset(Dataset):
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


def collect_acts(
    model,
    tokenizer,
    dataloader,
    model_type,
    adapter,
    device,
    num_layers,
    num_heads,
    head_dim,
):
    batch_activations_buffer: Dict[int, torch.Tensor] = {}
    hooks = []
    for l in range(num_layers):
        hook_fn = adapter.register_head_activation_hook(
            l, num_heads, head_dim, batch_activations_buffer
        )
        hooks.append(hook_fn)

    def _collect(is_fact: bool):
        all_acts: Dict[Tuple[int, int], List[np.ndarray]] = {}
        key = "fact_prompt" if is_fact else "cf_prompt"

        for batch in tqdm(dataloader, desc=f"Collecting {'Fact' if is_fact else 'CF'} Acts"):
            prompts = [format_prompt_for_model(p, model_type) for p in batch[key]]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(device)

            last_token_indices = get_last_token_indices_safe(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                tokenizer,
            )
            batch_range = torch.arange(inputs["input_ids"].shape[0], device=device)

            batch_activations_buffer.clear()
            with torch.no_grad():
                _ = model(**inputs)

            for l in range(num_layers):
                buf = batch_activations_buffer[l]  # [B, Seq, H, D]
                idx_device = buf.device
                act = buf[
                    batch_range.to(idx_device),
                    last_token_indices.to(idx_device),
                    :, :
                ]  # [B, H, D]

                act_np = act.detach().cpu().float().numpy()
                for h in range(num_heads):
                    k = (l, h)
                    if k not in all_acts:
                        all_acts[k] = []
                    all_acts[k].append(act_np[:, h, :])

        return {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}

    fact_acts = _collect(is_fact=True)
    cf_acts = _collect(is_fact=False)

    for h in hooks:
        h.remove()

    return fact_acts, cf_acts


def compute_md(adapter, tokenizer, fact_acts, cf_acts, num_layers, num_heads, head_dim):
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))
    if not cand_ids_list:
        raise ValueError("Could not resolve YES/NO candidate token IDs")

    mean_diff_p_yes = np.zeros((num_layers, num_heads), dtype=np.float64)
    yes_set = set(yes_ids)
    cand_ids = torch.tensor(cand_ids_list, dtype=torch.long)
    yes_mask = torch.tensor(
        [tid in yes_set for tid in cand_ids_list],
        dtype=torch.bool,
    )

    for l in range(num_layers):
        for h in range(num_heads):
            key = (l, h)
            if key not in fact_acts or key not in cf_acts:
                continue

            f_hd = torch.from_numpy(fact_acts[key])
            c_hd = torch.from_numpy(cf_acts[key])

            f_logits = adapter.project_head_activations_to_logits(
                l, h, f_hd, num_heads, head_dim
            )
            c_logits = adapter.project_head_activations_to_logits(
                l, h, c_hd, num_heads, head_dim
            )

            cand_ids_on_device = cand_ids.to(f_logits.device)
            yes_mask_on_device = yes_mask.to(f_logits.device)
            f_probs = torch.softmax(f_logits[:, cand_ids_on_device].float(), dim=-1)
            c_probs = torch.softmax(c_logits[:, cand_ids_on_device].float(), dim=-1)

            p_y_f = f_probs[:, yes_mask_on_device].sum(dim=-1)
            p_y_c = c_probs[:, yes_mask_on_device].sum(dim=-1)

            mean_diff_p_yes[l, h] = (p_y_f - p_y_c).abs().mean().item()

    return mean_diff_p_yes


def _is_baseline_adapter(adapter_path: str) -> bool:
    return adapter_path.strip().lower() in {"", "none", "baseline", "base"}


def run_experiment(adapter_path, base_model_path, dataset_path, qid, batch_size):
    print(f"\n--- Loading experiment: {adapter_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    base_model.eval()

    device = next(base_model.parameters()).device
    model_type = resolve_model_type(
        "auto", model=base_model, tokenizer=tokenizer, model_path=base_model_path
    )

    base_adapter = get_model_adapter(base_model, model_type=model_type, model_path=base_model_path)
    print(f"Using architecture adapter: {base_adapter.family} ({base_adapter.head_activation_kind})")
    config = base_adapter.get_config()
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]

    data, pairs = load_discrim_eval_pairs(dataset_path)
    id_to_sample = {int(item["id"]): item for item in data}

    target_pairs = []
    for a, b in pairs:
        a_int, b_int = int(a), int(b)
        if int(id_to_sample[a_int]["decision_question_id"]) != qid:
            continue
        target_pairs.append((id_to_sample[a_int], id_to_sample[b_int]))

    print(f"Found {len(target_pairs)} pairs for QID {qid}")
    if not target_pairs:
        raise RuntimeError(f"No pairs found for qid={qid}")

    ds = PairedQidDataset(target_pairs, "prompt")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if _is_baseline_adapter(adapter_path):
        eval_model = base_model
    else:
        eval_model = PeftModel.from_pretrained(base_model, adapter_path,
            trust_remote_code=True)
        eval_model.eval()
    peft_adapter = get_model_adapter(eval_model, model_type=model_type, model_path=base_model_path)

    f_acts, c_acts = collect_acts(
        eval_model, tokenizer, dl, model_type, peft_adapter, device, num_layers, num_heads, head_dim
    )
    md = compute_md(
        peft_adapter, tokenizer, f_acts, c_acts, num_layers, num_heads, head_dim
    )

    # Clean up
    del eval_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--first_adapter", type=str, required=True)
    parser.add_argument("--second_adapter", type=str, required=True)
    parser.add_argument("--first_label", type=str, default="PFairFT")
    parser.add_argument("--second_label", type=str, default="Global LoRA CE")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--qid", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        "base_model_path": args.base_model_path,
        "dataset_path": args.dataset_path,
        "qid": args.qid,
        "first_adapter": args.first_adapter,
        "second_adapter": args.second_adapter,
        "first_label": args.first_label,
        "second_label": args.second_label,
        "metric": "mean absolute factual-vs-counterfactual p_yes gap per head",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Running first adapter: {args.first_label}")
    first_md = run_experiment(
        args.first_adapter, args.base_model_path, args.dataset_path, args.qid, args.batch_size
    )
    np.save(os.path.join(args.output_dir, "first_md.npy"), first_md)

    print(f"Running second adapter: {args.second_label}")
    second_md = run_experiment(
        args.second_adapter, args.base_model_path, args.dataset_path, args.qid, args.batch_size
    )
    np.save(os.path.join(args.output_dir, "second_md.npy"), second_md)


if __name__ == "__main__":
    main()
