#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP23: Head-level fairness violation comparison on QID.

We compute per-head metric: mean |Δp_yes| between factual/counterfactual paired samples.
Compare exp4 vs exp5 adapters on base model Meta-Llama-3-8B-Instruct.

Outputs under output_dir:
- exp4_md.npy
- exp5_md.npy
"""

import argparse
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
from util import get_model_config
from hook import get_activation_hook_for_intervention, get_last_token_indices_safe
from sampling import load_discrim_eval_pairs


def _get_hf_layers_container(model):
    """Return the module that owns `layers` for llama/qwen style models.

    Works for both raw HF model and PEFT-wrapped model.
    """
    m = model
    # For PeftModel, base_model.model is the LlamaForCausalLM
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    
    # Common: LlamaForCausalLM -> .model (LlamaModel) -> .layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model

    # Some wrappers may already expose .layers
    if hasattr(m, "layers"):
        return m

    raise AttributeError("Cannot locate layers container (expected .model.layers)")


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
    device,
    num_layers,
    num_heads,
    head_dim,
):
    layers_owner = _get_hf_layers_container(model)

    batch_activations_buffer: Dict[int, torch.Tensor] = {}
    hooks = []
    for l in range(num_layers):
        layer_module = layers_owner.layers[l].self_attn.o_proj
        hook_fn = get_activation_hook_for_intervention(
            l, num_heads, head_dim, batch_activations_buffer
        )
        hooks.append(layer_module.register_forward_hook(hook_fn))

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


def compute_md(model, tokenizer, fact_acts, cf_acts, num_layers, num_heads, head_dim):
    layers_owner = _get_hf_layers_container(model)

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    cand_ids_list = list(dict.fromkeys(yes_ids + no_ids))
    if not cand_ids_list:
        raise ValueError("Could not resolve YES/NO candidate token IDs")

    # Access lm_head: model might be PeftModel
    m_for_head = model
    if hasattr(m_for_head, "base_model") and hasattr(m_for_head.base_model, "model"):
        m_for_head = m_for_head.base_model.model
    w_u = m_for_head.lm_head.weight

    mean_diff_p_yes = np.zeros((num_layers, num_heads), dtype=np.float64)
    yes_set = set(yes_ids)

    for l in range(num_layers):
        o_proj_weight = layers_owner.layers[l].self_attn.o_proj.weight
        o_proj_device = o_proj_weight.device
        o_dtype = o_proj_weight.dtype

        w_u_on_device = w_u.to(device=o_proj_device, dtype=o_dtype)
        cand_ids = torch.tensor(cand_ids_list, device=o_proj_device)
        yes_mask = torch.tensor(
            [tid in yes_set for tid in cand_ids_list],
            dtype=torch.bool,
            device=o_proj_device,
        )

        for h in range(num_heads):
            key = (l, h)
            if key not in fact_acts or key not in cf_acts:
                continue

            f_hd = torch.from_numpy(fact_acts[key]).to(device=o_proj_device, dtype=o_dtype)
            c_hd = torch.from_numpy(cf_acts[key]).to(device=o_proj_device, dtype=o_dtype)

            start = h * head_dim
            end = (h + 1) * head_dim
            o_slice = o_proj_weight[:, start:end]

            f_logits = (f_hd @ o_slice.t()) @ w_u_on_device.t()
            c_logits = (c_hd @ o_slice.t()) @ w_u_on_device.t()

            f_probs = torch.softmax(f_logits[:, cand_ids].float(), dim=-1)
            c_probs = torch.softmax(c_logits[:, cand_ids].float(), dim=-1)

            p_y_f = f_probs[:, yes_mask].sum(dim=-1)
            p_y_c = c_probs[:, yes_mask].sum(dim=-1)

            mean_diff_p_yes[l, h] = (p_y_f - p_y_c).abs().mean().item()

    return mean_diff_p_yes


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

    config = get_model_config(base_model)
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

    model_peft = PeftModel.from_pretrained(base_model, adapter_path,
        trust_remote_code=True)
    model_peft.eval()

    f_acts, c_acts = collect_acts(
        model_peft, tokenizer, dl, model_type, device, num_layers, num_heads, head_dim
    )
    md = compute_md(
        model_peft, tokenizer, f_acts, c_acts, num_layers, num_heads, head_dim
    )

    # Clean up
    del model_peft
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--exp4_adapter", type=str, required=True)
    parser.add_argument("--exp5_adapter", type=str, required=True)
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

    # Run exp4
    md_exp4 = run_experiment(
        args.exp4_adapter, args.base_model_path, args.dataset_path, args.qid, args.batch_size
    )
    np.save(os.path.join(args.output_dir, "exp4_md.npy"), md_exp4)

    # Run exp5
    md_exp5 = run_experiment(
        args.exp5_adapter, args.base_model_path, args.dataset_path, args.qid, args.batch_size
    )
    np.save(os.path.join(args.output_dir, "exp5_md.npy"), md_exp5)


if __name__ == "__main__":
    main()
