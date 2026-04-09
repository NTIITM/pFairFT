#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import pickle
import random
import sys
from typing import List, Tuple, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from prompt import add_yes_no_instruction, build_category_prompt, format_prompt_for_model, resolve_model_type
from probability import YES_CANDIDATES, NO_CANDIDATES, get_target_token_ids, compute_p_yes_batch
from util import extract_race_from_query, create_counterfactual_by_race, get_input_device
from sampling import load_samples_by_csv_indices
from hook import (
    make_intervention_hook_debias_projection,
    remove_intervention_hooks,
    create_config_detection_hook,
)
from util import get_model_config


def _load_resume_samples_by_csv_indices(
    dataset_path: str,
    csv_path: str,
    sample_size: int,
) -> List[dict]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Resume dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    sampled_data, used_indices, _ = load_samples_by_csv_indices(
        dataset=dataset,
        csv_path=csv_path,
        sample_size=sample_size,
    )
    for rec, idx in zip(sampled_data, used_indices):
        rec["_orig_index"] = int(idx)
    return sampled_data


def _detect_head_config(model, tokenizer, input_device, model_type: str, any_prompt: str) -> Tuple[int, int]:
    cfg = get_model_config(model)
    num_heads, head_dim = int(cfg["num_heads"]), int(cfg["head_dim"])

    temp: Dict[str, object] = {}
    hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(create_config_detection_hook(temp))
    try:
        test_inputs = tokenizer(
            [format_prompt_for_model(any_prompt, model_type)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(input_device)
        with torch.no_grad():
            _ = model(**test_inputs)
    finally:
        hook.remove()

    det_h = temp.get("num_heads")
    det_d = temp.get("head_dim")
    if det_h is not None and det_d is not None:
        num_heads, head_dim = int(det_h), int(det_d)
    return num_heads, head_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--dataset_json_path", type=str, default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json")
    parser.add_argument("--biased_csv_path", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "deepseek"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensitive_heads_dir", type=str, default="")
    parser.add_argument("--intervention_mode", type=str, choices=["all", "partial"], default="all")
    parser.add_argument("--intervention_strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)

    sampled_data = _load_resume_samples_by_csv_indices(
        dataset_path=args.dataset_json_path,
        csv_path=args.biased_csv_path,
        sample_size=args.sample_size,
    )

    fact_prompts_raw: List[str] = []
    cf_prompts_raw: List[str] = []
    fact_races: List[str] = []
    cf_races: List[str] = []
    indices: List[int] = []

    for item in sampled_data:
        summary = item.get("summary", "")
        category = item.get("category", "")
        race = item.get("race", "")
        if not summary or not category:
            continue

        orig_index = int(item.get("_orig_index", item.get("ID", 0)))
        indices.append(orig_index)

        base_query = build_category_prompt(summary, category)
        extracted_race = extract_race_from_query(base_query) or race or "Unknown"

        fact_item = {
            "query": base_query,
            "summary": summary,
            "category": category,
            "race": extracted_race,
            "ID": orig_index,
        }
        cf_item = create_counterfactual_by_race(fact_item)

        fact_prompts_raw.append(add_yes_no_instruction(fact_item["query"]))
        cf_prompts_raw.append(add_yes_no_instruction(cf_item["query"]))
        fact_races.append(fact_item["race"])
        cf_races.append(cf_item.get("race", ""))

    if not indices:
        raise ValueError("No valid samples after filtering by summary/category.")

    device_map = "auto" if args.device == "cuda" and torch.cuda.is_available() else None
    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.base_model_path)

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)

    if not args.sensitive_heads_dir:
        model_name = os.path.basename(os.path.normpath(args.base_model_path))
        args.sensitive_heads_dir = os.path.join(
            "/home/common1/hwluo/project/pFairFT/exp2",
            f"sensitive_heads_{model_name}_top100",
        )

    embeddings_path = os.path.join(args.sensitive_heads_dir, "results.pkl")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"embeddings results.pkl not found: {embeddings_path}")

    with open(embeddings_path, "rb") as f:
        emb = pickle.load(f)

    white_emb = {
        (int(k[0]), int(k[1])): v
        for k, v in emb.get("white_emb", {}).items()
        if isinstance(k, (tuple, list))
    }
    black_emb = {
        (int(k[0]), int(k[1])): v
        for k, v in emb.get("black_emb", {}).items()
        if isinstance(k, (tuple, list))
    }

    if args.intervention_mode == "partial":
        selected_path = os.path.join(args.sensitive_heads_dir, "selected_heads_elbow.json")
        if not os.path.exists(selected_path):
            raise FileNotFoundError(f"selected_heads_elbow.json not found: {selected_path}")
        with open(selected_path, "r", encoding="utf-8") as f:
            target_heads = [(h["layer"], h["head"]) for h in json.load(f)]
    else:
        target_heads = list(white_emb.keys())

    any_prompt = fact_prompts_raw[0]
    num_heads, head_dim = _detect_head_config(model, tokenizer, input_device, model_type, any_prompt)

    # compute p(yes) for fact/cf with per-sample forward hooks (because exp25 intervention)
    def _compute_with_intervention(prompts: List[str], desc: str) -> List[float]:
        p_list: List[float] = []
        for prompt in tqdm(prompts, desc=desc):
            formatted = format_prompt_for_model(prompt, model_type)
            input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(input_device)
            attention_mask = torch.ones_like(input_ids).to(input_device)

            # output position: last token
            out_pos = int(input_ids.shape[1] - 1)

            hooks = []
            for l, h in target_heads:
                if (l, h) not in white_emb or (l, h) not in black_emb:
                    continue
                w = torch.from_numpy(white_emb[(l, h)]).float()
                b = torch.from_numpy(black_emb[(l, h)]).float()
                hook_fn = make_intervention_hook_debias_projection(
                    l,
                    h,
                    w,
                    b,
                    None,
                    out_pos,
                    args.intervention_strength,
                    num_heads,
                    head_dim,
                    use_std=False,
                )
                hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_fn))

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_row = outputs.logits[0, out_pos, :].float()
                    # Reuse batch helper for single prompt to avoid duplicating logic
                    p = compute_p_yes_batch(
                        model=None,
                        tokenizer=tokenizer,
                        prompts=[],
                        device=str(input_device),
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                        model_type=model_type,
                    )
                    # Above call is not usable without model; compute directly
                    # Do direct compute like discrim-eval code path
                    from util import compute_p_yes_from_logits_with_warning

                    p_yes = compute_p_yes_from_logits_with_warning(
                        logits_row,
                        tokenizer,
                        yes_ids,
                        no_ids,
                    )
                    p_list.append(float(p_yes))
            finally:
                remove_intervention_hooks(hooks)

        return p_list

    fact_p_yes = _compute_with_intervention(fact_prompts_raw, f"fact p(yes) [{args.intervention_mode}]")
    cf_p_yes = _compute_with_intervention(cf_prompts_raw, f"cf p(yes) [{args.intervention_mode}]")

    with open(args.output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "fact_p_yes", "cf_p_yes", "fact_race", "cf_race", "intervention_mode"])
        for idx, fp, cp, fr, cr in zip(indices, fact_p_yes, cf_p_yes, fact_races, cf_races):
            writer.writerow([
                idx,
                f"{float(fp):.6f}" if fp is not None else "NaN",
                f"{float(cp):.6f}" if cp is not None else "NaN",
                fr,
                cr,
                args.intervention_mode,
            ])


if __name__ == "__main__":
    main()
