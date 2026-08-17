#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate p(yes) on Resume dataset top-100 biased samples (per exp2 ranking).

For each model/config (baseline / exp4 / exp5_KL / exp5_CE), this script:
- Loads resume dataset JSON
- Uses a CSV ranking file (from exp2/biased_samples_*/biased_samples_ranking.csv)
  to select the top-N samples by the CSV `index` column (sample_size, default 100)
- Builds fact and counterfactual prompts using build_category_prompt (方式A)
- Adds yes/no instruction
- Computes p(yes) on fact and counterfactual prompts
- Saves a CSV with columns: index, fact_p_yes, cf_p_yes, fact_race, cf_race
"""

import argparse
import csv
import json
import os
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from prompt import (
    add_yes_no_instruction,
    build_resume_prompt,
    create_debiased_prompt,
    format_prompt_for_model,
    resolve_model_type,
)
from probability import (
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
    compute_p_yes_batch,
)
from util import extract_race_from_query, create_counterfactual_by_race, get_input_device
from sampling import load_samples_by_csv_indices


def _load_resume_samples_by_csv_indices(
    dataset_path: str,
    csv_path: str,
    sample_size: int,
) -> List[dict]:
    """Wrapper around exp2 sampling: load resume data and select by CSV indices."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Resume dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Resume dataset must be a list of records.")

    sampled_data, used_indices, _ = load_samples_by_csv_indices(
        dataset=dataset,
        csv_path=csv_path,
        sample_size=sample_size,
    )

    # Attach original index for saving
    for rec, idx in zip(sampled_data, used_indices):
        rec["_orig_index"] = int(idx)
    return sampled_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Model configuration type (for bookkeeping only).")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base HF model.")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Optional LoRA/finetune adapter path. Empty for baseline.")
    parser.add_argument("--dataset_json_path", type=str, default=
                        "/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
                        help="Resume dataset JSON path.")
    parser.add_argument("--biased_csv_path", type=str, required=True,
                        help="Path to biased_samples_ranking.csv from exp2.")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of top samples to take from CSV order.")
    parser.add_argument("--output_csv_path", type=str, required=True,
                        help="Where to write per-sample results CSV.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"])
    parser.add_argument(
        "--resume_prompt_mode",
        type=str,
        default="category",
        choices=["summary_only", "category", "no_job_description"],
        help="Resume prompt body before the strict Yes/No instruction.",
    )
    parser.add_argument(
        "--prompt_type",
        choices=["prompt", "debiased_prompt"],
        default="prompt",
        help="Optionally append the canonical debiasing instruction.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)

    # 1. Load samples according to CSV indices (top-k)
    print(f"Loading resume samples (top-{args.sample_size}) using indices from: {args.biased_csv_path}")
    sampled_data = _load_resume_samples_by_csv_indices(
        dataset_path=args.dataset_json_path,
        csv_path=args.biased_csv_path,
        sample_size=args.sample_size,
    )
    print(f"Loaded {len(sampled_data)} samples from resume dataset.")

    # 2. Build fact / counterfactual records using build_category_prompt (方式A)
    fact_prompts_raw: List[str] = []
    cf_prompts_raw: List[str] = []
    fact_races: List[str] = []
    cf_races: List[str] = []
    indices: List[int] = []

    for item in sampled_data:
        summary = item.get("summary", "")
        category = item.get("category", "")
        race = item.get("race", "")
        if not summary:
            continue
        orig_index = int(item.get("_orig_index", item.get("ID", 0)))
        indices.append(orig_index)

        base_query = build_resume_prompt(
            summary=summary,
            category=category,
            mode=args.resume_prompt_mode,
        )
        extracted_race = extract_race_from_query(base_query) or race or "Unknown"

        fact_item = {
            "query": base_query,
            "summary": summary,
            "category": category,
            "race": extracted_race,
            "ID": orig_index,
        }
        cf_item = create_counterfactual_by_race(fact_item)

        fact_query = fact_item["query"]
        cf_query = cf_item["query"]
        if args.prompt_type == "debiased_prompt":
            fact_query = create_debiased_prompt(fact_query)
            cf_query = create_debiased_prompt(cf_query)

        fact_query_with_instr = add_yes_no_instruction(fact_query)
        cf_query_with_instr = add_yes_no_instruction(cf_query)

        fact_prompts_raw.append(fact_query_with_instr)
        cf_prompts_raw.append(cf_query_with_instr)
        fact_races.append(fact_item["race"])
        cf_races.append(cf_item.get("race", ""))

    if not indices:
        raise ValueError("No valid samples after filtering by summary/category.")

    print(f"Effective valid samples: {len(indices)}")

    # 3. Load model & tokenizer
    print(f"Loading base model from {args.base_model_path}")
    device_map = "auto" if args.device == "cuda" and torch.cuda.is_available() else None
    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = base_model
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path,
            trust_remote_code=True)

    model.eval()

    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.base_model_path)
    print(f"Using model_type={model_type}, input_device={input_device}")

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Failed to resolve yes/no token IDs.")

    # 4. Compute p(yes) for fact & cf prompts
    print("Computing p(yes) for fact prompts...")
    fact_p_yes_list = compute_p_yes_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=fact_prompts_raw,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing fact p(yes) [{args.mode}]",
        show_warnings=False,
    )

    print("Computing p(yes) for counterfactual prompts...")
    cf_p_yes_list = compute_p_yes_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=cf_prompts_raw,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing cf p(yes) [{args.mode}]",
        show_warnings=False,
    )

    if len(fact_p_yes_list) != len(indices) or len(cf_p_yes_list) != len(indices):
        raise RuntimeError("Length mismatch between prompts and returned probabilities.")

    # 5. Save to CSV
    with open(args.output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "fact_p_yes", "cf_p_yes", "fact_race", "cf_race"])
        for idx, fp, cp, fr, cr in zip(indices, fact_p_yes_list, cf_p_yes_list, fact_races, cf_races):
            writer.writerow([
                idx,
                f"{float(fp):.6f}" if fp is not None else "NaN",
                f"{float(cp):.6f}" if cp is not None else "NaN",
                fr,
                cr,
            ])

    metadata = {
        "dataset": "resume_top100",
        "dataset_json_path": os.path.abspath(args.dataset_json_path),
        "biased_csv_path": os.path.abspath(args.biased_csv_path),
        "base_model_path": os.path.abspath(args.base_model_path),
        "adapter_path": os.path.abspath(args.adapter_path) if args.adapter_path else None,
        "mode": args.mode,
        "model_type": model_type,
        "resume_prompt_mode": args.resume_prompt_mode,
        "prompt_type": args.prompt_type,
        "sample_size": args.sample_size,
        "num_output_rows": len(indices),
        "csv_path": os.path.abspath(args.output_csv_path),
    }
    with open(args.output_csv_path + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to {args.output_csv_path}")


if __name__ == "__main__":
    main()
