#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate p(yes) probabilities for models trained in exp4 (full) and exp5 (LoRA).
Reference: exp1/evaluate_bias.py
"""

import argparse
import csv
import math
import os
import sys
import fcntl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from probability import (
    get_target_token_ids, YES_CANDIDATES, NO_CANDIDATES,
    compute_p_yes_batch
)
from util import get_input_device
from prompt import build_category_prompt, resolve_model_type, add_yes_no_instruction
from sampling import load_discrim_eval_pairs


MODEL_NAME_CANONICAL = {
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
}


def infer_base_model_name(base_model_path: str) -> str:
    name = os.path.basename(os.path.normpath(base_model_path))
    for key in MODEL_NAME_CANONICAL:
        if key in base_model_path or key in name:
            return MODEL_NAME_CANONICAL[key]
    return name


def main():
    parser = argparse.ArgumentParser(description="Evaluate p(yes) for exp4 and exp5 models.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/discrim-eval/dataset_paired.json",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (for LoRA) or the full model path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter (if applicable).",
    )
    parser.add_argument("--mode", type=str, required=True,
                        help="Configuration (e.g. baseline, exp4, exp5_KL, etc.) for tracking.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model_name_suffix",
        type=str,
        default="",
        help="Suffix to append to model name in CSV.",
    )
    args = parser.parse_args()

    data, pairs = load_discrim_eval_pairs(args.dataset_path)
    matched_id_map = {}
    for a, b in pairs:
        matched_id_map[a] = b
        matched_id_map[b] = a

    print(f"Loading base model from {args.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True
    )

    if args.adapter_path:
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path,
            trust_remote_code=True)

    base_model_name = infer_base_model_name(args.base_model_path)
    if args.model_name_suffix:
        model_name = f"{base_model_name}_{args.model_name_suffix}"
    else:
        model_name = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.base_model_path)

    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)

    prompt_type = "prompt"

    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)

    print(f"Evaluating {prompt_type} for {model_name}...")
    prompts = [add_yes_no_instruction(build_category_prompt(item[prompt_type], "")) for item in data]
    p_yes_results = compute_p_yes_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=str(input_device),
        yes_ids=yes_ids,
        no_ids=no_ids,
        model_type=model_type,
        desc=f"Computing p(yes) ({prompt_type})",
    )

    rows = []
    for sample, p_yes in zip(data, p_yes_results):
        if p_yes is None or math.isnan(p_yes):
            continue
        s_id = int(sample["id"])
        rows.append(
            [
                s_id,
                matched_id_map.get(s_id, ""),
                prompt_type,
                model_name,
                sample.get("decision_question_id", ""),
                float(p_yes),
            ]
        )

    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
    with open(args.csv_path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            file_empty = f.tell() == 0
            writer = csv.writer(f)
            if file_empty:
                writer.writerow(["sample_id", "matched_id", "prompt_type", "model", "decision_question_id", "p_yes"])
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    print(f"Finished evaluating {model_name}. Results saved to {args.csv_path}")


if __name__ == "__main__":
    main()
