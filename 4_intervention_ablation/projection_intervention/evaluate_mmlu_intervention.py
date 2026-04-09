#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate MMLU performance under Projection Intervention (All vs Partial heads).
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import utilities from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from hook import (
    make_intervention_hook_debias_projection,
    remove_intervention_hooks,
    create_config_detection_hook,
)
from util import get_model_config, get_input_device
from prompt import format_prompt_for_model, resolve_model_type

CHOICE_LETTERS = ["A", "B", "C", "D"]

def build_mmlu_prompt(question: str, choices: List[str]) -> str:
    choices_block = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, choices))
    prompt = (
        "You are a knowledgeable AI assistant. Please answer the following multiple-choice question by choosing one option.\n\n"
        f"Question: {question}\nOptions:\n{choices_block}\n\n"
        "Answer with the single letter of the correct option (A, B, C, or D)."
    )
    return prompt

def get_choice_logit_with_intervention(
    model, tokenizer, prompt, choice_letter, device, 
    intervention_mode, target_heads, white_emb, black_emb, num_heads, head_dim, intervention_strength, model_type
) -> float:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out_pos = int(inputs["input_ids"].shape[1] - 1)

    hooks = []
    if intervention_mode != "baseline":
        for l, h in target_heads:
            if (l, h) in white_emb and (l, h) in black_emb:
                w = torch.from_numpy(white_emb[(l, h)]).float()
                b = torch.from_numpy(black_emb[(l, h)]).float()
                hook_fn = make_intervention_hook_debias_projection(
                    l, h, w, b, None, out_pos, intervention_strength, num_heads, head_dim
                )
                hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_fn))
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            first_step_logits = outputs.logits[:, -1, :]
            token_ids = tokenizer(choice_letter, add_special_tokens=False).input_ids
            token_id = token_ids[0]
            return first_step_logits[0, token_id].item()
    finally:
        remove_intervention_hooks(hooks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--intervention_mode", type=str, choices=["baseline", "all", "partial"], default="baseline")
    parser.add_argument("--intervention_strength", type=float, default=1.0)
    parser.add_argument("--sensitive_heads_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_type = resolve_model_type("auto", model=model, tokenizer=tokenizer, model_path=args.model_path)

    # Intervention Config
    target_heads, white_emb, black_emb = [], {}, {}
    num_heads, head_dim = 0, 0
    if args.intervention_mode != "baseline":
        if not args.sensitive_heads_dir:
            model_name = os.path.basename(os.path.normpath(args.model_path))
            args.sensitive_heads_dir = f"/home/common1/hwluo/project/pFairFT/exp2/sensitive_heads_{model_name}_top100"
        with open(os.path.join(args.sensitive_heads_dir, "results.pkl"), "rb") as f:
            emb_data = pickle.load(f)
        white_emb = {(int(k[0]), int(k[1])): v for k, v in emb_data.get("white_emb", {}).items() if isinstance(k, (tuple, list))}
        black_emb = {(int(k[0]), int(k[1])): v for k, v in emb_data.get("black_emb", {}).items() if isinstance(k, (tuple, list))}
        
        if args.intervention_mode == "partial":
            with open(os.path.join(args.sensitive_heads_dir, "selected_heads_elbow.json"), "r") as f:
                target_heads = [(h["layer"], h["head"]) for h in json.load(f)]
        else:
            target_heads = list(white_emb.keys())
        
        cfg = get_model_config(model)
        num_heads, head_dim = int(cfg["num_heads"]), int(cfg["head_dim"])

    dataset = load_dataset("cais/mmlu", "all", split=args.split)
    if args.max_samples > 0: dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    for idx, row in enumerate(tqdm(dataset, desc=f"MMLU {args.intervention_mode}")):
        prompt = format_prompt_for_model(build_mmlu_prompt(row["question"], row["choices"]), model_type)
        logits = [get_choice_logit_with_intervention(model, tokenizer, prompt, L, device, args.intervention_mode, target_heads, white_emb, black_emb, num_heads, head_dim, args.intervention_strength, model_type) for L in CHOICE_LETTERS]
        if np.argmax(logits) == row["answer"]: correct += 1

    acc = correct / len(dataset)
    with open(args.output_json, "w") as f:
        json.dump({"model": args.model_path, "mode": args.intervention_mode, "accuracy": acc, "total": len(dataset)}, f, indent=2)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__": main()
