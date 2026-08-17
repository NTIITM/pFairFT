#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Cross Entropy on MMLU to measure capability preservation.
Calculates the standard Cross-Entropy Loss against the correct answer label.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from prompt import format_prompt_for_model, resolve_model_type

CHOICE_LETTERS = ["A", "B", "C", "D"]

def build_mmlu_prompt(question, choices):
    choices_block = "\n".join(f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, choices))
    return (
        "You are a knowledgeable AI assistant. "
        "Please answer the following multiple-choice question by choosing one option.\n\n"
        f"Question: {question}\n"
        f"Options:\n{choices_block}\n\n"
        "Answer with the single letter of the correct option (A, B, C, or D)."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading base model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if args.adapter_path:
        print(f"Loading peft adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path, trust_remote_code=True)
    
    model.eval()
    model_type = resolve_model_type(
        args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Using model_type={model_type}")

    download_config = DownloadConfig(local_files_only=os.getenv("HF_DATASETS_OFFLINE") == "1")
    ds = load_dataset(
        "cais/mmlu",
        "all",
        split=args.split,
        download_config=download_config,
    )
    
    # Evaluate on the entire split to get the true holistic CE (expected ~2.478)
    samples = list(ds)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    
    ce_losses = []
    loss_fct = torch.nn.CrossEntropyLoss()

    for idx, row in enumerate(tqdm(samples)):
        user_prompt = build_mmlu_prompt(row["question"], row["choices"])
        full_prompt = format_prompt_for_model(user_prompt, model_type)
        
        target_letter = CHOICE_LETTERS[int(row["answer"])]
        full_prompt += target_letter
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ce_losses.append(loss)

    final_ce = float(np.mean(ce_losses))
    print(f"MMLU Absolute CE (Standard Loss): {final_ce:.4f}")

    # Output back to JSON format expected by aggregation scripts
    res = {
        "ce": final_ce,
        "count": len(samples),
        "dataset": "cais/mmlu",
        "split": args.split,
        "model_path": os.path.abspath(args.model_path),
        "adapter_path": os.path.abspath(args.adapter_path) if args.adapter_path else None,
        "model_type": model_type,
        "ce_definition": "mean full-sequence causal-lm loss on formatted prompt plus correct answer letter",
    }
    
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
