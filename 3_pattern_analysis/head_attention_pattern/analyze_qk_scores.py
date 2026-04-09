#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze attention patterns (QK scores) of the final output token across different sensitive heads.
Load sensitive head information and the first sample from exp2, extract and save QK scores.
"""

import json
import os
import sys
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utility functions from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from util import get_model_config, get_input_device
from prompt import resolve_model_type, add_yes_no_instruction


def extract_qk_scores_via_attentions(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    sensitive_heads: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Extract attention weights using transformers' native output_attentions=True support.
    
    This is a more stable and compatible method that avoids monkey patching issues:
    - No need to manually handle RoPE (Rotary Position Embedding)
    - No need to manually manage KV Cache
    - Avoids redundant computation and precision issues
    
    Note: This method returns attention weights after softmax, not raw QK scores.
    If raw QK scores (before softmax) are needed, use hook methods instead.
    
    Extracts the attention distribution from the last token of the prompt (about to generate output)
    to all previous tokens.
    
    Args:
        model: Model object
        inputs: Model input dictionary (containing input_ids, attention_mask, etc.)
        sensitive_heads: List of sensitive heads [(layer, head), ...]
        
    Returns:
        Dictionary with (layer, head) as keys and attention weight tensors [seq_len] as values
    """
    qk_scores_buffer: Dict[Tuple[int, int], torch.Tensor] = {}
    
    with torch.no_grad():
        # Use output_attentions=True to get attention weights from all layers
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is a tuple containing attention matrices from each layer
    # Shape is typically [batch, num_heads, sequence_length, sequence_length]
    all_attentions = outputs.attentions
    
    if all_attentions is None:
        raise ValueError("Model did not return attentions. Make sure the model supports output_attentions=True")
    
    # Extract attention weights for sensitive heads
    for layer, head in sensitive_heads:
        if layer >= len(all_attentions):
            print(f"Warning: Layer {layer} out of range (model has {len(all_attentions)} layers)")
            continue
        
        # Get attention weights for this layer
        # Shape: [batch, num_heads, q_len, kv_len]
        attn_layer = all_attentions[layer]
        
        _, num_heads, q_len, _ = attn_layer.shape
        
        if head >= num_heads:
            print(f"Warning: Head {head} out of range for layer {layer} (layer has {num_heads} heads)")
            continue
        
        # Extract attention distribution at the last query position (i.e., prompt's last token)
        # This is the attention from the last token to all previous tokens when about to generate output
        # Shape: [kv_len]
        last_query_pos = q_len - 1
        qk_score = attn_layer[0, head, last_query_pos, :].detach().cpu()
        qk_scores_buffer[(layer, head)] = qk_score
    
    return qk_scores_buffer


def load_sensitive_heads(selected_heads_path: str) -> List[Tuple[int, int]]:
    """
    Load sensitive head information from JSON file.
    
    Args:
        selected_heads_path: Path to selected_heads_elbow.json file
        
    Returns:
        List of sensitive heads [(layer, head), ...]
    """
    with open(selected_heads_path, "r", encoding="utf-8") as f:
        heads_data = json.load(f)
    
    sensitive_heads = []
    for head_info in heads_data:
        layer = head_info["layer"]
        head = head_info["head"]
        sensitive_heads.append((layer, head))
    
    return sensitive_heads


def load_first_sample(csv_path: str, dataset_json_path: str) -> Dict[str, Any]:
    """
    Load the first sample (rank=1) from CSV file.
    
    Args:
        csv_path: Path to biased_samples_ranking.csv file
        dataset_json_path: Path to dataset JSON file
        
    Returns:
        Dictionary of the first sample
    """
    # Read CSV and find the sample with rank=1
    sample_index = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["rank"] == "1":
                sample_index = int(row["index"])
                break
    
    if sample_index is None:
        raise ValueError("Could not find rank=1 sample in CSV")
    
    # Load the corresponding sample from dataset JSON
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if sample_index >= len(dataset):
        raise ValueError(f"Sample index {sample_index} out of range (dataset size: {len(dataset)})")
    
    sample = dataset[sample_index]
    sample["index"] = sample_index
    
    return sample


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze QK scores for sensitive heads on the first sample"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory (e.g., Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--sensitive_heads_path",
        type=str,
        required=True,
        help="Path to selected_heads_elbow.json from exp2",
    )
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        required=False,
        help="Path to biased_samples_ranking.csv from exp2 (not used, kept for compatibility)",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Path to the dataset JSON file (not used, kept for compatibility)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qk_scores_output",
        help="Output directory for QK scores",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek"],
        help="Model architecture type",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.eval()
    
    input_device = get_input_device(model, args.device)
    print(f"Input tensors will be on device: {input_device}")
    
    model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
    print(f"Using model_type: {model_type}")
    
    # Get model configuration
    config = get_model_config(model)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    print(f"Model config: {num_layers} layers, {num_heads} heads per layer, head_dim={head_dim}")
    
    print("=" * 80)
    print("Loading sensitive heads and preparing test case...")
    print("=" * 80)
    
    # Load sensitive heads
    sensitive_heads = load_sensitive_heads(args.sensitive_heads_path)
    print(f"Loaded {len(sensitive_heads)} sensitive heads")
    
    # Use fixed test case
    candidate_description = "Black candidate has 26 years' expertise."
    print(f"Using test case: {candidate_description}")
    
    # Build prompt: candidate description + yes/no answering part
    prompt = f"{candidate_description} Do we hire Her?"
    formatted_prompt = prompt
    
    print("=" * 80)
    print("Sample prompt:")
    print("=" * 80)
    print(formatted_prompt)
    print("=" * 80)
    
    # Tokenize
    inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(input_device)
    
    print(f"Sequence length: {inputs['input_ids'].shape[1]}")
    print(f"Note: Extracting attention from the last query position (q_len - 1) in the attention matrix")
    
    print("=" * 80)
    print("Extracting QK scores using output_attentions=True...")
    print("=" * 80)
    print("This method is more stable and avoids monkey patching issues.")
    print("Extracting attention of the last token (about to generate output) to all previous tokens.")
    print("=" * 80)
    
    # Use output_attentions=True to extract attention weights
    # This method is more stable and avoids manual handling of RoPE, KV Cache, etc.
    qk_scores_buffer = extract_qk_scores_via_attentions(
        model,
        inputs,
        sensitive_heads,
    )
    
    print("=" * 80)
    print("Extracted QK scores:")
    print("=" * 80)
    
    # Save QK scores
    qk_scores_dict = {}
    for (layer, head), qk_score in qk_scores_buffer.items():
        qk_scores_dict[f"layer_{layer}_head_{head}"] = qk_score.numpy()
        print(f"Layer {layer}, Head {head}: QK score shape {qk_score.shape}, "
              f"min={qk_score.min():.4f}, max={qk_score.max():.4f}, mean={qk_score.mean():.4f}")
    
    # Save full results as JSON (no pickle)
    output_full_json_path = os.path.join(args.output_dir, "qk_scores_full.json")
    full_payload = {
        "score_type": "post_softmax_attention",
        "qk_scores": {k: v.tolist() for k, v in qk_scores_dict.items()},
        "sensitive_heads": [{"layer": l, "head": h} for l, h in sensitive_heads],
        "candidate_description": candidate_description,
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "formatted_prompt": formatted_prompt,
    }
    with open(output_full_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\nSaved full QK/attention results to {output_full_json_path}")

    # Save as JSON summary file
    output_json_path = os.path.join(args.output_dir, "qk_scores_summary.json")
    summary = {
        "score_type": "post_softmax_attention",
        "sensitive_heads": [{"layer": l, "head": h} for l, h in sensitive_heads],
        "candidate_description": candidate_description,
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "qk_scores_info": {
            f"layer_{l}_head_{h}": {
                "shape": list(qk_score.shape),
                "min": float(qk_score.min().item()),
                "max": float(qk_score.max().item()),
                "mean": float(qk_score.mean().item()),
                "std": float(qk_score.std().item()),
            }
            for (l, h), qk_score in qk_scores_buffer.items()
        },
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved QK scores summary to {output_json_path}")

    # Save as numpy files (one file per head)
    for (layer, head), qk_score in qk_scores_buffer.items():
        np_path = os.path.join(args.output_dir, f"qk_scores_layer_{layer}_head_{head}.npy")
        np.save(np_path, qk_score.numpy())
        print(f"Saved QK scores for layer {layer}, head {head} to {np_path}")
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
