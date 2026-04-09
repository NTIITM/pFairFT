#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize QK scores: Display relative importance of tokens after removing special symbols.
Automated to read all results from qk_scores_output and save to individual folders.
"""

import json
import os
import sys
import argparse
import re
import unicodedata
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Import utility functions from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from transformers import AutoTokenizer


def clean_token_text(text: str) -> str:
    """
    Clean special characters from token text for better visualization.
    Only keep ASCII letters and digits.
    """
    if not text:
        return ""
    
    # Unicode normalization: decompose then recompose (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Remove Llama special tokens: <|...|>
    text = re.sub(r'<\|[^|]*\|>', '', text)
    
    # Remove Qwen/XML-like tags: <...> or </...>
    text = re.sub(r'</?[^>]*>', '', text)
    
    # Remove the word "assistant" (case-insensitive, as whole word)
    text = re.sub(r'\bassistant\b', '', text, flags=re.IGNORECASE)
    
    # Remove all control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b-\u200d\ufeff\u2060-\u206f]', '', text)
    
    # Remove Unicode combining characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'M')
    
    # Only keep ASCII letters (a-z, A-Z) and digits (0-9)
    text = ''.join(char for char in text 
                   if (('a' <= char <= 'z') or ('A' <= char <= 'Z') or ('0' <= char <= '9')))
    
    return text


def plot_relative_importance(
    qk_scores: np.ndarray,
    token_texts: List[str],
    layer: int,
    head: int,
    output_dir: str,
    top_k: int = 30,
):
    """
    Plots the relative importance bar chart and saves associated data.
    Only includes tokens that are non-empty after cleaning.
    """
    # 1. Prepare cleaned tokens and their original scores
    cleaned_tokens_data = []
    for i, raw_text in enumerate(token_texts):
        cleaned = clean_token_text(raw_text)
        if cleaned:
            cleaned_tokens_data.append({
                "pos": i,
                "token": cleaned,
                "score": float(qk_scores[i])
            })
    
    if not cleaned_tokens_data:
        print(f"Warning: No valid tokens after cleaning for L{layer} H{head}")
        return

    # 2. Calculate relative importance (normalized among valid tokens)
    total_score = sum(d["score"] for d in cleaned_tokens_data)
    for d in cleaned_tokens_data:
        d["relative_importance"] = d["score"] / total_score if total_score > 0 else 0.0

    # Sort by relative importance
    sorted_data = sorted(cleaned_tokens_data, key=lambda x: x["relative_importance"], reverse=True)
    plot_data = sorted_data[:top_k]

    # 3. Save Data Files
    # Save importance data
    importance_path = os.path.join(output_dir, "importance.json")
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)
    
    # Save tokens (original vs cleaned)
    tokens_path = os.path.join(output_dir, "tokens.json")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump([{"pos": i, "raw": t, "cleaned": clean_token_text(t)} 
                  for i, t in enumerate(token_texts)], f, indent=2, ensure_ascii=False)
    
    # Save raw scores
    np.save(os.path.join(output_dir, "raw_scores.npy"), qk_scores)

    # 4. Plotting
    labels = [f"{d['token']}" for d in plot_data]
    importances = [d["relative_importance"] for d in plot_data]

    # Make the figure height scale with the number of bars so large fonts don't overflow
    n = max(len(labels), 1)
    fig_h = max(8.0, 0.45 * n + 3.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # Colors: max value in red (255,17,17), others in gray
    highlight_red = (255 / 255.0, 17 / 255.0, 17 / 255.0)
    base_gray = (0.7, 0.7, 0.7)
    colors = [base_gray for _ in importances]
    if importances:
        max_idx = int(np.argmax(importances))
        colors[max_idx] = highlight_red

    ax.barh(range(len(labels)), importances, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=24)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

    ax.invert_yaxis()  # Highest importance at the top
    ax.set_xlabel("Relative Importance", fontsize=24, fontweight='bold')
    ax.set_title(f"Layer {layer}, Head {head}", fontsize=28, fontweight='bold')

    # Add value labels (avoid clipping outside the right border)
    # (keep within axes and also leave enough right margin)

    right_max = max(importances) if importances else 0.0
    ax.set_xlim(0.0, right_max * 1.20 if right_max > 0 else 1.0)

    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
    for i, v in enumerate(importances):
        x_text = min(v + x_offset, ax.get_xlim()[1] - x_offset)
        ax.text(
            x_text,
            i,
            f"{v*100:.2f}%",
            va="center",
            ha="left",
            fontsize=18,
            fontweight="bold",
            clip_on=True,
        )

    # Give extra left/right margins so tick labels and annotations stay inside the canvas
    fig.subplots_adjust(left=0.35, right=0.98, top=0.92, bottom=0.10)

    plt.savefig(os.path.join(output_dir, f"importance_plot_L{layer}_H{head}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_all_results(qk_scores_json: str, model_path: str, output_root: str, top_k: int = 30):
    """
    Main entry point: reads the JSON, iterates over all heads, 
    and creates separate folders for each.
    """
    # Ensure absolute path for output_root
    output_root = os.path.abspath(output_root)
    
    if not os.path.exists(qk_scores_json):
        print(f"Error: JSON file not found at {qk_scores_json}")
        return

    with open(qk_scores_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    qk_scores_dict = {k: np.array(v, dtype=np.float32) for k, v in data["qk_scores"].items()}
    formatted_prompt = data.get("formatted_prompt", "")
    sequence_length = data["sequence_length"]

    if not formatted_prompt:
        print("Error: formatted_prompt not found in JSON. Please rerun analyze_qk_scores.py.")
        return

    # Load tokenizer to get token texts
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Use encode to get IDs first to ensure exact alignment with sequence_length
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    
    # Strictly align with sequence_length
    if len(input_ids) > sequence_length:
        input_ids = input_ids[:sequence_length]
    
    token_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]
    
    # Final check on alignment
    if len(token_texts) != sequence_length:
        print(f"Warning: token_texts length ({len(token_texts)}) != sequence_length ({sequence_length})")

    for key, scores in qk_scores_dict.items():
        # Parse key like 'layer_14_head_1'
        parts = key.split('_')
        layer, head = int(parts[1]), int(parts[3])
        
        # Create dedicated folder
        head_dir = os.path.join(output_root, f"L{layer}_H{head}")
        os.makedirs(head_dir, exist_ok=True)
        
        output_file = os.path.join(head_dir, f"importance_plot_L{layer}_H{head}.pdf")
        plot_relative_importance(scores, token_texts, layer, head, head_dir, top_k=top_k)
        print(f"Saved: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description="One-click visualization of QK scores")
    parser.add_argument(
        "--qk_scores_json",
        type=str,
        default="./qk_scores_output/qk_scores_full.json",
        help="Path to qk_scores_full.json file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (for tokenizer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_output",
        help="Output root directory for visualizations",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top scores to display in the plot",
    )
    args = parser.parse_args()
    
    plot_all_results(args.qk_scores_json, args.model_path, args.output_dir, top_k=args.top_k)
    print(f"\nAll visualizations and data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
