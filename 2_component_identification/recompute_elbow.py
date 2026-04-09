import os
import sys
import json
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from util import compute_elbow_point, compute_elbow_point_by_acceleration
from plot import plot_elbow_point_vs_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/common1/hwluo/project/pFairFT/results/Qwen1.5-MoE-A2.7B-Chat/sensitive_heads")
    args = parser.parse_args()

    results_path = os.path.join(args.results_dir, "results.pkl")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    print(f"Loading {results_path}...")
    with open(results_path, "rb") as f:
        results_data = pickle.load(f)

    heatmap_kl = results_data["heatmap"]
    num_layers, num_heads = heatmap_kl.shape

    flat_kl = heatmap_kl.flatten()
    valid_kl = flat_kl[np.isfinite(flat_kl)]
    
    if len(valid_kl) == 0:
        print("Error: No valid KL values found.")
        return

    sorted_scores = np.sort(valid_kl)[::-1]

    # Original Method
    orig_idx, orig_score = compute_elbow_point(sorted_scores)
    orig_selected = []
    for l in range(num_layers):
        for h in range(num_heads):
            if np.isfinite(heatmap_kl[l, h]) and heatmap_kl[l, h] >= orig_score:
                orig_selected.append({"layer": l, "head": h})

    print("-" * 50)
    print("Original Distance Method:")
    print(f"Elbow Score: {orig_score:.6f}")
    print(f"Elbow Rank: {orig_idx + 1}")
    print(f"Selected Heads: {len(orig_selected)}")

    # New Acceleration Method
    acc_idx, acc_score = compute_elbow_point_by_acceleration(sorted_scores)
    acc_selected = []
    for l in range(num_layers):
        for h in range(num_heads):
            if np.isfinite(heatmap_kl[l, h]) and heatmap_kl[l, h] >= acc_score:
                acc_selected.append({"layer": l, "head": h})

    print("-" * 50)
    print("New Acceleration (2nd Derivative) Method:")
    print(f"Elbow Score: {acc_score:.6f}")
    print(f"Elbow Rank: {acc_idx + 1}")
    print(f"Selected Heads: {len(acc_selected)}")
    print("-" * 50)

    # Save new selected heads
    out_json = os.path.join(args.results_dir, "selected_heads_elbow_v2.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(acc_selected, f, indent=2)
    print(f"Saved new selection to {out_json}")

    # Plot new elbow point
    out_png = os.path.join(args.results_dir, "elbow_point_vs_rank_v2.png")
    plot_elbow_point_vs_rank(
        heatmap_kl,
        acc_idx,
        acc_score,
        out_png,
        title="Elbow Point vs Rank (Acceleration Method)"
    )
    print(f"Saved new plot to {out_png}")

if __name__ == "__main__":
    main()
