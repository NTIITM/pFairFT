#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP21: Plot head metrics comparing prompt vs debiased_prompt for Qwen3-4B QID 12.
"""

import argparse
import json
import os
from typing import Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _load_selected_heads(json_path: str) -> Set[Tuple[int, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected: Set[Tuple[int, int]] = set()
    for item in data:
        selected.add((int(item.get("layer", 0)), int(item.get("head", 0))))
    return selected


def _split_points(
    values: np.ndarray,
    selected_heads: Set[Tuple[int, int]],
):
    num_layers, num_heads = values.shape

    xs_sel = []
    ys_sel = []
    xs_other = []
    ys_other = []

    for l in range(num_layers):
        for h in range(num_heads):
            v = float(values[l, h])
            if (l, h) in selected_heads:
                xs_sel.append(l)
                ys_sel.append(v)
            else:
                xs_other.append(l)
                ys_other.append(v)

    return (xs_other, ys_other, xs_sel, ys_sel)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_xticks([])
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--sensitive_heads_json",
        type=str,
        required=True,
        help="Path to selected_heads_elbow.json (exp2_old sensitive heads)",
    )
    args = parser.parse_args()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    selected_heads = _load_selected_heads(args.sensitive_heads_json)

    # Load data
    prompt_kl = np.load(os.path.join(args.input_dir, "prompt_kl.npy"))
    prompt_md = np.load(os.path.join(args.input_dir, "prompt_md.npy"))
    debiased_kl = np.load(os.path.join(args.input_dir, "debiased_prompt_kl.npy"))
    debiased_md = np.load(os.path.join(args.input_dir, "debiased_prompt_md.npy"))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharex=False, sharey=False)

    # ========== Mean Diff Plot (keep only right panel) ==========
    # ax.set_title("Mean |Δp(yes)| (Fact vs CF)", fontweight="bold")

    # others: always white
    xs_o, ys_o, xs_s, ys_s = _split_points(prompt_md, selected_heads)
    if xs_o:
        ax.scatter(xs_o, ys_o, c="white", edgecolors="black", alpha=0.6, label="Other heads")

    # key heads: prompt -> red dot
    if xs_s:
        ax.scatter(xs_s, ys_s, c="red", edgecolors="black", alpha=0.9, label="Key Heads (Original)")

    # key heads: debiased -> star
    _, _, xs_s2, ys_s2 = _split_points(debiased_md, selected_heads)
    if xs_s2:
        ax.scatter(
            xs_s2,
            ys_s2,
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Key Heads (debiased)",
        )

    ax.set_ylabel(r"$\mathcal{I}_{l,h}$", fontweight="bold")
    _style_axis(ax)

    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    fig.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
