#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP23: Plot head metric comparing exp4 vs exp5 on QID 33.

Plotting rule (per user request, following exp21/plot_head_expression_exp21.py style):
- Read `sensitive_heads_json`.
- Non-sensitive heads: white small circles.
- Sensitive heads:
  - exp5: red dots.
  - exp4: stars.

Input files under `input_dir`:
- exp4_md.npy
- exp5_md.npy

Output: PDF.
"""

import argparse
import json
import os
from typing import List, Set, Tuple

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


def _split_points(values: np.ndarray, selected_heads: Set[Tuple[int, int]]):
    L, H = values.shape
    xs_sel: List[int] = []
    ys_sel: List[float] = []
    xs_oth: List[int] = []
    ys_oth: List[float] = []

    for l in range(L):
        for h in range(H):
            v = float(values[l, h])
            if (l, h) in selected_heads:
                xs_sel.append(l)
                ys_sel.append(v)
            else:
                xs_oth.append(l)
                ys_oth.append(v)

    return xs_oth, ys_oth, xs_sel, ys_sel


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Layer", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks([])
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--sensitive_heads_json", type=str, required=True)
    args = p.parse_args()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    selected = _load_selected_heads(args.sensitive_heads_json)

    md_exp4 = np.load(os.path.join(args.input_dir, "exp4_md.npy"))
    md_exp5 = np.load(os.path.join(args.input_dir, "exp5_md.npy"))

    if md_exp4.shape != md_exp5.shape:
        raise ValueError(f"Shape mismatch: exp4 {md_exp4.shape} vs exp5 {md_exp5.shape}")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2), sharex=False, sharey=False)

    # Non-sensitive heads (white small balls) - use exp5 values as background coordinates
    xs_o, ys_o, xs_s5, ys_s5 = _split_points(md_exp5, selected)
    if xs_o:
        ax.scatter(
            xs_o,
            ys_o,
            c="white",
            edgecolors="black",
            alpha=0.6,
            s=25,
            label="Other heads",
        )

    # Sensitive heads: exp5 red dots
    if xs_s5:
        ax.scatter(
            xs_s5,
            ys_s5,
            c="red",
            edgecolors="black",
            alpha=0.9,
            s=35,
            label="Key Heads (Global)",
        )

    # Sensitive heads: exp4 stars (same head indices, y from exp4)
    xs_s4, ys_s4 = [], []
    L, H = md_exp4.shape
    for (l, h) in selected:
        if 0 <= l < L and 0 <= h < H:
            xs_s4.append(l)
            ys_s4.append(float(md_exp4[l, h]))

    if xs_s4:
        ax.scatter(
            xs_s4,
            ys_s4,
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.95,
            label="Key Heads (pFairFT)",
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
