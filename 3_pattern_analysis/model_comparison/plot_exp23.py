#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP23: Plot head metric comparing two adapters on a QID subset.

Plotting rule (per user request, following exp21/plot_head_expression_exp21.py style):
- Read `sensitive_heads_json`.
- Non-sensitive heads: white small circles.
- Sensitive heads:
  - second adapter: red dots.
  - first adapter: stars.

Input files under `input_dir`:
- first_md.npy
- second_md.npy

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
    p.add_argument("--first_label", type=str, default="PFairFT")
    p.add_argument("--second_label", type=str, default="Global LoRA CE")
    args = p.parse_args()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    selected = _load_selected_heads(args.sensitive_heads_json)

    first_md = np.load(os.path.join(args.input_dir, "first_md.npy"))
    second_md = np.load(os.path.join(args.input_dir, "second_md.npy"))

    if first_md.shape != second_md.shape:
        raise ValueError(f"Shape mismatch: first {first_md.shape} vs second {second_md.shape}")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2), sharex=False, sharey=False)

    # Non-sensitive heads use second-adapter values as background coordinates.
    xs_o, ys_o, xs_second, ys_second = _split_points(second_md, selected)
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

    # Sensitive heads: second adapter red dots.
    if xs_second:
        ax.scatter(
            xs_second,
            ys_second,
            c="red",
            edgecolors="black",
            alpha=0.9,
            s=35,
            label=f"Key Heads ({args.second_label})",
        )

    # Sensitive heads: first adapter stars at the same head indices.
    xs_first, ys_first = [], []
    L, H = first_md.shape
    for (l, h) in selected:
        if 0 <= l < L and 0 <= h < H:
            xs_first.append(l)
            ys_first.append(float(first_md[l, h]))

    if xs_first:
        ax.scatter(
            xs_first,
            ys_first,
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.95,
            label=f"Key Heads ({args.first_label})",
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
