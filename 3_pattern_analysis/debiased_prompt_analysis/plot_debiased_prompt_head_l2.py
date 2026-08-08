#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot head activation L2 norms before and after debiased prompting."""

import argparse
import json
import os
from typing import List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


YLABEL_EXPR = r"$\|h_{\ell,\mathrm{head}}\|_2$"


def _load_selected_heads(json_path: str) -> Set[Tuple[int, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected: Set[Tuple[int, int]] = set()
    for item in data:
        selected.add((int(item.get("layer", 0)), int(item.get("head", 0))))
    return selected


def _scatter_expr(
    ax: plt.Axes,
    expr: np.ndarray,
    selected_heads: Set[Tuple[int, int]],
    star_heads: Set[Tuple[int, int]] | None = None,
) -> None:
    L, H = expr.shape
    xs_sel: List[int] = []
    ys_sel: List[float] = []
    xs_oth: List[int] = []
    ys_oth: List[float] = []

    for l in range(L):
        for h in range(H):
            v = float(expr[l, h])
            if (l, h) in selected_heads:
                xs_sel.append(l)
                ys_sel.append(v)
            else:
                xs_oth.append(l)
                ys_oth.append(v)

    if xs_oth:
        ax.scatter(xs_oth, ys_oth, facecolors="none", edgecolors="black", alpha=0.6, label="Other heads")
    if xs_sel:
        ax.scatter(xs_sel, ys_sel, c="red", edgecolors="black", alpha=0.9, label="Sensitive heads")

    if star_heads:
        xs_star: List[int] = []
        ys_star: List[float] = []
        for (l, h) in star_heads:
            if 0 <= l < L and 0 <= h < H:
                xs_star.append(l)
                ys_star.append(float(expr[l, h]))
        if xs_star:
            ax.scatter(
                xs_star,
                ys_star,
                marker="*",
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Star heads",
            )

    ax.set_xlabel("Layer", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def _plot_three(expr_prompt: np.ndarray, expr_deb: np.ndarray, selected: Set[Tuple[int, int]], out_path: str) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)

    axes[0].set_title("Sensitive heads (prompt)", fontweight="bold")
    _scatter_expr(axes[0], expr_prompt, selected)

    axes[1].set_title("Other heads (prompt)", fontweight="bold")
    _scatter_expr(axes[1], expr_prompt, set())

    axes[2].set_title("Sensitive heads (debiased)", fontweight="bold")
    _scatter_expr(axes[2], expr_deb, selected)

    axes[0].set_ylabel(YLABEL_EXPR, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis_dir",
        type=str,
        default="",
        help="Directory containing expr_l2_prompt.npy and expr_l2_debiased_prompt.npy.",
    )
    p.add_argument(
        "--exp21_model_dir",
        type=str,
        default="",
        help="Deprecated alias for --analysis_dir.",
    )
    p.add_argument(
        "--selected_heads_json",
        type=str,
        required=True,
        help="Path to selected_heads_elbow.json from the matching head-selection run.",
    )
    p.add_argument("--out_path", type=str, required=True)
    args = p.parse_args()

    analysis_dir = args.analysis_dir or args.exp21_model_dir
    if not analysis_dir:
        raise ValueError("Provide --analysis_dir (or deprecated --exp21_model_dir).")

    expr_prompt = np.load(os.path.join(analysis_dir, "expr_l2_prompt.npy"))
    debiased_path = os.path.join(analysis_dir, "expr_l2_debiased_prompt.npy")
    if not os.path.exists(debiased_path):
        debiased_path = os.path.join(analysis_dir, "expr_l2_debiased.npy")
    expr_deb = np.load(debiased_path)

    selected = _load_selected_heads(args.selected_heads_json)

    _plot_three(expr_prompt, expr_deb, selected, args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
