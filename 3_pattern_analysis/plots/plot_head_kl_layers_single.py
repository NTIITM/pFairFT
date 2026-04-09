#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只画 Llama 8B (Meta-Llama-3-8B-Instruct) 一个子图，画图逻辑与 exp22/plot_comparison_exp22.py 一致：
- 单面板，Other heads（白点黑边）、Key Heads (Original)（红点）、Key Heads (MLP elbow)（* 标记）
- Legend 在左上角，Layer 为 x 轴，整数刻度。
数据与路径逻辑来自 exp20/plot_head_kl_layers.py（exp20 与 exp2_old）。
"""

import argparse
import json
import os
from typing import Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODEL_NAME = "Meta-Llama-3-8B-Instruct"
YLABEL_MATH = r"$\mathcal{I}_{l,h}$"


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
    """与 plot_comparison_exp22 一致：按 selected 分成 other / key 的 x,y 列表。"""
    num_layers, num_heads = values.shape
    xs_sel = []
    ys_sel = []
    xs_other = []
    ys_other = []

    for l in range(num_layers):
        for h in range(num_heads):
            v = float(np.abs(values[l, h]))
            if (l, h) in selected_heads:
                xs_sel.append(l)
                ys_sel.append(v)
            else:
                xs_other.append(l)
                ys_other.append(v)
    return xs_other, ys_other, xs_sel, ys_sel


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Layer", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_yticks([])
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def _plot_single(
    ax: plt.Axes,
    values: np.ndarray,
    selected_elbow: Set[Tuple[int, int]],
    selected_mlp: Optional[Set[Tuple[int, int]]],
) -> None:
    """在 ax 上画单图：other 白点、key(elbow) 红点、key(mlp) 星号，风格同 plot_comparison_exp22。"""
    xs_o, ys_o, xs_k, ys_k = _split_points(values, selected_elbow)
    if xs_o:
        ax.scatter(xs_o, ys_o, c="white", edgecolors="black", alpha=0.6, label="Other heads")
    if xs_k:
        ax.scatter(xs_k, ys_k, c="red", edgecolors="black", alpha=0.9, label="Key Heads")

    if selected_mlp:
        _, _, xs_k2, ys_k2 = _split_points(values, selected_mlp)
        if xs_k2:
            ax.scatter(
                xs_k2,
                ys_k2,
                marker="*",
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Key Heads with\nintervention on MLPs",
            )
    ax.set_ylabel(YLABEL_MATH, fontweight="bold")
    _style_axis(ax)
    ax.legend(loc="upper left", frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot single Llama 8B head metrics (KL and mean_diff), one panel per metric."
    )
    parser.add_argument(
        "--exp20_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp20",
    )
    parser.add_argument(
        "--exp2_old_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp2_old",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Default: <exp20_root>/plots_p_yes_all_models",
    )
    args = parser.parse_args()

    exp20_root = os.path.abspath(args.exp20_root)
    exp2_old_root = os.path.abspath(args.exp2_old_root)
    model_dir = os.path.join(exp20_root, MODEL_NAME)
    sens_dir = os.path.join(exp2_old_root, f"sensitive_heads_{MODEL_NAME}_top100")
    elbow_json = os.path.join(sens_dir, "selected_heads_elbow.json")
    mlp_json = os.path.join(model_dir, "selected_heads_mlp_elbow.json")

    if not os.path.isdir(model_dir):
        print(f"Error: Model dir not found: {model_dir}")
        return
    if not os.path.isfile(elbow_json):
        print(f"Error: Elbow heads not found: {elbow_json}")
        return

    output_root = args.output_root or os.path.join(exp20_root, "plots_p_yes_all_models")
    os.makedirs(output_root, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    selected_elbow = _load_selected_heads(elbow_json)
    selected_mlp = _load_selected_heads(mlp_json) if os.path.isfile(mlp_json) else None

    # KL plot
    kl_path = os.path.join(model_dir, "kl_p_yes.npy")
    if os.path.isfile(kl_path):
        vals = np.load(kl_path)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
        _plot_single(ax, vals, selected_elbow, selected_mlp)
        plt.tight_layout()
        out_kl = os.path.join(output_root, "llama_8b_kl_p_yes.pdf")
        fig.savefig(out_kl, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_kl}")

    # Mean diff plot
    md_path = os.path.join(model_dir, "mean_diff_p_yes.npy")
    if os.path.isfile(md_path):
        vals = np.load(md_path)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
        _plot_single(ax, vals, selected_elbow, selected_mlp)
        plt.tight_layout()
        out_md = os.path.join(output_root, "llama_8b_mean_diff_p_yes.pdf")
        fig.savefig(out_md, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_md}")

    if not os.path.isfile(kl_path) and not os.path.isfile(md_path):
        print("No kl_p_yes.npy or mean_diff_p_yes.npy found.")
    else:
        print(f"Done. Outputs in: {output_root}")


if __name__ == "__main__":
    main()
