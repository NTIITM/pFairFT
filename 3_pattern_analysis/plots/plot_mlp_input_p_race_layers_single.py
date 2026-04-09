#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只画 Llama 8B (Meta-Llama-3-8B-Instruct) 的 MLP input p(race) 单子图。
画图逻辑参照 exp8/plot_intervention_single_LLM.py：单图、图内 legend、无上方标题。
数据与路径同 exp20/plot_mlp_input_p_race_layers.py。
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODEL_NAME = "Meta-Llama-3-8B-Instruct"
YLABEL_MATH = r"$\Delta_S$"


def _set_font():
    """与 plot_intervention_single_LLM.py 一致；数学公式用 cm 正确显示。"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["mathtext.fontset"] = "cm"


def _plot_single(ax: plt.Axes, vals: np.ndarray) -> None:
    x = np.arange(len(vals), dtype=int)
    ax.plot(x, vals, marker="o", label="p(race)")
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel(YLABEL_MATH)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=16, frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot single Llama 8B per-layer MLP input p(race) mean abs diff."
    )
    parser.add_argument(
        "--exp20_root",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp20",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Default: <exp20_root>/plots_mlp_input_p_race_all_models",
    )
    args = parser.parse_args()

    exp20_root = os.path.abspath(args.exp20_root)
    model_dir = os.path.join(exp20_root, MODEL_NAME)
    output_root = args.output_root or os.path.join(
        exp20_root, "plots_mlp_input_p_race_all_models"
    )
    os.makedirs(output_root, exist_ok=True)

    in_name = "mlp_input_mean_abs_diff_p_race.npy"
    in_path = os.path.join(model_dir, in_name)
    if not os.path.isfile(in_path):
        print(f"Error: Not found {in_path}")
        return

    vals = np.load(in_path)
    if vals.ndim != 1:
        print(f"Error: Expected 1D array, got shape {vals.shape}")
        return

    _set_font()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _plot_single(ax, vals)
    fig.tight_layout()
    out_path = os.path.join(output_root, "llama_8b_mlp_input_p_race_abs_diff.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Done. Outputs in: {output_root}")


if __name__ == "__main__":
    main()
