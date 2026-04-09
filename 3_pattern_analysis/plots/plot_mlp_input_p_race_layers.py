#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot per-layer mean absolute difference in p(race) between fact and counterfactual MLP inputs.

Inputs per model (from exp20):
- mlp_input_mean_abs_diff_p_race.npy   shape [num_layers]

Output:
- Qwen and Llama group line plots for Mean(|p_race_fact - p_race_cf|).
"""

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


QWEN_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
LLAMA_MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct",
]

MODEL_DISPLAY = {
    "Qwen3-1.7B": "Qwen 1.7B",
    "Qwen3-4B": "Qwen 4B",
    "Qwen3-8B": "Qwen 8B",
    "Llama-3.2-1B-Instruct": "Llama 1B",
    "Llama-3.2-3B-Instruct": "Llama 3B",
    "Meta-Llama-3-8B-Instruct": "Llama 8B",
}

# Y-axis label（数学风格）
YLABEL_MATH = r"$\Delta_S$"


def _plot_group(exp20_root: str, models: List[str], in_name: str, out_path: str) -> None:
    if not models:
        return
    # 与 plot_mlp_output_kl_layers.py 一致的字体与布局
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_dir = os.path.join(exp20_root, model_name)
        in_path = os.path.join(model_dir, in_name)
        if not os.path.exists(in_path):
            ax.axis("off")
            continue

        vals = np.load(in_path)
        if vals.ndim != 1:
            ax.axis("off")
            continue

        x = np.arange(len(vals), dtype=int)
        ax.plot(x, vals, marker="o", label="p(race)")
        ax.set_xlabel("Layer")
        ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[0].set_ylabel(YLABEL_MATH)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-layer MLP-input p(race) for all models.")
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
    output_root = args.output_root or os.path.join(exp20_root, "plots_mlp_input_p_race_all_models")
    os.makedirs(output_root, exist_ok=True)

    _plot_group(
        exp20_root,
        QWEN_MODELS,
        in_name="mlp_input_mean_abs_diff_p_race.npy",
        out_path=os.path.join(output_root, "qwen_mlp_input_p_race_abs_diff.pdf"),
    )

    _plot_group(
        exp20_root,
        LLAMA_MODELS,
        in_name="mlp_input_mean_abs_diff_p_race.npy",
        out_path=os.path.join(output_root, "llama_mlp_input_p_race_abs_diff.pdf"),
    )


if __name__ == "__main__":
    main()
