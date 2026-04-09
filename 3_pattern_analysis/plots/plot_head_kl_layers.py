#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP20 plotting: KL(p_yes) and mean diff of p_yes for all models.

Inputs per model (produced by exp20/analyze_head_kl_resume.py and _mlp.py):
- exp20/<MODEL_NAME>/kl_p_yes.npy             shape [L, H] (baseline)
- exp20/<MODEL_NAME>/mean_diff_p_yes.npy      shape [L, H] (baseline)
- exp20/<MODEL_NAME>/kl_p_yes_mlp.npy         shape [L, H] (MLP intervention)
- exp20/<MODEL_NAME>/mean_diff_p_yes_mlp.npy  shape [L, H] (MLP intervention)
- exp20/<MODEL_NAME>/selected_heads_mlp_elbow.json (heads selected by MLP KL elbow)

Coloring:
- Red: heads selected by exp2_old elbow criterion
  exp2_old/sensitive_heads_<MODEL_NAME>_top100/selected_heads_elbow.json
- White: other heads
- Black '*' marker: heads selected by MLP KL elbow (from selected_heads_mlp_elbow.json)

Figures:
- Two 1x3 group figures for Qwen models (KL, mean_diff) using baseline KL/mean_diff
- Two 1x3 group figures for Llama models (KL, mean_diff) using baseline KL/mean_diff

Tables:
- Per-model CSV table: model, layer, head, kl_p_yes, mean_diff_p_yes, selected_by_elbow
"""

import argparse
import csv
import json
import os
from typing import List, Set, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODEL_DISPLAY = {
    "Qwen3-1.7B": "Qwen 1.7B",
    "Qwen3-4B": "Qwen 4B",
    "Qwen3-8B": "Qwen 8B",
    "Llama-3.2-1B-Instruct": "Llama 1B",
    "Llama-3.2-3B-Instruct": "Llama 3B",
    "Meta-Llama-3-8B-Instruct": "Llama 8B",
}

QWEN_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
LLAMA_MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct",
]
MODEL_ORDER = QWEN_MODELS + LLAMA_MODELS

# Y-axis label (与 plot_mlp_output_kl_layers.py 一致)
YLABEL_MATH = r"$\mathcal{I}_{l,h}$"


def _load_selected_heads(json_path: str) -> Set[Tuple[int, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected: Set[Tuple[int, int]] = set()
    for item in data:
        selected.add((int(item.get("layer", 0)), int(item.get("head", 0))))
    return selected


def _discover_models(exp20_root: str) -> List[str]:
    models: List[str] = []
    if not os.path.isdir(exp20_root):
        return models
    for name in os.listdir(exp20_root):
        model_dir = os.path.join(exp20_root, name)
        if not os.path.isdir(model_dir):
            continue
        if os.path.isfile(os.path.join(model_dir, "kl_p_yes.npy")) and os.path.isfile(
            os.path.join(model_dir, "mean_diff_p_yes.npy")
        ):
            models.append(name)

    ordered: List[str] = []
    for m in MODEL_ORDER:
        if m in models:
            ordered.append(m)
    for m in sorted(models):
        if m not in ordered:
            ordered.append(m)
    return ordered


def _scatter_on_axis(
    ax: plt.Axes,
    values: np.ndarray,
    selected_heads: Set[Tuple[int, int]],
    selected_heads_mlp: Optional[Set[Tuple[int, int]]] = None,
) -> None:
    num_layers, num_heads = values.shape

    xs_selected: List[int] = []
    ys_selected: List[float] = []
    xs_other: List[int] = []
    ys_other: List[float] = []

    for l in range(num_layers):
        for h in range(num_heads):
            v = float(np.abs(values[l, h]))
            if (l, h) in selected_heads:
                xs_selected.append(l)
                ys_selected.append(v)
            else:
                xs_other.append(l)
                ys_other.append(v)

    if xs_other:
        ax.scatter(xs_other, ys_other, facecolors="none", edgecolors="black", alpha=0.6, label="Other heads")
    if xs_selected:
        ax.scatter(xs_selected, ys_selected, c="red", edgecolors="black", alpha=0.9, label="Identified Key heads")

    # MLP elbow: black star marker
    if selected_heads_mlp:
        xs_star: List[int] = []
        ys_star: List[float] = []
        for l in range(num_layers):
            for h in range(num_heads):
                if (l, h) in selected_heads_mlp:
                    xs_star.append(l)
                    ys_star.append(float(np.abs(values[l, h])))
        if xs_star:
            ax.scatter(
                xs_star,
                ys_star,
                marker="*",
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Identified Key heads with intervention on Key MLPs",
            )

    ax.set_xlabel("Layer", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)


def _plot_group(
    group_models: List[str],
    exp20_root: str,
    exp2_old_root: str,
    out_path: str,
    metric: str,
    tables_dir: str,
) -> None:
    # 风格与 plot_mlp_output_kl_layers.py 一致
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)

    for idx in range(3):
        ax = axes[idx]
        if idx >= len(group_models):
            ax.axis("off")
            continue

        model_name = group_models[idx]
        disp = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(disp, fontweight="bold")

        model_dir = os.path.join(exp20_root, model_name)
        val_path = os.path.join(model_dir, f"{metric}.npy")
        if not os.path.exists(val_path):
            ax.text(0.5, 0.5, "Missing data", ha="center", va="center")
            ax.set_xticks([])
            continue
        values = np.load(val_path)
        ax.set_xlabel("Layer", fontweight="bold")

        # baseline elbow heads from exp2_old
        sens_dir = os.path.join(exp2_old_root, f"sensitive_heads_{model_name}_top100")
        json_path = os.path.join(sens_dir, "selected_heads_elbow.json")
        if not os.path.exists(json_path):
            ax.text(0.5, 0.5, "Missing elbow heads", ha="center", va="center")
            ax.set_xticks([])
            continue
        selected = _load_selected_heads(json_path)

        # MLP elbow heads (optional)
        mlp_json_path = os.path.join(model_dir, "selected_heads_mlp_elbow.json")
        selected_mlp: Optional[Set[Tuple[int, int]]] = None
        if os.path.exists(mlp_json_path):
            selected_mlp = _load_selected_heads(mlp_json_path)

        _scatter_on_axis(ax, values, selected, selected_mlp)

        # save per-model table (once per metric plot call is fine; overwrite same content)
        os.makedirs(tables_dir, exist_ok=True)
        kl = np.load(os.path.join(model_dir, "kl_p_yes.npy"))
        md = np.load(os.path.join(model_dir, "mean_diff_p_yes.npy"))
        table_path = os.path.join(tables_dir, f"head_metrics_{model_name}.csv")
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "layer", "head", "kl_p_yes", "mean_diff_p_yes", "selected_by_elbow"])
            L, H = kl.shape
            for l in range(L):
                for h in range(H):
                    w.writerow([
                        model_name,
                        l,
                        h,
                        f"{float(kl[l, h]):.6f}",
                        f"{float(md[l, h]):.6f}",
                        1 if (l, h) in selected else 0,
                    ])

    axes[0].set_ylabel(YLABEL_MATH, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="EXP20: plot all models (Qwen/Llama) for KL and mean-diff.")
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

    models = _discover_models(exp20_root)
    if not models:
        print(f"No models found under {exp20_root} with required .npy outputs")
        return

    output_root = args.output_root or os.path.join(exp20_root, "plots_p_yes_all_models")
    os.makedirs(output_root, exist_ok=True)
    tables_dir = os.path.join(output_root, "tables")

    qwen_models = [m for m in QWEN_MODELS if m in models]
    llama_models = [m for m in LLAMA_MODELS if m in models]

    if qwen_models:
        _plot_group(
            group_models=qwen_models,
            exp20_root=exp20_root,
            exp2_old_root=exp2_old_root,
            out_path=os.path.join(output_root, "qwen_kl_p_yes.png"),
            metric="kl_p_yes",
            tables_dir=tables_dir,
        )
        _plot_group(
            group_models=qwen_models,
            exp20_root=exp20_root,
            exp2_old_root=exp2_old_root,
            out_path=os.path.join(output_root, "qwen_mean_diff_p_yes.pdf"),
            metric="mean_diff_p_yes",
            tables_dir=tables_dir,
        )

    if llama_models:
        _plot_group(
            group_models=llama_models,
            exp20_root=exp20_root,
            exp2_old_root=exp2_old_root,
            out_path=os.path.join(output_root, "llama_kl_p_yes.png"),
            metric="kl_p_yes",
            tables_dir=tables_dir,
        )
        _plot_group(
            group_models=llama_models,
            exp20_root=exp20_root,
            exp2_old_root=exp2_old_root,
            out_path=os.path.join(output_root, "llama_mean_diff_p_yes.pdf"),
            metric="mean_diff_p_yes",
            tables_dir=tables_dir,
        )

    print(f"Done. Outputs in: {output_root}")


if __name__ == "__main__":
    main()
