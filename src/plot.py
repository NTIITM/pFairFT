#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘图工具函数模块（论文级样式）
包含热力图、肘点图等可视化函数
风格与 plot_heads_mlp_kl.py 一致：beige-to-red colormap, Times New Roman 字体
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from typing import Tuple

# 导入 util 中的函数
from util import compute_elbow_point, compute_rank_array

# ──────────────────────────────────────────────────────────────────────────────
# 全局字体设置：Times New Roman（与 plot_heads_mlp_kl.py 一致）
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


def _create_white_to_red_cmap() -> LinearSegmentedColormap:
    """创建从米色到红色的渐变颜色映射（与 plot_heads_mlp_kl.py 一致）"""
    colors = [
        "#f5f3f4",  # 米色 (beige)
        "#FFE5E5",
        "#FFCCCC",
        "#FFB2B2",
        "#FF9999",
        "#FF8080",
        "#FF6666",
        "#FF4D4D",
        "#FF3333",
        "#FF1A1A",
        "#FF0000",  # 红色
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("beige_to_red", colors, N=n_bins)
    return cmap


# 预创建 colormap
_BEIGE_TO_RED_CMAP = _create_white_to_red_cmap()


def plot_kl_heatmap(
    heatmap_kl: np.ndarray,
    output_path: str,
    title: str = "KL Divergence Heatmap",
    num_layers: int = None,
    num_heads: int = None,
) -> None:
    """
    绘制 KL 散度热力图（论文级样式）。

    Args:
        heatmap_kl: KL 散度矩阵，形状为 (num_layers, num_heads)
        output_path: 输出文件路径
        title: 图表标题
        num_layers: 层数（如果为 None，则从 heatmap_kl.shape[0] 推断）
        num_heads: 头数（如果为 None，则从 heatmap_kl.shape[1] 推断）
    """
    if num_layers is None:
        num_layers = heatmap_kl.shape[0]
    if num_heads is None:
        num_heads = heatmap_kl.shape[1]

    # 自动计算图像尺寸
    fig_width = max(6, num_heads * 0.35)
    fig_height = max(4, num_layers * 0.18)

    heatmap_flipped = np.flipud(heatmap_kl)

    # 颜色范围
    valid_kl = heatmap_kl.flatten()
    valid_kl = valid_kl[np.isfinite(valid_kl)]
    vmin = float(np.min(valid_kl))
    vmax = float(np.max(valid_kl))

    # Layer 标签：每隔 5 层标一次
    layer_labels = [str(i) if i % 5 == 0 else "" for i in range(num_layers - 1, -1, -1)]
    head_labels = [str(i) if i % 5 == 0 else "" for i in range(num_heads)]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("#F5F5DC")  # 米色背景

    sns.heatmap(
        heatmap_flipped,
        cmap=_BEIGE_TO_RED_CMAP,
        vmin=vmin,
        vmax=vmax,
        yticklabels=layer_labels,
        xticklabels=head_labels,
        cbar=True,
        cbar_kws={"label": "KL Divergence", "shrink": 0.8},
        ax=ax,
        linewidths=0,
        linecolor="gray",
    )

    # 删除子图灰框
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_ylabel("Layer", fontsize=18, fontfamily="serif")
    ax.set_xlabel("Head", fontsize=18, fontfamily="serif")
    ax.set_title(title, fontsize=20, fontweight="bold", fontfamily="serif")
    ax.tick_params(labelsize=14, axis="y")
    ax.tick_params(labelsize=14, axis="x")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")
        label.set_fontsize(14)

    # Colorbar 样式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily("serif")
    cbar.set_label("KL Divergence", fontsize=14, fontfamily="serif", rotation=270, labelpad=18)
    cbar.outline.set_visible(False)

    plt.tight_layout()

    # 同时保存 PDF 和 PNG
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    if not output_path.endswith(".pdf"):
        plt.savefig(pdf_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved KL heatmap to: {output_path}")
    if not output_path.endswith(".pdf"):
        print(f"Saved KL heatmap (PDF) to: {pdf_path}")


def plot_elbow_point_vs_rank(
    heatmap_kl: np.ndarray,
    elbow_idx: int,
    elbow_score: float,
    output_path: str,
    title: str = "Elbow Point vs Rank",
) -> None:
    """
    绘制肘点与排名的关系图（论文级样式）。

    Args:
        heatmap_kl: KL 散度矩阵
        elbow_idx: 肘点索引
        elbow_score: 肘点分数
        output_path: 输出文件路径
        title: 图表标题
    """
    # 计算每个 head 的 rank
    flat_kl_values = heatmap_kl.flatten()
    ranks = rankdata(-flat_kl_values, method='ordinal')  # 负号表示从大到小排序

    # 准备数据用于绘图
    valid_mask = np.isfinite(flat_kl_values)
    valid_kl = flat_kl_values[valid_mask]
    valid_ranks = ranks[valid_mask]

    # 计算肘点的 rank
    elbow_rank_value = int(elbow_idx) + 1
    elbow_kl_value = float(elbow_score)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(valid_ranks, valid_kl, alpha=0.6, s=30, label='All heads', color='steelblue')

    # 标注肘部点
    ax.scatter(
        elbow_rank_value,
        elbow_kl_value,
        s=200,
        marker='*',
        color='red',
        edgecolors='black',
        linewidths=1.5,
        label=f'Elbow (Rank={elbow_rank_value})',
        zorder=5,
    )

    # 添加趋势线
    if len(valid_ranks) > 1:
        z = np.polyfit(valid_ranks, valid_kl, 1)
        p = np.poly1d(z)
        rank_range = np.linspace(valid_ranks.min(), valid_ranks.max(), 100)
        ax.plot(rank_range, p(rank_range), "r--", alpha=0.8, linewidth=1.5, label='Trend')

    # 添加肘部点的水平线和垂直线
    ax.axhline(y=elbow_kl_value, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=elbow_rank_value, color='red', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel("Rank", fontsize=16, fontweight="bold", fontfamily="serif")
    ax.set_ylabel("KL Divergence", fontsize=16, fontweight="bold", fontfamily="serif")
    ax.set_title(title, fontsize=18, fontweight='bold', fontfamily="serif")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12, prop={"family": "serif"})
    ax.tick_params(labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")

    plt.tight_layout()

    # 同时保存 PDF 和 PNG
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    if not output_path.endswith(".pdf"):
        plt.savefig(pdf_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved elbow point vs rank plot to: {output_path}")
    if not output_path.endswith(".pdf"):
        print(f"Saved elbow plot (PDF) to: {pdf_path}")


def plot_rank_heatmap(
    heatmap_kl: np.ndarray,
    output_path: str,
    title: str = "Rank Heatmap",
    num_layers: int = None,
    num_heads: int = None,
) -> None:
    """
    绘制排名热力图（论文级样式）。

    Args:
        heatmap_kl: KL 散度矩阵
        output_path: 输出文件路径
        title: 图表标题
        num_layers: 层数
        num_heads: 头数
    """
    if num_layers is None:
        num_layers = heatmap_kl.shape[0]
    if num_heads is None:
        num_heads = heatmap_kl.shape[1]

    # 计算每个 head 的 rank，并翻转（与 kl heatmap 保持一致：layer 0 在底，layer N-1 在顶）
    rank_array = compute_rank_array(heatmap_kl)
    rank_array_flipped = np.flipud(rank_array)

    # 自动计算图像尺寸
    fig_width = max(6, num_heads * 0.35)
    fig_height = max(4, num_layers * 0.18)

    layer_labels = [str(i) if i % 5 == 0 else "" for i in range(num_layers - 1, -1, -1)]
    head_labels = [str(i) if i % 5 == 0 else "" for i in range(num_heads)]

    # 创建蓝色系 colormap（rank 越小 = KL 越高 = 越红）
    rank_colors = [
        "#FF0000",  # Rank 1 (highest KL) — red
        "#FF4D4D",
        "#FF9999",
        "#FFCCCC",
        "#f5f3f4",  # middle — beige
        "#E0E0E0",
        "#C0C0C0",
        "#A0A0A0",
        "#808080",  # lowest rank — gray
    ]
    rank_cmap = LinearSegmentedColormap.from_list("rank_red_beige_gray", rank_colors, N=256)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("#F5F5DC")

    sns.heatmap(
        rank_array_flipped,
        cmap=rank_cmap,
        yticklabels=layer_labels,
        xticklabels=head_labels,
        annot=False,
        fmt=".0f",
        cbar=True,
        cbar_kws={'label': 'Rank (lower = higher KL)', 'shrink': 0.8},
        ax=ax,
        linewidths=0,
    )
    # 不调用 invert_yaxis()，已通过 flipud 完成翻转

    # 删除子图灰框
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_ylabel("Layer", fontsize=18, fontfamily="serif")
    ax.set_xlabel("Head", fontsize=18, fontfamily="serif")
    ax.set_title(title, fontsize=20, fontweight='bold', fontfamily="serif")
    ax.tick_params(labelsize=14, axis="y")
    ax.tick_params(labelsize=14, axis="x")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")
        label.set_fontsize(14)

    # Colorbar 样式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily("serif")
    cbar.set_label("Rank (lower = higher KL)", fontsize=14, fontfamily="serif", rotation=270, labelpad=18)
    cbar.outline.set_visible(False)

    plt.tight_layout()

    # 同时保存 PDF 和 PNG
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    if not output_path.endswith(".pdf"):
        plt.savefig(pdf_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved rank heatmap to: {output_path}")
    if not output_path.endswith(".pdf"):
        print(f"Saved rank heatmap (PDF) to: {pdf_path}")
