#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本：读取 exp_heads.sh 与 exp_MLPs.sh 产生的 KL 结果，
分别对 Llama 模型组与 Qwen 模型组绘制横向 3×1 的组合热力图：
- 横向排列：每一列对应一个模型（最多3个模型）
- 每个模型内部：左侧为 attention heads 的 KL 热力图，中间为 MLP 层 KL 热力图，右侧为独立的颜色条
- 每个模型使用自己的颜色范围，颜色样式参考 fairness_llm/data/compas/visualize_heads_mlp_kl.py

使用方式（示例，在 Python 中调用）：

    from exp2.plot_heads_mlp_kl import plot_all_heads_mlp_kl
    plot_all_heads_mlp_kl()

将在 exp2 目录下生成：
- llama_heads_mlp_kl.pdf
- qwen_heads_mlp_kl.pdf
"""

import os
import pickle
import sys
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

# Add parent directory to path to import util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from util import get_model_display_name  # type: ignore


# 设置全局字体为 Times New Roman（参考原 visualize_heads_mlp_kl）
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


def _create_white_to_red_cmap() -> LinearSegmentedColormap:
    """创建从米色到红色的渐变颜色映射"""
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


def _load_heads_heatmap(results_pkl_path: str) -> np.ndarray:
    """从 heads 分析结果中加载 KL 热力图（形状 [num_layers, num_heads]）"""
    with open(results_pkl_path, "rb") as f:
        data = pickle.load(f)
    if "heatmap" not in data:
        raise ValueError(f"Key 'heatmap' not found in {results_pkl_path}")
    return np.asarray(data["heatmap"], dtype=np.float64)


def _load_mlp_scores(results_pkl_path: str) -> np.ndarray:
    """从 MLP 分析结果中加载每层 KL 分数（形状 [num_layers]）"""
    with open(results_pkl_path, "rb") as f:
        data = pickle.load(f)
    # 对应 analyze_race_sensitive_MLPs.py 中的 results_mlp.pkl
    if "layer_kl_scores" in data:
        scores = data["layer_kl_scores"]
    elif "layer_scores" in data:
        # 兼容老命名
        scores = data["layer_scores"]
    else:
        raise ValueError(f"Key 'layer_kl_scores' or 'layer_scores' not found in {results_pkl_path}")
    return np.asarray(scores, dtype=np.float64)


def _collect_model_results(
    exp2_dir: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    从 exp2 目录收集各模型的 heads / MLP KL 结果。

    返回：
        heads_results[group][display_name] = heads_kl (num_layers, num_heads)
        mlp_results[group][display_name]   = mlp_kl (num_layers,)
        其中 group ∈ {"llama", "qwen"}
    """
    heads_results: Dict[str, Dict[str, np.ndarray]] = {"llama": {}, "qwen": {}}
    mlp_results: Dict[str, Dict[str, np.ndarray]] = {"llama": {}, "qwen": {}}

    # 遍历所有 sensitive_heads_*_top100 目录，尝试匹配对应的 MLP 结果
    for name in os.listdir(exp2_dir):
        if not name.startswith("sensitive_heads_") or not name.endswith("_top100"):
            continue

        model_name = name[len("sensitive_heads_") : -len("_top100")]
        heads_dir = os.path.join(exp2_dir, name)
        mlp_dir = os.path.join(exp2_dir, f"sensitive_MLPs_{model_name}_top100")

        heads_pkl = os.path.join(heads_dir, "results.pkl")
        mlp_pkl = os.path.join(mlp_dir, "results_mlp.pkl")

        if not os.path.isfile(heads_pkl):
            print(f"[WARN] Heads results not found for {model_name}: {heads_pkl}")
            continue
        if not os.path.isfile(mlp_pkl):
            print(f"[WARN] MLP results not found for {model_name}: {mlp_pkl}")
            continue

        display_name = get_model_display_name(model_name)
        display_lower = display_name.lower()
        model_name_lower = model_name.lower()
        # 识别llama模型：支持两种命名方式
        # 1. Llama-3.2-* 格式
        # 2. Meta-Llama-* 格式
        if "llama" in display_lower or "llama" in model_name_lower:
            group = "llama"
        elif "qwen" in display_lower or "qwen" in model_name_lower:
            group = "qwen"
        else:
            # 其他模型暂不绘图
            continue

        try:
            heads_kl = _load_heads_heatmap(heads_pkl)
            mlp_kl = _load_mlp_scores(mlp_pkl)
        except Exception as e:
            print(f"[WARN] Failed to load results for model {model_name}: {e}")
            continue

        # 维度检查
        if heads_kl.shape[0] != mlp_kl.shape[0]:
            print(
                f"[WARN] Layer mismatch for {model_name}: "
                f"heads layers={heads_kl.shape[0]}, mlp layers={mlp_kl.shape[0]}"
            )
            continue

        heads_results[group][display_name] = heads_kl
        mlp_results[group][display_name] = mlp_kl
        print(f"[INFO] Loaded results for {model_name} -> {display_name} (group: {group})")

    return heads_results, mlp_results


def _plot_group_heads_mlp(
    group_name: str,
    models: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    为某一模型组（Llama 或 Qwen）绘制横向 3×1 组合热力图（3列1行）。

    Args:
        group_name: "Llama" 或 "Qwen"
        models: [(display_name, heads_kl, mlp_kl), ...]，顺序即为从左到右的列顺序
        output_path: 输出图片路径
    """
    if not models:
        print(f"[INFO] No models found for group {group_name}, skip plotting.")
        return

    # 为了统一风格，这里只取最多 3 个模型（若多于 3 个，按 display_name 排序后取前 3 个）
    models = sorted(models, key=lambda x: x[0])[:3]
    n_models = len(models)

    cmap = _create_white_to_red_cmap()

    # 计算每个模型的最大head数量，用于设置图像大小
    max_heads = max(heads_kl.shape[1] for _, heads_kl, _ in models)
    max_layers = max(heads_kl.shape[0] for _, heads_kl, _ in models)

    # 自动设置图像大小：横向布局，宽度缩小（横轴调小）
    heads_width = max(5, max_heads * 0.16)
    mlp_width = 0.4  # MLP热力图
    cbar_width = 0.28
    single_model_width = heads_width + mlp_width
    total_width = n_models * single_model_width + cbar_width + (n_models - 1) * 0.12
    height = max(4, max_layers * 0.09)  # 纵轴缩短，避免图过高

    fig = plt.figure(figsize=(total_width, height))
    # 1 行，n_models * 2 列（每个模型：heads, mlp）+ 1 列（共享colorbar）
    gs = fig.add_gridspec(
        1,
        n_models * 2 + 1,
        width_ratios=[heads_width, mlp_width] * n_models + [cbar_width],
        wspace=0.35,  # 更紧凑间距
        hspace=0.3,
    )

    # 计算全局颜色范围（用于共享颜色条）
    all_values: List[float] = []
    for _, heads_kl, mlp_kl in models:
        all_values.append(heads_kl.flatten())
        all_values.append(mlp_kl.flatten())
    all_values_flat = np.concatenate(all_values)
    valid_kl_global = all_values_flat[np.isfinite(all_values_flat)]
    vmin_global = float(np.min(valid_kl_global))
    vmax_global = float(np.max(valid_kl_global))

    for col_idx, (display_name, heads_kl, mlp_kl) in enumerate(models):
        num_layers, num_heads = heads_kl.shape
        mlp_kl_2d = mlp_kl.reshape(-1, 1)

        # 反转层索引，使 layer 0 在图底部
        heads_kl_plot = np.flipud(heads_kl)
        mlp_kl_plot = np.flipud(mlp_kl_2d)

        # 分别计算heads和mlp的颜色范围
        heads_values = heads_kl.flatten()
        heads_valid = heads_values[np.isfinite(heads_values)]
        heads_vmin = float(np.min(heads_valid))
        heads_vmax = float(np.max(heads_valid))

        mlp_values = mlp_kl.flatten()
        mlp_valid = mlp_values[np.isfinite(mlp_values)]
        mlp_vmin = float(np.min(mlp_valid))
        mlp_vmax = float(np.max(mlp_valid))

        # Layer 标签：每隔 5 层标一次
        layer_labels = [str(i) if i % 5 == 0 else "" for i in range(num_layers - 1, -1, -1)]
        head_labels = [str(i) if i % 5 == 0 else "" for i in range(num_heads)]

        # 第一个子图：heads
        ax_heads = fig.add_subplot(gs[0, col_idx * 2 + 0])
        # 设置背景色为米色
        ax_heads.set_facecolor("#F5F5DC")  # 米色 (beige)
        sns.heatmap(
            heads_kl_plot,
            cmap=cmap,
            vmin=heads_vmin,
            vmax=heads_vmax,
            yticklabels=layer_labels,
            xticklabels=head_labels,
            cbar=False,
            ax=ax_heads,
            linewidths=0,
            linecolor="gray",
        )
        # 删除子图灰框
        for spine in ax_heads.spines.values():
            spine.set_visible(False)
        ax_heads.set_xlabel("Head", fontsize=32, fontfamily="serif", labelpad=6)
        if col_idx == 0:
            ax_heads.set_ylabel("Layer", fontsize=32, fontfamily="serif")
        else:
            ax_heads.set_ylabel("", fontsize=32, fontfamily="serif")
        # 将模型名作为heads子图的标题
        ax_heads.set_title(display_name, fontsize=34, fontweight="bold", fontfamily="serif")
        # 纵轴刻度标签用 pad 左移，与热力图内容分开（不靠调大 left 压窄整图）
        ax_heads.tick_params(labelsize=28, axis="y")  # , pad=12
        ax_heads.tick_params(labelsize=28, axis="x")
        for label in ax_heads.get_xticklabels() + ax_heads.get_yticklabels():
            label.set_fontfamily("serif")
            label.set_fontsize(28)

        # 第二个子图：MLP（与 Head 对齐：用 xlabel "MLP"，去掉 xticklabels）
        ax_mlp = fig.add_subplot(gs[0, col_idx * 2 + 1])
        # 设置背景色为米色
        ax_mlp.set_facecolor("#F5F5DC")  # 米色 (beige)
        sns.heatmap(
            mlp_kl_plot,
            cmap=cmap,
            vmin=mlp_vmin,
            vmax=mlp_vmax,
            yticklabels=layer_labels,
            xticklabels=False,  # 不用 tick 显示 MLP，改用 xlabel 与 Head 对齐
            cbar=False,
            ax=ax_mlp,
            linewidths=0,
            linecolor="gray",
        )
        # 删除子图灰框
        for spine in ax_mlp.spines.values():
            spine.set_visible(False)
        # MLP 无横轴刻度，xlabel 需下移才能与 Head 的 xlabel 对齐（Head 下方还有刻度标签）
        ax_mlp.set_xlabel("MLP", fontsize=32, fontfamily="serif", labelpad=36)
        ax_mlp.set_ylabel("", fontsize=32, fontfamily="serif")
        ax_mlp.tick_params(labelsize=28, axis="y") # , pad=12
        ax_mlp.tick_params(labelsize=28, axis="x")
        for label in ax_mlp.get_xticklabels() + ax_mlp.get_yticklabels():
            label.set_fontfamily("serif")
            label.set_fontsize(28)

    # 只在最右边添加共享颜色条
    ax_cbar = fig.add_subplot(gs[0, n_models * 2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation="vertical")
    cbar.set_label("KL Divergence", fontsize=32, rotation=270, labelpad=36, fontfamily="serif")
    # 删除颜色条刻度
    cbar.ax.tick_params(labelsize=0, length=0)  # 隐藏刻度
    cbar.ax.set_yticks([])  # 删除所有刻度
    for label in cbar.ax.get_yticklabels():
        label.set_visible(False)
    cbar.outline.set_visible(False)
    for spine in ["top", "bottom", "left", "right"]:
        cbar.ax.spines[spine].set_visible(False)

    # 左侧留足空间，避免纵轴刻度和 Layer 与热力图重叠
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.08, wspace=0.15)
    # 保存为PDF格式
    if output_path.endswith('.png'):
        output_path = output_path[:-4] + '.pdf'
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", format='pdf')
    plt.close()

    print(f"[INFO] Saved {group_name} combined heads+MLP KL heatmap to: {output_path}")


def plot_all_heads_mlp_kl() -> None:
    """
    对 Llama 与 Qwen 模型组分别绘制 3×1 组合热力图：
    - 输出文件：
        - llama_heads_mlp_kl.pdf
        - qwen_heads_mlp_kl.pdf
    """
    exp2_dir = os.path.dirname(os.path.abspath(__file__))
    heads_results, mlp_results = _collect_model_results(exp2_dir)

    # 组装 (display_name, heads_kl, mlp_kl) 列表
    llama_models: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for disp_name, heads_kl in heads_results["llama"].items():
        if disp_name in mlp_results["llama"]:
            llama_models.append((disp_name, heads_kl, mlp_results["llama"][disp_name]))

    qwen_models: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for disp_name, heads_kl in heads_results["qwen"].items():
        if disp_name in mlp_results["qwen"]:
            qwen_models.append((disp_name, heads_kl, mlp_results["qwen"][disp_name]))

    # Llama 组
    llama_out = os.path.join(exp2_dir, "llama_heads_mlp_kl.pdf")
    _plot_group_heads_mlp("Llama", llama_models, llama_out)

    # Qwen 组
    qwen_out = os.path.join(exp2_dir, "qwen_heads_mlp_kl.pdf")
    _plot_group_heads_mlp("Qwen", qwen_models, qwen_out)


if __name__ == "__main__":
    plot_all_heads_mlp_kl()
