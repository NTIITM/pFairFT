#!/usr/bin/env python3
"""Render the single-model Llama 3 8B Figure 1-4 experiment outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "base": "#7A7A7A",
    "random": "#E69F00",
    "mlp": "#009E73",
    "head": "#0072B2",
    "secondary": "#D55E00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a dictionary")
    return value


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, output: Path, sources: list[Path], metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        **metadata,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sources
        ],
        "outputs": [str(output.with_suffix(suffix).resolve()) for suffix in (".pdf", ".png")],
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.07, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")


def render_figure1(root: Path, output: Path, model_name: str) -> None:
    heads_path = root / "figure1/components/heads/results.pkl"
    mlp_path = root / "figure1/components/mlps/identification/results_mlp.pkl"
    selected_heads_path = root / "figure1/components/heads/selected_heads_elbow.json"
    selected_mlp_path = root / "figure1/components/mlps/selected/selected_mlp_layers_elbow.json"
    heads_data, mlp_data = load_pickle(heads_path), load_pickle(mlp_path)
    heads = np.asarray(heads_data["heatmap"], dtype=float)
    mlps = np.asarray(mlp_data["layer_kl_scores"], dtype=float)
    selected_heads = load_json(selected_heads_path)
    selected_mlps = load_json(selected_mlp_path)
    if heads.shape != (32, 32) or mlps.shape != (32,):
        raise ValueError(f"unexpected Llama 3 8B component shapes: {heads.shape}, {mlps.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.7), constrained_layout=True)
    image = axes[0].imshow(heads, cmap="Reds", aspect="auto")
    axes[0].set(xlabel="Head", ylabel="Layer", title="Head KL Intensity")
    fig.colorbar(image, ax=axes[0], fraction=0.05, pad=0.03)
    axes[1].plot(np.arange(32), mlps, color=COLORS["mlp"], marker="o", markersize=3)
    for row in selected_mlps:
        axes[1].scatter(int(row["layer"]), mlps[int(row["layer"])], color=COLORS["secondary"], zorder=3)
    axes[1].set(xlabel="Layer", ylabel="KL divergence", title="MLP KL Intensity")
    ranked = np.sort(heads[np.isfinite(heads)].reshape(-1))[::-1]
    elbow_rank = int(heads_data.get("elbow_rank", int(heads_data["elbow_idx"]) + 1))
    axes[2].plot(np.arange(1, ranked.size + 1), ranked, color=COLORS["head"])
    axes[2].scatter(elbow_rank, ranked[elbow_rank - 1], marker="*", s=90, color=COLORS["secondary"], zorder=3)
    axes[2].axvline(elbow_rank, color=COLORS["secondary"], linestyle="--", linewidth=0.8)
    axes[2].set(xlabel="Head rank", ylabel="KL divergence", title=f"Elbow Selection (K={len(selected_heads)})")
    for letter, ax in zip("abc", axes):
        panel_label(ax, letter)
    save(
        fig,
        output / "figure1",
        [heads_path, mlp_path, selected_heads_path, selected_mlp_path],
        {"figure": 1, "model": model_name, "selected_head_count": len(selected_heads)},
    )


def load_resume_curve(path: Path) -> tuple[dict[int, float], dict[int, list[dict[str, str]]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows(path):
        grouped[int(row["head_count"])].append(row)
    means = {
        count: float(np.mean([abs(float(row["fact_p_yes"]) - float(row["cf_p_yes"])) for row in rows]))
        for count, rows in grouped.items()
    }
    return dict(sorted(means.items())), grouped


def mean_resume_gap(path: Path) -> float:
    rows = csv_rows(path)
    return float(np.mean([abs(float(row["fact_p_yes"]) - float(row["cf_p_yes"])) for row in rows]))


def render_figure2(root: Path, output: Path, model_name: str) -> None:
    sensitive_path = root / "figure2/head_resume/sensitive/intervention_results_by_head_count.csv"
    random_path = root / "figure2/head_resume/random_seed_42/intervention_results_by_head_count_random.csv"
    mlp_path = root / "figure2/mlp_resume/per_sample.csv"
    sensitive, sensitive_rows = load_resume_curve(sensitive_path)
    random, random_rows = load_resume_curve(random_path)
    final_count = max(sensitive)
    if sorted(sensitive) != sorted(random) or final_count not in random:
        raise ValueError("sensitive/random Resume head-count grids differ")
    base = sensitive[0]
    mlp = mean_resume_gap(mlp_path)

    fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.75), constrained_layout=True)
    labels = ["Base", "Random Heads", "Key MLPs", "Key Heads"]
    values = [base, random[final_count], mlp, sensitive[final_count]]
    axes[0].bar(labels, values, color=[COLORS[k] for k in ("base", "random", "mlp", "head")], edgecolor="black", linewidth=0.4)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="Fairness Violation", title="Component Intervention")

    conditions = [sensitive_rows[0], random_rows[final_count], sensitive_rows[final_count]]
    count = min(map(len, conditions))
    x = np.arange(3)
    for index in range(count):
        y = [abs(float(rows[index]["fact_p_yes"]) - float(rows[index]["cf_p_yes"])) for rows in conditions]
        axes[1].plot(x, y, color="#8A8A8A", alpha=0.2, linewidth=0.45)
    axes[1].set_xticks(x, ["Base", "Random", "Key"])
    axes[1].set(ylabel="Per-sample Violation", title="Paired Resume Samples")

    axes[2].plot(sensitive, list(sensitive.values()), marker="o", color=COLORS["head"], label="Key Heads")
    axes[2].plot(random, list(random.values()), marker="s", color=COLORS["random"], label="Random Heads")
    axes[2].set(xlabel="Intervened Heads", ylabel="Fairness Violation", title="Head-count Sweep")
    axes[2].legend(frameon=False)
    for letter, ax in zip("abc", axes):
        panel_label(ax, letter)
    save(
        fig,
        output / "figure2",
        [sensitive_path, random_path, mlp_path],
        {"figure": 2, "model": model_name, "head_counts": sorted(sensitive), "seed": 42},
    )


def load_heads(path: Path) -> set[tuple[int, int]]:
    return {(int(row["layer"]), int(row["head"])) for row in load_json(path)}


def render_figure3(root: Path, output: Path, model_name: str) -> None:
    exp20 = root / "figure3/exp20"
    head_root = root / "figure3/head_logit"
    after_root = root / "figure3/head_logit_after_mlp"
    attention_path = root / "figure3/attention/fixed/qk_scores_full.json"
    selected_path = root / "figure1/components/heads/selected_heads_elbow.json"
    mlp_selected_path = root / "figure1/components/mlps/selected/selected_mlp_layers_elbow.json"
    input_path = exp20 / "mlp_input_mean_abs_diff_p_race.npy"
    original_path = exp20 / "mlp_mean_diff_p_yes.npy"
    intervened_path = exp20 / "mlp_mean_diff_p_yes_intervened.npy"
    head_path = head_root / "mean_diff_p_yes.npy"
    after_path = after_root / "mean_diff_p_yes_mlp.npy"
    mlp_input = np.load(input_path)
    original, intervened = np.load(original_path), np.load(intervened_path)
    head_values, after_values = np.load(head_path), np.load(after_path)
    selected = load_heads(selected_path)
    attention = load_json(attention_path)

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.2), constrained_layout=True)
    axes[0, 0].plot(mlp_input, marker="o", markersize=3, color=COLORS["head"])
    axes[0, 0].set(xlabel="Layer", ylabel=r"$\Delta_S$", title="MLP-input Race Difference")
    axes[0, 1].plot(original, marker="o", markersize=3, label="Original", color=COLORS["head"])
    axes[0, 1].plot(intervened, marker="s", markersize=3, linestyle="--", label="Key-head Intervention", color=COLORS["random"])
    axes[0, 1].set(xlabel="Layer", ylabel="Discriminatory Intensity", title="MLP-output Intensity")
    axes[0, 1].legend(frameon=False)
    image = axes[1, 0].imshow(np.abs(head_values), cmap="Blues", aspect="auto")
    for layer, head in selected:
        axes[1, 0].scatter(head, layer, facecolors="none", edgecolors=COLORS["secondary"], s=18, linewidths=0.7)
    axes[1, 0].set(xlabel="Head", ylabel="Layer", title="Head-level Intensity")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.04, pad=0.03)

    scores = attention["qk_scores"]
    best_key = max(scores, key=lambda key: max(scores[key]))
    values = np.asarray(scores[best_key], dtype=float)
    tokens = str(attention.get("formatted_prompt", "")).split()
    if len(tokens) != len(values):
        tokens = [str(index) for index in range(len(values))]
    axes[1, 1].barh(np.arange(len(values)), values, color=[COLORS["secondary"] if "black" in token.lower() else "#A6A6A6" for token in tokens])
    axes[1, 1].set_yticks(np.arange(len(values)), tokens, fontsize=6)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(xlabel="Attention", title=f"Token Attention ({best_key.replace('_', ' ')})")
    for letter, ax in zip("abcd", axes.flat):
        panel_label(ax, letter)
    save(
        fig,
        output / "figure3",
        [input_path, original_path, intervened_path, head_path, after_path, attention_path, selected_path, mlp_selected_path],
        {
            "figure": 3,
            "model": model_name,
            "probe_surface": "next_mlp_input_cumulative_residual",
            "attention_head": best_key,
            "after_mlp_shape": list(after_values.shape),
        },
    )


def paired_by_qid(path: Path) -> dict[int, float]:
    rows = {int(row["sample_id"]): row for row in csv_rows(path)}
    values: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for sample_id, row in rows.items():
        matched_id = int(row["matched_id"])
        pair = tuple(sorted((sample_id, matched_id)))
        if pair in seen or matched_id not in rows:
            continue
        seen.add(pair)
        values[int(row["decision_question_id"])].append(
            abs(float(row["p_yes"]) - float(rows[matched_id]["p_yes"]))
        )
    return {qid: float(np.mean(gaps)) for qid, gaps in values.items()}


def head_count_by_qid(path: Path) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in csv_rows(path):
        grouped[int(row["head_count"])].append(float(row["mean_p_yes_gap"]))
    return {count: float(np.mean(values)) for count, values in sorted(grouped.items())}


def summary_values(path: Path) -> dict[str, float]:
    mapping = {row["condition"]: float(row["fairness_violation"]) for row in csv_rows(path)}
    required = {"base", "random_heads", "key_mlps", "key_heads"}
    if not required <= mapping.keys():
        raise ValueError(f"{path}: missing conditions {sorted(required - mapping.keys())}")
    return mapping


def render_figure4(root: Path, output: Path, model_name: str) -> None:
    baseline_path = root / "figure4/discrim/baseline.csv"
    key_path = root / "figure4/discrim/head_all/sensitive/per_sample.csv"
    random_path = root / "figure4/discrim/head_all/random_seed_42/per_sample.csv"
    key_count_path = root / "figure4/discrim/head_count/sensitive/results.csv"
    random_count_path = root / "figure4/discrim/head_count/random_seed_42/results.csv"
    compas_path = root / "figure4/compas/high_gap/evaluation/summary.csv"
    adult_path = root / "figure4/adult/high_gap/evaluation/summary.csv"
    base, key, random = paired_by_qid(baseline_path), paired_by_qid(key_path), paired_by_qid(random_path)
    key_counts, random_counts = head_count_by_qid(key_count_path), head_count_by_qid(random_count_path)
    compas, adult = summary_values(compas_path), summary_values(adult_path)

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.8), constrained_layout=True)
    ordered = sorted(base, key=base.get, reverse=True)
    x = np.arange(len(ordered))
    axes[0, 0].plot(x, [base[q] for q in ordered], label="Base", color=COLORS["base"])
    axes[0, 0].plot(x, [random[q] for q in ordered], label="Random Heads", color=COLORS["random"])
    axes[0, 0].plot(x, [key[q] for q in ordered], label="Key Heads", color=COLORS["head"])
    axes[0, 0].set(xlabel="Decision-question rank", ylabel="Fairness Violation", title="Discrim-Eval Intervention")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(key_counts, list(key_counts.values()), marker="o", label="Key Heads", color=COLORS["head"])
    axes[0, 1].plot(random_counts, list(random_counts.values()), marker="s", label="Random Heads", color=COLORS["random"])
    axes[0, 1].set(xlabel="Intervened Heads", ylabel="Fairness Violation", title="Discrim-Eval Head-count Sweep")
    axes[0, 1].legend(frameon=False)
    fields = ["base", "random_heads", "key_mlps", "key_heads"]
    labels = ["Base", "Random Heads", "Key MLPs", "Key Heads"]
    colors = [COLORS[k] for k in ("base", "random", "mlp", "head")]
    for ax, values, title in ((axes[1, 0], compas, "COMPAS High-gap Top-100"), (axes[1, 1], adult, "Adult Race High-gap Top-100")):
        ax.bar(labels, [values[field] for field in fields], color=colors, edgecolor="black", linewidth=0.4)
        ax.tick_params(axis="x", rotation=20)
        ax.set(ylabel="Fairness Violation", title=title)
    for letter, ax in zip("abcd", axes.flat):
        panel_label(ax, letter)
    save(
        fig,
        output / "figure4",
        [baseline_path, key_path, random_path, key_count_path, random_count_path, compas_path, adult_path],
        {"figure": 4, "model": model_name, "seed": 42, "discrim_qids": len(base)},
    )


RENDERERS: dict[int, Callable[[Path, Path, str], None]] = {
    1: render_figure1,
    2: render_figure2,
    3: render_figure3,
    4: render_figure4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, choices=sorted(RENDERERS), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model-name", default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    RENDERERS[args.figure](args.result_root.resolve(), args.output_dir.resolve(), args.model_name)


if __name__ == "__main__":
    main()
