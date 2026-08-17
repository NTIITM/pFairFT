#!/usr/bin/env python3
"""Build the current nine-panel Meta-Llama-3-8B Figure 5."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT
DATA = SOURCE_ROOT / "data" / "current"
DERIVED = DATA / "derived"
PANELS = ROOT / "panels"
MODEL_LABEL = "Llama-3 8B"
LEGEND_FONT_SIZE = 6.3
BIAS_LEVELS = ("High", "Medium", "Low")
BIAS_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#4caf50"}
GROUPS = ("Black", "White")
CONDITIONS = ("base", "pfairft")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.2,
            "font.weight": "bold",
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_manifest() -> dict:
    path = DATA / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run prepare_figure5_data.py before plotting."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_heads() -> set[tuple[int, int]]:
    with (DATA / "heads" / "selected_heads_elbow.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    heads = {(int(row["layer"]), int(row["head"])) for row in rows}
    if not heads:
        raise ValueError("No selected heads found")
    return heads


def paired_stats(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {int(row["sample_id"]): row for row in csv.DictReader(handle)}
    gaps: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for sample_id, row in rows.items():
        matched_id = int(row["matched_id"])
        pair = tuple(sorted((sample_id, matched_id)))
        if pair in seen or matched_id not in rows:
            continue
        seen.add(pair)
        matched = rows[matched_id]
        qid = int(row["decision_question_id"])
        if int(matched["decision_question_id"]) != qid:
            continue
        gaps[qid].append(abs(float(row["p_yes"]) - float(matched["p_yes"])))
    result = {}
    for qid, values in gaps.items():
        array = np.asarray(values, dtype=np.float64)
        result[qid] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "pairs": int(array.size),
        }
    if len(result) != 70 or {row["pairs"] for row in result.values()} != {18}:
        raise ValueError(f"{path}: expected 70 QIDs with 18 pairs each")
    return result


def load_context() -> dict[int, dict[str, dict[str, float]]]:
    with (DATA / "context" / "panel_c_context_results.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(qid): {
            condition: {
                "mean": float(values["mean"]),
                "std": float(values["std"]),
                "pairs": int(values["count"]),
            }
            for condition, values in conditions.items()
        }
        for qid, conditions in payload["summary"].items()
    }


def aligned_series() -> dict[str, dict[int, dict[str, float]]]:
    names = [
        "original",
        "debiased_prompt",
        "global",
        "pfairft",
        "pfairft_kl",
        "pfairft_ce",
        "inference_time",
    ]
    data = {name: paired_stats(DATA / "downstream" / f"{name}.csv") for name in names}
    qids = set(data["original"])
    for name, values in data.items():
        if set(values) != qids:
            raise ValueError(f"QID mismatch for {name}")
    return data


def load_external_downstream(model_name: str) -> dict[str, dict[int, dict[str, float]]]:
    """Load the active Figure-5 g inputs for a cross-model comparison."""
    root = ROOT.parents[1] / "results" / model_name
    downstream = root / "downstream_evaluation"
    paths = {
        "original": downstream / "discrim_baseline_pkfair_3epoch_fresh.csv",
        "debiased_prompt": downstream / "discrim_baseline_debiased_prompt_figure5_fresh.csv",
        "global": downstream / "discrim_global_lora_raw_summary_qv_current_ranking_full_3epoch.csv",
        "pfairft": downstream / "discrim_pkfair_kl_pkfair_3epoch_fresh.csv",
        "pfairft_kl": downstream / "discrim_pkfair_pkfair_3epoch_fresh.csv",
        "inference_time": root / "inference_time_figure5" / "discrim_partial.csv",
    }
    data = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {model_name} g input: {path}")
        data[name] = paired_stats(path)
    return data


def load_geometry(domain: str) -> list[dict]:
    path = DATA / "activation_geometry" / f"{domain}_geometry.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sensitive_residual"] = float(row["sensitive_residual"])
        row["orthogonal_pc1"] = float(row["orthogonal_pc1"])
        row["orthogonal_pc2"] = float(row["orthogonal_pc2"])
    return rows


def build_scene_transitions(downstream: dict) -> dict:
    qids = sorted(downstream["original"])
    if len(qids) != 70:
        raise ValueError(f"Expected 70 Discrim-Eval scenes, found {len(qids)}")
    ordered = sorted(qids, key=lambda qid: (downstream["original"][qid]["mean"], qid))
    low_count, medium_count = 23, 23
    low_boundary = (
        downstream["original"][ordered[low_count - 1]]["mean"]
        + downstream["original"][ordered[low_count]]["mean"]
    ) / 2.0
    high_boundary = (
        downstream["original"][ordered[low_count + medium_count - 1]]["mean"]
        + downstream["original"][ordered[low_count + medium_count]]["mean"]
    ) / 2.0

    def level(value: float) -> str:
        if value <= low_boundary:
            return "Low"
        if value <= high_boundary:
            return "Medium"
        return "High"

    rows = []
    matrices = {method: defaultdict(int) for method in ("global", "pfairft")}
    for qid in qids:
        values = {
            name: float(downstream[name][qid]["mean"])
            for name in ("original", "global", "pfairft")
        }
        levels = {name: level(value) for name, value in values.items()}
        rows.append(
            {
                "decision_question_id": qid,
                "base_mean_gap": values["original"],
                "global_mean_gap": values["global"],
                "pfairft_mean_gap": values["pfairft"],
                "base_level": levels["original"],
                "global_level": levels["global"],
                "pfairft_level": levels["pfairft"],
            }
        )
        for method in matrices:
            matrices[method][(levels["original"], levels[method])] += 1

    base_counts = {
        bias_level: sum(row["base_level"] == bias_level for row in rows)
        for bias_level in BIAS_LEVELS
    }
    if base_counts != {"High": 24, "Medium": 23, "Low": 23}:
        raise ValueError(f"Unexpected Base scene bins: {base_counts}")
    for method, matrix in matrices.items():
        if sum(matrix.values()) != 70:
            raise ValueError(f"{method} transition matrix does not sum to 70")
    return {
        "rows": rows,
        "matrices": matrices,
        "thresholds": {"low_medium": low_boundary, "medium_high": high_boundary},
        "base_counts": base_counts,
    }


def write_line_derived(path: Path, data: dict, series: list[str]) -> None:
    ordered = sorted(data[series[0]], key=lambda qid: data[series[0]][qid]["mean"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "decision_question_id", "series", "mean_gap", "std_gap", "pair_count"])
        for rank, qid in enumerate(ordered):
            for name in series:
                row = data[name][qid]
                writer.writerow([rank, qid, name, row["mean"], row["std"], row["pairs"]])


def write_context_derived(path: Path, data: dict) -> None:
    ordered = sorted(data, key=lambda qid: data[qid]["Original"]["mean"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "decision_question_id", "series", "mean_gap", "std_gap", "pair_count"])
        for rank, qid in enumerate(ordered):
            for name in ("Original", "Debiased", "Context+Debiased"):
                row = data[qid][name]
                writer.writerow([rank, qid, name, row["mean"], row["std"], row["pairs"]])


def write_head_derived(
    path: Path,
    selected: set[tuple[int, int]],
    arrays: list[np.ndarray],
    labels: list[str],
) -> None:
    if not arrays or any(array.shape != (32, 32) for array in arrays):
        raise ValueError(f"Invalid head-array shape for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "head", "is_sensitive", *labels])
        for layer in range(32):
            for head in range(32):
                writer.writerow(
                    [
                        layer,
                        head,
                        int((layer, head) in selected),
                        *[float(array[layer, head]) for array in arrays],
                    ]
                )


def write_scene_derived(transitions: dict) -> None:
    rows = transitions["rows"]
    with (DERIVED / "panel_i_scene_levels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (DERIVED / "panel_i_transition_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "base_level", "method_level", "scene_count"])
        for method in ("global", "pfairft"):
            matrix = transitions["matrices"][method]
            for source in BIAS_LEVELS:
                for target in BIAS_LEVELS:
                    writer.writerow([method, source, target, matrix[(source, target)]])
    thresholds = transitions["thresholds"]
    with (DERIVED / "panel_i_thresholds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["boundary", "mean_gap"])
        writer.writerow(["low_medium", thresholds["low_medium"]])
        writer.writerow(["medium_high", thresholds["medium_high"]])


def write_geometry_derived(domain: str, rows: list[dict]) -> None:
    path = DERIVED / f"panel_h_{domain}_geometry.csv"
    fieldnames = [
        "condition",
        "sample_id",
        "matched_id",
        "pair_id",
        "group",
        "decision_question_id",
        "sensitive_residual",
        "orthogonal_pc1",
        "orthogonal_pc2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def prepare_data() -> dict:
    manifest = load_manifest()
    selected = load_selected_heads()
    downstream = aligned_series()
    context = load_context()
    head_dir = DATA / "heads"
    arrays = {
        "b_debiased": np.load(head_dir / "panel_b" / "debiased_prompt_md.npy"),
        "b_context": np.load(head_dir / "panel_b" / "debiased_prompt_with_context_md.npy"),
        "original": np.load(head_dir / "panel_e_f" / "original_md.npy"),
        "global": np.load(head_dir / "panel_e_f" / "global_md.npy"),
        "pfairft": np.load(head_dir / "panel_e_f" / "pfairft_md.npy"),
    }
    activation_metadata = load_manifest_file(DATA / "activation_geometry" / "metadata.json")
    geometries = {domain: load_geometry(domain) for domain in ("resume", "discrim")}
    transitions = build_scene_transitions(downstream)
    if any(array.shape != (32, 32) or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("All head arrays must be finite and shaped (32, 32).")

    DERIVED.mkdir(parents=True, exist_ok=True)
    write_line_derived(DERIVED / "panel_a_by_question.csv", downstream, ["original", "debiased_prompt"])
    write_head_derived(
        DERIVED / "panel_b_head_scores.csv",
        selected,
        [arrays["b_debiased"], arrays["b_context"]],
        ["debiased_prompt", "debiased_prompt_with_context"],
    )
    write_context_derived(DERIVED / "panel_c_context_performance.csv", context)
    write_line_derived(DERIVED / "panel_d_by_question.csv", downstream, ["original", "global"])
    write_head_derived(
        DERIVED / "panel_e_head_scores.csv",
        selected,
        [arrays["original"], arrays["global"]],
        ["original", "global"],
    )
    write_head_derived(
        DERIVED / "panel_f_head_scores.csv",
        selected,
        [arrays["global"], arrays["pfairft"]],
        ["global", "pfairft"],
    )
    write_line_derived(
        DERIVED / "panel_g_by_question.csv",
        downstream,
        [
            "original",
            "debiased_prompt",
            "global",
            "pfairft",
            "pfairft_kl",
            "pfairft_ce",
            "inference_time",
        ],
    )
    write_scene_derived(transitions)
    for domain, rows in geometries.items():
        write_geometry_derived(domain, rows)
    return {
        "manifest": manifest,
        "selected": selected,
        "downstream": downstream,
        "context": context,
        "arrays": arrays,
        "activation_metadata": activation_metadata,
        "geometries": geometries,
        "transitions": transitions,
    }


def style_line_axis(ax: plt.Axes, xlabel: str = MODEL_LABEL) -> None:
    ax.set_xticks([])
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=1.5)
    ax.set_ylabel(r"Fairness Violation$\downarrow$", fontweight="bold", labelpad=0.2)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.45)
    ax.tick_params(axis="y", pad=1.2, length=2.2, width=0.55)


def plot_two_lines(
    fig: plt.Figure,
    spec,
    data: dict,
    first: str,
    second: str,
    first_label: str,
    second_label: str,
    first_color: str,
    second_color: str,
    xlabel: str = MODEL_LABEL,
) -> None:
    ax = fig.add_subplot(spec)
    ordered = sorted(data[first], key=lambda qid: data[first][qid]["mean"], reverse=True)
    x = np.arange(len(ordered))
    for name, label, color in (
        (first, first_label, first_color),
        (second, second_label, second_color),
    ):
        means = np.asarray([data[name][qid]["mean"] for qid in ordered])
        stds = np.asarray([data[name][qid]["std"] for qid in ordered])
        ax.plot(x, means, color=color, label=label)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.16)
    style_line_axis(ax, xlabel=xlabel)
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )


def plot_context(fig: plt.Figure, spec, data: dict) -> None:
    ax = fig.add_subplot(spec)
    ordered = sorted(data, key=lambda qid: data[qid]["Original"]["mean"], reverse=True)
    x = np.arange(len(ordered))
    conditions = ["Original", "Debiased", "Context+Debiased"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    width = 0.25
    for index, (condition, color) in enumerate(zip(conditions, colors)):
        ax.bar(
            x + (index - 1) * width,
            [data[qid][condition]["mean"] for qid in ordered],
            width,
            label=condition,
            color=color,
            alpha=0.85,
        )
    labels = {40: "Publishing", 12: "ID Loan", 94: "Business Loan"}
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(qid, f"Q{qid}") for qid in ordered], fontsize=6.0)
    ax.set_ylabel(r"Fairness Violation$\downarrow$", fontweight="bold", labelpad=0.2)
    ax.set_ylim(0.0, 0.5)
    ax.set_title(f"{MODEL_LABEL} Context Performance", fontweight="bold", pad=2.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.45)
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=5.7,
        handlelength=1.25,
        handletextpad=0.3,
        borderpad=0.3,
        labelspacing=0.3,
    )


def split_head_points(values: np.ndarray, selected: set[tuple[int, int]]):
    other_x, other_y, key_x, key_y = [], [], [], []
    for layer in range(values.shape[0]):
        for head in range(values.shape[1]):
            target_x, target_y = (key_x, key_y) if (layer, head) in selected else (other_x, other_y)
            target_x.append(layer)
            target_y.append(float(values[layer, head]))
    return other_x, other_y, key_x, key_y


def plot_head_comparison(
    fig: plt.Figure,
    spec,
    background: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    selected: set[tuple[int, int]],
    first_label: str,
    second_label: str,
    ylabel: str,
    title: str,
    hide_xticks: bool,
    scientific_y: bool = False,
) -> None:
    ax = fig.add_subplot(spec)
    other_x, other_y, _, _ = split_head_points(background, selected)
    _, _, first_x, first_y = split_head_points(first, selected)
    _, _, second_x, second_y = split_head_points(second, selected)
    ax.scatter(
        other_x,
        other_y,
        c="white",
        edgecolors="black",
        alpha=0.55,
        s=9,
        linewidths=0.5,
        label="Other heads",
    )
    ax.scatter(
        first_x,
        first_y,
        c="red",
        edgecolors="black",
        alpha=0.9,
        s=13,
        linewidths=0.5,
        label=f"Key Heads ({first_label})",
    )
    ax.scatter(
        second_x,
        second_y,
        marker="*",
        s=34,
        facecolors="none",
        edgecolors="black",
        linewidths=0.85,
        label=f"Key Heads ({second_label})",
    )
    ax.set_title(title, fontweight="bold", pad=2.5)
    ax.set_xlabel("Layer", fontweight="bold", labelpad=1.5)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=0.2)
    if scientific_y:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_offset_position("left")
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(6.4)
        offset_text.set_fontweight("bold")
        offset_text.set_ha("left")
        offset_text.set_x(0.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    if hide_xticks:
        ax.set_xticks([])
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.45)
    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=5.7 if hide_xticks else LEGEND_FONT_SIZE,
        handletextpad=0.35,
        borderpad=0.32,
        labelspacing=0.28,
    )


def plot_panel_b(fig: plt.Figure, spec, all_data: dict) -> None:
    arrays = all_data["arrays"]
    plot_head_comparison(
        fig,
        spec,
        arrays["b_debiased"],
        arrays["b_debiased"],
        arrays["b_context"],
        all_data["selected"],
        "Debiased",
        "Context+Debiased",
        "Fairness violation",
        rf"{MODEL_LABEL}: Mean $|\Delta p(\mathrm{{yes}})|$",
        False,
        True,
    )


def plot_panel_e(fig: plt.Figure, spec, all_data: dict) -> None:
    arrays = all_data["arrays"]
    plot_head_comparison(
        fig,
        spec,
        arrays["original"],
        arrays["original"],
        arrays["global"],
        all_data["selected"],
        "Original",
        "Global",
        r"$I_{l,h}$",
        MODEL_LABEL,
        True,
        True,
    )


def plot_panel_f(fig: plt.Figure, spec, all_data: dict) -> None:
    arrays = all_data["arrays"]
    plot_head_comparison(
        fig,
        spec,
        arrays["global"],
        arrays["global"],
        arrays["pfairft"],
        all_data["selected"],
        "Global",
        "PFairFT",
        r"$I_{l,h}$",
        MODEL_LABEL,
        False,
    )


def _plot_panel_g_axis(ax: plt.Axes, data: dict, xlabel: str) -> None:
    # The CSV keys are bound directly to the workflow's public method names.
    series = [
        ("original", "Original", "tab:blue"),
        ("debiased_prompt", "Debiased Prompt", "tab:purple"),
        ("global", "Global", "tab:red"),
        ("pfairft", "PFairFT", "tab:green"),
        ("pfairft_kl", "PFairFT-KL", "tab:orange"),
        ("pfairft_ce", "PFairFT-CE", "tab:cyan"),
        ("inference_time", "Inference Time", "tab:brown"),
    ]
    ordered = sorted(data["original"], key=lambda qid: data["original"][qid]["mean"], reverse=True)
    x = np.arange(len(ordered))
    for key, label, color in series:
        ax.plot(x, [data[key][qid]["mean"] for qid in ordered], color=color, label=label)
    style_line_axis(ax, xlabel=xlabel)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        fontsize=5.5,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.65,
        borderpad=0.3,
        labelspacing=0.25,
    )


def plot_panel_g(
    fig: plt.Figure,
    spec,
    data: dict,
    xlabel: str = MODEL_LABEL,
) -> None:
    if "original" in data:
        _plot_panel_g_axis(fig.add_subplot(spec), data, xlabel)
        return
    models = list(data.items())
    if not models:
        raise ValueError("Panel g requires at least one model")
    axes = spec.subgridspec(1, len(models), wspace=0.28)
    for index, (model_label, model_data) in enumerate(models):
        _plot_panel_g_axis(fig.add_subplot(axes[0, index]), model_data, model_label)


def _add_embedding_ellipse(
    ax: plt.Axes, coordinates: np.ndarray, color: str
) -> None:
    if len(coordinates) < 3:
        return
    covariance = np.cov(coordinates, rowvar=False)
    if not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    scale = np.sqrt(3.2188758248682006)  # 80% bivariate-normal contour.
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    ax.add_patch(
        Ellipse(
            coordinates.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=0.85,
            alpha=0.85,
            zorder=3,
        )
    )


def plot_panel_h(fig: plt.Figure, spec, all_data: dict, domain: str = "discrim") -> None:
    rows = all_data["geometries"][domain]
    metadata = all_data["activation_metadata"]
    selected = metadata["selected_head"]
    inner = spec.subgridspec(1, 2, wspace=0.06)
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]

    # The latent plane contains d and the most informative Base direction in d-perp.
    # A fixed rotation keeps display axes unnamed while preserving all geometry.
    angle = np.deg2rad(32.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    raw_coordinates = np.asarray(
        [[row["sensitive_residual"], row["orthogonal_pc1"]] for row in rows],
        dtype=np.float64,
    )
    display_coordinates = raw_coordinates @ rotation.T
    x_limit = max(float(np.max(np.abs(display_coordinates[:, 0]))) * 1.12, 1e-6)
    y_limit = max(float(np.max(np.abs(display_coordinates[:, 1]))) * 1.12, 1e-6)
    point_size = 7 if domain == "discrim" else 14
    point_alpha = 0.22 if domain == "discrim" else 0.55
    domain_suffix = " Resume" if domain == "resume" else ""
    condition_titles = {
        "base": f"Base{domain_suffix} (L{selected['layer']}H{selected['head']})",
        "pfairft": "After PFairFT",
    }
    group_style = {
        "Black": ("#ef3b2c", "Black-associated"),
        "White": ("#2563eb", "White-associated"),
    }
    d_vector = rotation @ np.asarray([1.0, 0.0])
    fair_vector = rotation @ np.asarray([0.0, 1.0])
    for ax, condition in zip(axes, CONDITIONS):
        condition_rows = [row for row in rows if row["condition"] == condition]
        for group in GROUPS:
            group_rows = [row for row in condition_rows if row["group"] == group]
            color, label = group_style[group]
            coordinates = np.asarray(
                [
                    [row["sensitive_residual"], row["orthogonal_pc1"]]
                    for row in group_rows
                ],
                dtype=np.float64,
            ) @ rotation.T
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=point_size,
                c=color,
                alpha=point_alpha,
                linewidths=0,
                rasterized=domain == "discrim",
                label=label,
            )
            _add_embedding_ellipse(ax, coordinates, color)
            center = coordinates.mean(axis=0)
            ax.scatter(
                [center[0]],
                [center[1]],
                marker="+",
                s=38,
                c=color,
                linewidths=1.4,
                zorder=4,
            )
        fair_extent = min(
            x_limit / max(abs(fair_vector[0]), 1e-9),
            y_limit / max(abs(fair_vector[1]), 1e-9),
        )
        fair_line = np.asarray([-fair_extent, fair_extent])[:, None] * fair_vector
        ax.plot(
            fair_line[:, 0],
            fair_line[:, 1],
            color="#4b5563",
            linestyle="--",
            linewidth=0.85,
            alpha=0.8,
            label="Fairness boundary" if condition == "base" else None,
            zorder=2,
        )
        arrow_length = 0.50 * min(x_limit, y_limit)
        arrow_start = -0.72 * arrow_length * d_vector
        ax.annotate(
            "",
            xy=arrow_start + arrow_length * d_vector,
            xytext=arrow_start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111827",
                "lw": 1.7,
                "mutation_scale": 12,
            },
            zorder=10,
        )
        arrow_tip = arrow_start + arrow_length * d_vector
        ax.text(
            arrow_tip[0] + 0.025 * x_limit,
            arrow_tip[1] + 0.025 * y_limit,
            r"$d$",
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=11,
        )
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(condition_titles[condition], fontweight="bold", pad=2.0)
        ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.4)
        ax.tick_params(pad=1.0, length=2.0, width=0.5)
        ax.set_aspect("equal", adjustable="box")
    axes[1].set_yticklabels([])
    axes[0].legend(
        loc="upper left",
        frameon=True,
        fontsize=5.2,
        markerscale=1.7,
        handletextpad=0.25,
        borderpad=0.25,
        labelspacing=0.2,
    )


def _node_intervals(counts: dict[str, int], scale: float = 0.0095) -> dict[str, tuple[float, float]]:
    heights = {level: counts[level] * scale for level in BIAS_LEVELS}
    return {
        "High": (0.94 - heights["High"], 0.94),
        "Medium": (0.5 - heights["Medium"] / 2.0, 0.5 + heights["Medium"] / 2.0),
        "Low": (0.06, 0.06 + heights["Low"]),
    }


def _segments(
    interval: tuple[float, float], counts: dict[str, int], scale: float = 0.0095
) -> dict[str, tuple[float, float]]:
    cursor = interval[0]
    output = {}
    for level in reversed(BIAS_LEVELS):
        height = counts[level] * scale
        output[level] = (cursor, cursor + height)
        cursor += height
    if not np.isclose(cursor, interval[1], atol=1e-9):
        raise ValueError("Sankey segment allocation does not fill its node")
    return output


def _ribbon(
    ax: plt.Axes,
    x0: float,
    source: tuple[float, float],
    x1: float,
    target: tuple[float, float],
    color: str,
) -> None:
    bend = abs(x1 - x0) * 0.46
    direction = 1.0 if x1 > x0 else -1.0
    c0, c1 = x0 + direction * bend, x1 - direction * bend
    vertices = [
        (x0, source[0]),
        (c0, source[0]),
        (c1, target[0]),
        (x1, target[0]),
        (x1, target[1]),
        (c1, target[1]),
        (c0, source[1]),
        (x0, source[1]),
        (x0, source[0]),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=0.34,
            zorder=1,
        )
    )


def plot_panel_i(fig: plt.Figure, spec, all_data: dict) -> None:
    ax = fig.add_subplot(spec)
    transitions = all_data["transitions"]
    base_counts = transitions["base_counts"]
    matrices = transitions["matrices"]
    method_counts = {
        method: {
            target: sum(matrix[(source, target)] for source in BIAS_LEVELS)
            for target in BIAS_LEVELS
        }
        for method, matrix in matrices.items()
    }
    node_counts = {
        "global": method_counts["global"],
        "base": base_counts,
        "pfairft": method_counts["pfairft"],
    }
    scene_scale = 0.0105
    intervals = {
        name: _node_intervals(counts, scale=scene_scale)
        for name, counts in node_counts.items()
    }
    x_positions = {"global": 0.12, "base": 0.465, "pfairft": 0.81}
    node_width = 0.075

    for method, target_x, source_x in (
        ("global", x_positions["global"] + node_width, x_positions["base"]),
        ("pfairft", x_positions["pfairft"], x_positions["base"] + node_width),
    ):
        matrix = matrices[method]
        base_segments = {
            source: _segments(
                intervals["base"][source],
                {target: matrix[(source, target)] for target in BIAS_LEVELS},
                scale=scene_scale,
            )
            for source in BIAS_LEVELS
        }
        target_segments = {
            target: _segments(
                intervals[method][target],
                {source: matrix[(source, target)] for source in BIAS_LEVELS},
                scale=scene_scale,
            )
            for target in BIAS_LEVELS
            if node_counts[method][target] > 0
        }
        for source in BIAS_LEVELS:
            for target in BIAS_LEVELS:
                if matrix[(source, target)] == 0:
                    continue
                _ribbon(
                    ax,
                    source_x,
                    base_segments[source][target],
                    target_x,
                    target_segments[target][source],
                    BIAS_COLORS[target],
                )

    for name in ("global", "base", "pfairft"):
        x = x_positions[name]
        for level in BIAS_LEVELS:
            count = node_counts[name][level]
            if count == 0:
                continue
            bottom, top = intervals[name][level]
            ax.add_patch(
                Rectangle(
                    (x, bottom),
                    node_width,
                    top - bottom,
                    facecolor=BIAS_COLORS[level],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
            )
            text_x = x - 0.012 if name == "global" else x + node_width + 0.012
            ax.text(
                text_x,
                (bottom + top) / 2.0,
                f"{level} {count}",
                ha="right" if name == "global" else "left",
                va="center",
                color=BIAS_COLORS[level],
                fontsize=6.5,
                fontweight="bold",
                zorder=4,
            )
    for name, label in (("global", "Global"), ("base", "Base"), ("pfairft", "PFairFT")):
        ax.text(
            x_positions[name] + node_width / 2.0,
            1.005,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )
    ax.text(
        0.32,
        1.005,
        r"$\leftarrow$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
    )
    ax.text(
        0.70,
        1.005,
        r"$\rightarrow$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def add_panel_label(fig: plt.Figure, spec, letter: str, x_offset: float) -> None:
    box = spec.get_position(fig)
    fig.text(box.x0 - x_offset, box.y1 + 0.012, letter, fontsize=11, fontweight="bold", va="bottom")


def save_plot_pair(output_stem: Path, width_cm: float, plotter, letter: str) -> None:
    fig = plt.figure(figsize=(width_cm / 2.54, 4.8 / 2.54), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.14, right=0.985, top=0.87, bottom=0.17)
    plotter(fig, grid[0])
    fig.text(0.018, 0.985, letter, fontsize=11, fontweight="bold", va="top")
    fig.savefig(Path(f"{output_stem}.pdf"), dpi=600)
    fig.savefig(Path(f"{output_stem}.png"), dpi=400)
    plt.close(fig)


def save_single_panels(all_data: dict) -> None:
    downstream = all_data["downstream"]
    panel_specs = [
        (
            "a",
            10.5,
            lambda fig, spec: plot_two_lines(
                fig,
                spec,
                downstream,
                "original",
                "debiased_prompt",
                "Original Prompt",
                "Debiased Prompt",
                "tab:blue",
                "tab:orange",
            ),
        ),
        ("b", 10.5, lambda fig, spec: plot_panel_b(fig, spec, all_data)),
        ("c", 7.0, lambda fig, spec: plot_context(fig, spec, all_data["context"])),
        (
            "d",
            7.0,
            lambda fig, spec: plot_two_lines(
                fig,
                spec,
                downstream,
                "original",
                "global",
                "Original",
                "Global",
                "tab:blue",
                "tab:red",
            ),
        ),
        ("e", 7.0, lambda fig, spec: plot_panel_e(fig, spec, all_data)),
        (
            "f",
            21.0,
            lambda fig, spec: plot_panel_g(
                fig,
                spec,
                {"Llama-3 8B": downstream},
            ),
        ),
        ("g", 7.0, lambda fig, spec: plot_panel_i(fig, spec, all_data)),
        ("h", 10.5, lambda fig, spec: plot_panel_f(fig, spec, all_data)),
        ("i", 10.5, lambda fig, spec: plot_panel_h(fig, spec, all_data, "resume")),
    ]
    for letter, width_cm, plotter in panel_specs:
        save_plot_pair(PANELS / f"panel_{letter}", width_cm, plotter, letter)
    save_plot_pair(
        PANELS / "panel_h_resume",
        10.5,
        lambda fig, spec: plot_panel_h(fig, spec, all_data, "resume"),
        "h",
    )
    save_plot_pair(
        PANELS / "panel_h_discrim",
        10.5,
        lambda fig, spec: plot_panel_h(fig, spec, all_data, "discrim"),
        "h",
    )


def save_combined(all_data: dict) -> None:
    downstream = all_data["downstream"]
    fig = plt.figure(figsize=(21.0 / 2.54, 19.4 / 2.54), facecolor="white")
    outer = fig.add_gridspec(
        4,
        6,
        left=0.062,
        right=0.992,
        top=0.94,
        bottom=0.07,
        hspace=0.44,
        wspace=0.42,
        height_ratios=(1.0, 1.0, 1.0, 1.16),
    )
    # The final two rows follow the manuscript's reading order: comparison,
    # scene transitions, the head-level result, then activation geometry.
    specs = {
        "a": outer[0, 0:3],
        "b": outer[0, 3:6],
        "c": outer[1, 0:2],
        "d": outer[1, 2:4],
        "e": outer[1, 4:6],
        "f": outer[2, 0:4],
        "g": outer[2, 4:6],
        "h": outer[3, 0:2],
        "i": outer[3, 2:6],
    }
    plot_two_lines(
        fig,
        specs["a"],
        downstream,
        "original",
        "debiased_prompt",
        "Original Prompt",
        "Debiased Prompt",
        "tab:blue",
        "tab:orange",
    )
    plot_panel_b(fig, specs["b"], all_data)
    plot_context(fig, specs["c"], all_data["context"])
    plot_two_lines(
        fig,
        specs["d"],
        downstream,
        "original",
        "global",
        "Original",
        "Global",
        "tab:blue",
        "tab:red",
    )
    plot_panel_e(fig, specs["e"], all_data)
    plot_panel_g(
        fig,
        specs["f"],
        {"Llama-3 8B": downstream},
    )
    plot_panel_i(fig, specs["g"], all_data)
    plot_panel_f(fig, specs["h"], all_data)
    plot_panel_h(fig, specs["i"], all_data, "resume")
    for letter in "abcdefghi":
        add_panel_label(fig, specs[letter], letter, 0.04 if letter not in {"a", "c"} else 0.045)
    fig.savefig(ROOT / "figure5.pdf", dpi=600)
    fig.savefig(ROOT / "figure5.png", dpi=400)
    plt.close(fig)


def main() -> None:
    global DATA, DERIVED, PANELS, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Render only the current Llama 3 8B snapshot.",
    )
    args = parser.parse_args()
    DATA = args.data_dir.expanduser().resolve()
    ROOT = args.output_dir.expanduser().resolve()
    DERIVED = DATA / "derived"
    PANELS = ROOT / "panels"
    configure_style()
    PANELS.mkdir(parents=True, exist_ok=True)
    all_data = prepare_data()
    save_single_panels(all_data)
    save_combined(all_data)
    print(f"Saved {ROOT / 'figure5.pdf'}")
    print(f"Saved {ROOT / 'figure5.png'}")


if __name__ == "__main__":
    main()
