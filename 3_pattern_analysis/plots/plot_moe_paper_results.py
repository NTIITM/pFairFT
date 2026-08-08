#!/usr/bin/env python
"""Plot the model-specific MOE paper analyses from the canonical results tree."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


Head = Tuple[int, int]


def _load_heads(path: Path) -> Set[Head]:
    with path.open("r", encoding="utf-8") as f:
        return {(int(row["layer"]), int(row["head"])) for row in json.load(f)}


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper-suite inputs:\n" + "\n".join(missing))


def _copy_plot(
    source: Path,
    destination_dir: Path,
    category: str,
    style_reference: str,
) -> Dict[str, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return {
        "category": category,
        "source": str(source),
        "destination": str(destination),
        "plotting_script_or_style_reference": style_reference,
        "mode": "copied_without_replotting",
    }


def _collect_existing_plots(
    root: Path,
    output_dir: Path,
    selected: Set[Head],
) -> List[Dict[str, str]]:
    """Collect canonical plot outputs without changing their original styling."""
    records: List[Dict[str, str]] = []
    style_root = "/home/common1/hwluo/project/fairness_llm_new copy"
    groups = [
        (
            "exp2_component_identification/head",
            root / "sensitive_heads_moefreeze_top100_summary_only_current_ranking",
            ("*.pdf", "*.png"),
            f"{style_root}/exp2/plot_heads_mlp_kl.py",
        ),
        (
            "exp2_component_identification/mlp",
            root / "mlp_analysis/identification_top100",
            ("*.pdf", "*.png"),
            f"{style_root}/exp2/plot_heads_mlp_kl.py",
        ),
        (
            "exp9_resume_head_count",
            root / "intervention_ablation/head_ablation_plots",
            ("sensitive_vs_random_head_count.pdf",),
            f"{style_root}/exp9/plot_intervention_all_models.py",
        ),
        (
            "exp10_discrim_head_count",
            root / "intervention_ablation/head_ablation_plots",
            (
                "discrim_sensitive_vs_random_head_count.pdf",
                "discrim_scenarios_sensitive_vs_random.pdf",
            ),
            f"{style_root}/exp10/plot_discrim_eval_head_count.py",
        ),
        (
            "exp21_debiased_prompt",
            root / "pattern_analysis/debiased_prompt_qid33",
            ("*.pdf", "*.png"),
            f"{style_root}/exp21/plot_comparison.py",
        ),
        (
            "exp23_adapter_comparison",
            root / "downstream_head_analysis/qid33_pfairft_vs_global",
            ("*.pdf", "*.png"),
            f"{style_root}/exp23/plot_exp23.py",
        ),
        (
            "exp23_adapter_comparison",
            root / "downstream_head_analysis/qid33",
            ("*.pdf", "*.png"),
            f"{style_root}/exp23/plot_exp23.py",
        ),
        (
            "figure8_downstream",
            root / "downstream_evaluation",
            ("Figure8_*.pdf", "Figure8_*.png"),
            "6_downstream_evaluation/plot_figure8.py and plot_resume_figure8.py",
        ),
    ]
    for category, source_dir, patterns, style_reference in groups:
        sources = sorted({path for pattern in patterns for path in source_dir.glob(pattern)})
        if not sources:
            # Older runs may have a qid33 baseline-comparison directory without
            # a rendered artifact; the current paper suite writes the adapter
            # comparison to qid33_pfairft_vs_global instead.
            if category == "exp23_adapter_comparison" and source_dir.name == "qid33":
                continue
            raise FileNotFoundError(f"No plot outputs found for {category} in {source_dir}")
        for source in sources:
            records.append(
                _copy_plot(
                    source,
                    output_dir / category,
                    category,
                    style_reference,
                )
            )

    expected_names = {
        f"importance_plot_L{layer}_H{head}.pdf" for layer, head in selected
    }
    for variant in ("fixed", "resume_rank1"):
        category = f"exp11_attention_importance/{variant}"
        source_dir = root / f"pattern_analysis/attention_pattern/{variant}/plots"
        sources = sorted(source_dir.glob("L*/importance_plot_L*_H*.pdf"))
        actual_names = {path.name for path in sources}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            if missing:
                raise RuntimeError(
                    f"Incomplete {category}: expected {len(expected_names)} selected-head plots, "
                    f"found {len(actual_names)}; missing={missing}, extra={extra}"
                )
            # Ignore stale plots from an older selected-head set.  The current
            # paper_plots collection must contain exactly the current heads.
            sources = [path for path in sources if path.name in expected_names]
        for source in sources:
            records.append(
                _copy_plot(
                    source,
                    output_dir / category,
                    category,
                    "3_pattern_analysis/head_attention_pattern/visualize_qk_scores.py",
                )
            )
    return records


def _generated_record(
    path: Path,
    category: str,
    style_reference: str,
) -> Dict[str, str]:
    return {
        "category": category,
        "source": "generated_from_manifest_input_files",
        "destination": str(path),
        "plotting_script_or_style_reference": style_reference,
        "mode": "generated",
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "font.weight": "bold",
            "mathtext.fontset": "cm",
            "axes.grid": False,
        }
    )


def _plot_head_metric(
    ax: plt.Axes,
    values: np.ndarray,
    selected: Set[Head],
    selected_after_mlp: Set[Head],
) -> None:
    if values.ndim != 2:
        raise ValueError(f"Expected [layers, heads], got {values.shape}")
    other_x: List[int] = []
    other_y: List[float] = []
    selected_x: List[int] = []
    selected_y: List[float] = []
    for layer in range(values.shape[0]):
        for head in range(values.shape[1]):
            target_x = selected_x if (layer, head) in selected else other_x
            target_y = selected_y if (layer, head) in selected else other_y
            target_x.append(layer)
            target_y.append(abs(float(values[layer, head])))
    ax.scatter(
        other_x,
        other_y,
        c="white",
        edgecolors="black",
        alpha=0.6,
        label="Other heads",
    )
    ax.scatter(
        selected_x,
        selected_y,
        c="red",
        edgecolors="black",
        alpha=0.9,
        label="Key Heads",
    )
    star_x = []
    star_y = []
    for layer, head in selected_after_mlp:
        if 0 <= layer < values.shape[0] and 0 <= head < values.shape[1]:
            star_x.append(layer)
            star_y.append(abs(float(values[layer, head])))
    if star_x:
        ax.scatter(
            star_x,
            star_y,
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Key Heads with\nintervention on MLPs",
        )
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel(r"$\mathcal{I}_{l,h}$", fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_yticks([])
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True)


def _plot_head_metrics(
    baseline_dir: Path,
    selected: Set[Head],
    selected_after_mlp: Set[Head],
    output_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    metric_specs = [
        ("kl_p_yes.npy", "head_logit_kl_p_yes.pdf"),
        (
            "mean_diff_p_yes.npy",
            "head_logit_mean_diff_p_yes.pdf",
        ),
    ]
    for baseline_name, output_name in metric_specs:
        values = np.load(baseline_dir / baseline_name)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        _plot_head_metric(ax, values, selected, selected_after_mlp)
        fig.tight_layout()
        out = output_dir / output_name
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def _plot_exp20(exp20_dir: Path, output_dir: Path) -> List[Path]:
    specs = [
        (
            "mlp_kl_p_yes.npy",
            "mlp_kl_p_yes_intervened.npy",
            "qwen_moe_mlp_kl_p_yes.pdf",
        ),
        (
            "mlp_mean_diff_p_yes.npy",
            "mlp_mean_diff_p_yes_intervened.npy",
            "qwen_moe_mlp_mean_diff_p_yes.pdf",
        ),
    ]
    outputs: List[Path] = []
    for original_name, intervened_name, output_name in specs:
        original = np.load(exp20_dir / original_name)
        intervened = np.load(exp20_dir / intervened_name)
        if original.shape != intervened.shape:
            raise ValueError(f"EXP20 shape mismatch: {original.shape} != {intervened.shape}")
        x = np.arange(len(original))
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(x, original, marker="o", label="Original")
        ax.plot(
            x,
            intervened,
            marker="s",
            linestyle="--",
            label="Intervened on Key Heads",
        )
        ax.set_xlabel("Layer", fontweight="bold")
        ax.set_ylabel(r"$\mathcal{I}_{l}$")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=16, frameon=True)
        fig.tight_layout()
        out = output_dir / output_name
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)

    residual = np.load(exp20_dir / "mlp_input_mean_abs_diff_p_race.npy")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(np.arange(len(residual)), residual, marker="o", label="p(race)")
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel(r"$\Delta_S$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=16, frameon=True)
    fig.tight_layout()
    out = output_dir / "qwen_moe_mlp_input_cumulative_residual_p_race.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)

    cosine = np.load(exp20_dir / "mlp_input_delta_cos_race_signed.npy")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(np.arange(len(cosine)), cosine, marker="o", label="Race direction")
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel(r"$\cos(\Delta h_l, v_S)$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=16, frameon=True)
    fig.tight_layout()
    out = output_dir / "qwen_moe_mlp_input_cumulative_residual_race_cosine.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)
    return outputs


def _load_resume_gap(path: Path) -> float:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values.append(abs(float(row["fact_p_yes"]) - float(row["cf_p_yes"])))
            except (KeyError, ValueError):
                continue
    if not values:
        raise ValueError(f"No paired Resume values in {path}")
    return float(np.mean(values))


def _load_discrim_gap(path: Path) -> float:
    rows: Dict[int, Tuple[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sample_id = int(row["sample_id"])
                matched_id = int(row["matched_id"])
                rows[sample_id] = (matched_id, float(row["p_yes"]))
            except (KeyError, TypeError, ValueError):
                continue
    gaps: List[float] = []
    seen: Set[Tuple[int, int]] = set()
    for sample_id, (matched_id, value) in rows.items():
        if matched_id not in rows:
            continue
        pair = tuple(sorted((sample_id, matched_id)))
        if pair in seen:
            continue
        seen.add(pair)
        gaps.append(abs(value - rows[matched_id][1]))
    if not gaps:
        raise ValueError(f"No matched Discrim-Eval pairs in {path}")
    return float(np.mean(gaps))


def _load_discrim_qid_gaps(path: Path) -> Dict[int, List[float]]:
    rows: Dict[int, Tuple[int, int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prompt_type", "prompt") != "prompt":
                continue
            try:
                sample_id = int(row["sample_id"])
                rows[sample_id] = (
                    int(row["matched_id"]),
                    int(row["decision_question_id"]),
                    float(row["p_yes"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    result: Dict[int, List[float]] = {}
    seen: Set[Tuple[int, int]] = set()
    for sample_id, (matched_id, qid, value) in rows.items():
        if matched_id not in rows:
            continue
        pair = tuple(sorted((sample_id, matched_id)))
        if pair in seen:
            continue
        seen.add(pair)
        matched_qid = rows[matched_id][1]
        if matched_qid != qid:
            continue
        result.setdefault(qid, []).append(abs(value - rows[matched_id][2]))
    if not result:
        raise ValueError(f"No matched Discrim-Eval scenario values in {path}")
    return result


def _plot_all_head_intervention(
    baseline_path: Path,
    sensitive_path: Path,
    random_paths: List[Path],
    output_dir: Path,
    model_label: str,
) -> Path:
    baseline = _load_discrim_qid_gaps(baseline_path)
    sensitive = _load_discrim_qid_gaps(sensitive_path)
    random_runs = [_load_discrim_qid_gaps(path) for path in random_paths]
    qids = sorted(baseline, key=lambda qid: np.mean(baseline[qid]), reverse=True)

    x = np.arange(len(qids))
    def extract(values: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        means = np.asarray([np.mean(values.get(qid, [0.0])) for qid in qids])
        stds = np.asarray([np.std(values.get(qid, [0.0])) for qid in qids])
        return means, stds

    baseline_mean, baseline_std = extract(baseline)
    sensitive_mean, sensitive_std = extract(sensitive)
    random_means = []
    random_stds = []
    for run in random_runs:
        mean, std = extract(run)
        random_means.append(mean)
        random_stds.append(std)
    random_mean = np.mean(np.stack(random_means), axis=0)
    random_std = np.mean(np.stack(random_stds), axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for mean, std, label, color in (
        (baseline_mean, baseline_std, "Baseline (Original)", "tab:blue"),
        (sensitive_mean, sensitive_std, "Key Heads", "tab:red"),
        (random_mean, random_std, "Random Heads", "tab:green"),
    ):
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel(model_label, fontweight="bold")
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=16)
    fig.tight_layout()
    out = output_dir / "head_intervention_sensitive_vs_random_discrim.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_mlp_intervention(
    discrim_baseline: Path,
    discrim_head: Path,
    discrim_mlp: Path,
    output_dir: Path,
    model_label: str,
) -> Path:
    baseline = _load_discrim_qid_gaps(discrim_baseline)
    head = _load_discrim_qid_gaps(discrim_head)
    mlp = _load_discrim_qid_gaps(discrim_mlp)
    qids = sorted(baseline, key=lambda qid: np.mean(baseline[qid]), reverse=True)
    x = np.arange(len(qids))

    def extract(values: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([np.mean(values.get(qid, [0.0])) for qid in qids]),
            np.asarray([np.std(values.get(qid, [0.0])) for qid in qids]),
        )

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for values, label, color in (
        (baseline, "Baseline (Original)", "tab:blue"),
        (head, "Head Negative Intervention", "tab:red"),
        (mlp, "MLP Negative Intervention", "purple"),
    ):
        mean, std = extract(values)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel(model_label, fontweight="bold")
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()
    out = output_dir / "head_vs_mlp_intervention_discrim.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_router(router_dir: Path, output_dir: Path) -> Path:
    rows: List[dict] = []
    with (router_dir / "router_metrics_by_layer.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (router_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    kinds = sorted({row["router_kind"] for row in rows})
    colors = {
        "native_fact_cf": "tab:blue",
        "fact_head_change": "tab:red",
        "cf_head_change": "tab:green",
    }
    labels = {
        "native_fact_cf": "Fact vs CF",
        "fact_head_change": "Fact: original vs head intervention",
        "cf_head_change": "CF: original vs head intervention",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for kind in kinds:
        selected_rows = sorted(
            (row for row in rows if row["router_kind"] == kind),
            key=lambda row: int(row["layer"]),
        )
        for prefix, color in colors.items():
            axes[0].plot(
                [int(row["layer"]) for row in selected_rows],
                [float(row[f"{prefix}_js"]) for row in selected_rows],
                marker="o",
                markersize=3,
                color=color,
                alpha=0.8,
                label=f"{labels[prefix]} ({kind})",
            )
            axes[1].plot(
                [int(row["layer"]) for row in selected_rows],
                [float(row[f"{prefix}_topk_overlap"]) for row in selected_rows],
                marker="o",
                markersize=3,
                color=color,
                alpha=0.8,
                label=f"{labels[prefix]} ({kind})",
            )
    axes[0].set_title("Router distribution change")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Jensen-Shannon divergence")
    axes[0].grid(alpha=0.3, linestyle="--", linewidth=0.5)
    axes[1].set_title("Top-k expert stability")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Top-k overlap")
    axes[1].grid(alpha=0.3, linestyle="--", linewidth=0.5)

    final_gaps = metadata["final_mean_abs_fact_cf_p_yes_gap"]
    names = ["native_router", "frozen_fact", "head_intervention"]
    axes[2].bar(
        np.arange(len(names)),
        [float(final_gaps[name]) for name in names],
        color=["tab:blue", "tab:orange", "tab:red"],
    )
    axes[2].set_xticks(np.arange(len(names)), ["Native", "Frozen fact\nrouter", "Head\nintervention"])
    axes[2].set_title("End-to-end fairness")
    axes[2].set_ylabel("Mean absolute paired p(yes) gap")
    axes[2].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=3, fontsize=10, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = output_dir / "qwen_moe_router_metrics.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--selected_heads_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(args.selected_heads_json)
    head_dir = root / "pattern_analysis/head_logit_resume_top100"
    head_mlp_dir = root / "pattern_analysis/head_logit_with_mlp_intervention"
    exp20_dir = root / "mlp_analysis/exp20_residual_top100"
    router_dir = root / "mlp_analysis/router_top100"
    downstream = root / "downstream_evaluation"
    discrim_mlp = root / "intervention_ablation/mlp_discrim/per_sample.csv"
    discrim_sensitive = root / "intervention_ablation/head_discrim_all/negative_seed_42/per_sample.csv"
    discrim_random = [
        root / "intervention_ablation/head_discrim_all/negative_random_seed_42/per_sample.csv"
    ]
    required = [
        selected_path,
        head_dir / "kl_p_yes.npy",
        head_dir / "mean_diff_p_yes.npy",
        head_mlp_dir / "kl_p_yes_mlp.npy",
        head_mlp_dir / "mean_diff_p_yes_mlp.npy",
        exp20_dir / "mlp_kl_p_yes.npy",
        exp20_dir / "mlp_kl_p_yes_intervened.npy",
        exp20_dir / "mlp_mean_diff_p_yes.npy",
        exp20_dir / "mlp_mean_diff_p_yes_intervened.npy",
        exp20_dir / "mlp_input_mean_abs_diff_p_race.npy",
        exp20_dir / "mlp_input_delta_cos_race_signed.npy",
        head_mlp_dir / "selected_heads_mlp_elbow.json",
        router_dir / "router_metrics_by_layer.csv",
        router_dir / "metadata.json",
        downstream / "discrim_baseline_resume_standard_fresh.csv",
        discrim_mlp,
        discrim_sensitive,
    ]
    required.extend(discrim_random)
    _require(required)
    _style()
    selected = _load_heads(selected_path)
    selected_after_mlp = _load_heads(head_mlp_dir / "selected_heads_mlp_elbow.json")
    style_root = "/home/common1/hwluo/project/fairness_llm_new copy"
    exp20_output = output_dir / "exp20_head_mlp_layer_analysis"
    exp8_output = output_dir / "exp8_head_intervention"
    exp15_output = output_dir / "exp15_head_vs_mlp_intervention"
    router_output = output_dir / "moe_router_analysis"
    for directory in (exp20_output, exp8_output, exp15_output, router_output):
        directory.mkdir(parents=True, exist_ok=True)

    generated_records: List[Dict[str, str]] = []
    head_outputs = _plot_head_metrics(
        head_dir, selected, selected_after_mlp, exp20_output
    )
    for path in head_outputs:
        generated_records.append(
            _generated_record(
                path,
                "exp20_head_mlp_layer_analysis",
                f"{style_root}/exp20/plot_head_kl_layers_single.py",
            )
        )
    mlp_outputs = _plot_exp20(exp20_dir, exp20_output)
    for path in mlp_outputs:
        reference = (
            f"{style_root}/exp20/plot_mlp_input_p_race_layers_single.py"
            if "cumulative_residual" in path.name
            else f"{style_root}/exp20/plot_mlp_output_kl_layers_single.py"
        )
        generated_records.append(
            _generated_record(path, "exp20_head_mlp_layer_analysis", reference)
        )
    exp8_path = (
        _plot_all_head_intervention(
            downstream / "discrim_baseline_resume_standard_fresh.csv",
            discrim_sensitive,
            discrim_random,
            exp8_output,
            args.model_name,
        )
    )
    generated_records.append(
        _generated_record(
            exp8_path,
            "exp8_head_intervention",
            f"{style_root}/exp8/plot_intervention_single_LLM.py",
        )
    )
    exp15_path = (
        _plot_mlp_intervention(
            downstream / "discrim_baseline_resume_standard_fresh.csv",
            discrim_sensitive,
            discrim_mlp,
            exp15_output,
            args.model_name,
        )
    )
    generated_records.append(
        _generated_record(
            exp15_path,
            "exp15_head_vs_mlp_intervention",
            f"{style_root}/exp15/plot_intervention_qwen_llama_grid_with_mlp.py",
        )
    )
    router_path = _plot_router(router_dir, router_output)
    generated_records.append(
        _generated_record(
            router_path,
            "moe_router_analysis",
            "MOE-only; Times New Roman/tab-color/dashed-grid paper convention",
        )
    )
    collected_records = _collect_existing_plots(root, output_dir, selected)

    legacy_flat_names = {Path(record["destination"]).name for record in generated_records}
    legacy_flat_names.add("mlp_intervention_resume_discrim.pdf")
    for name in legacy_flat_names:
        legacy_path = output_dir / name
        if legacy_path.is_file():
            legacy_path.unlink()

    artifacts = generated_records + collected_records
    counts_by_category = dict(sorted(Counter(
        record["category"] for record in artifacts
    ).items()))
    manifest = {
        "model_name": args.model_name,
        "results_root": str(root),
        "selected_heads_json": str(selected_path),
        "probe_surface": "next_mlp_input_cumulative_residual",
        "probe_norm": "none",
        "semantic_projection": "W_U h",
        "style_reference_root": style_root,
        "style_references": {
            "head_intervention": "exp8/plot_intervention_single_LLM.py",
            "resume_head_count": "exp9/plot_intervention_all_models.py",
            "discrim_head_count": "exp10/plot_discrim_eval_head_count.py",
            "head_logit": "exp20/plot_head_kl_layers_single.py",
            "mlp_layers": "exp20/plot_mlp_output_kl_layers_single.py",
            "mlp_race": "exp20/plot_mlp_input_p_race_layers_single.py",
            "head_vs_mlp": "exp15/plot_intervention_qwen_llama_grid_with_mlp.py",
            "router": "MOE-only; uses the same Times New Roman/tab-color/dashed-grid convention",
        },
        "organization": "Structured by paper experiment; canonical source plots are preserved.",
        "artifact_count": len(artifacts),
        "counts_by_category": counts_by_category,
        "artifacts": artifacts,
        "input_files": [str(path) for path in required],
    }
    with (output_dir / "plot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for record in artifacts:
        print(f"Saved: {record['destination']}")
    print(f"Organized {len(artifacts)} plot artifacts across {len(counts_by_category)} categories")


if __name__ == "__main__":
    main()
