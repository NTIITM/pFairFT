#!/usr/bin/env python3
"""Render standalone and grouped cross-model Figure 5 appendix panels."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_figure5 import (
    build_scene_transitions,
    configure_style,
    plot_panel_g,
    plot_panel_i,
    plot_two_lines,
    save_plot_pair,
)

MODELS = {
    "Qwen3-1.7B": "Qwen 1.7B",
    "Qwen3-4B": "Qwen 4B",
    "Qwen3-8B": "Qwen 8B",
    "Llama-3.2-1B-Instruct": "Llama 1B",
    "Llama-3.2-3B-Instruct": "Llama 3B",
    "DeepSeek-V2-Lite-Chat": "DeepSeek-V2-Lite",
    "JetMoE-8B-Chat": "JetMoE-8B",
    "OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B",
}
CORE_MODELS = tuple(MODELS)[:6]
PANEL_MODELS = {
    "a": CORE_MODELS,
    "d": CORE_MODELS,
    "g": tuple(MODELS),
    "i": CORE_MODELS,
}
PANEL_WIDTH_CM = {"a": 10.5, "d": 7.0, "g": 10.5, "i": 10.5}
DEFAULT_GLOBAL = "global_lora_raw_summary_qv_current_ranking_full_3epoch"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired(path: Path, prompt_type: str | None = "prompt") -> dict:
    by_sample = {}
    for row in read_csv(path):
        if prompt_type is not None and row.get("prompt_type", "prompt") != prompt_type:
            continue
        by_sample[int(row["sample_id"])] = {
            "matched": int(row["matched_id"]),
            "question": int(row["decision_question_id"]),
            "probability": float(row["p_yes"]),
        }

    grouped: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for sample_id, sample in by_sample.items():
        matched_id = sample["matched"]
        pair = tuple(sorted((sample_id, matched_id)))
        if pair in seen or matched_id not in by_sample:
            continue
        seen.add(pair)
        grouped[sample["question"]].append(
            abs(sample["probability"] - by_sample[matched_id]["probability"])
        )
    return {
        question: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for question, values in grouped.items()
    }


def validate(root: Path, global_name: str) -> dict[str, Path]:
    downstream = root / "downstream_evaluation"
    paths = {
        "original": downstream / "discrim_baseline_pkfair_3epoch_fresh.csv",
        "debiased_prompt": downstream
        / "discrim_baseline_debiased_prompt_figure5_fresh.csv",
        "global": downstream / f"discrim_{global_name}.csv",
        "pfairft": downstream / "discrim_pkfair_kl_pkfair_3epoch_fresh.csv",
        "pfairft_kl": downstream / "discrim_pkfair_pkfair_3epoch_fresh.csv",
        "pfairft_ce": downstream / "discrim_pkfair_ce_pkfair_3epoch_fresh.csv",
        "inference_time": root / "inference_time_figure5" / "discrim_partial.csv",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{root.name}: missing {name}: {path}")
        row_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
        if row_count != 2520:
            raise ValueError(f"{path}: expected 2520 rows, found {row_count}")
    return paths


def load_model_data(root: Path, global_name: str) -> dict:
    paths = validate(root, global_name)
    return {
        name: paired(
            path,
            None
            if name == "inference_time"
            else ("debiased_prompt" if name == "debiased_prompt" else "prompt"),
        )
        for name, path in paths.items()
    }


def make_plotter(letter: str, data: dict, label: str):
    if letter == "a":
        return lambda fig, spec: plot_two_lines(
            fig,
            spec,
            data,
            "original",
            "debiased_prompt",
            "Original Prompt",
            "Debiased Prompt",
            "tab:blue",
            "tab:orange",
            xlabel=label,
        )
    if letter == "d":
        return lambda fig, spec: plot_two_lines(
            fig,
            spec,
            data,
            "original",
            "global",
            "Original",
            "Global",
            "tab:blue",
            "tab:red",
            xlabel=label,
        )
    if letter == "g":
        return lambda fig, spec: plot_panel_g(fig, spec, data, xlabel=label)
    if letter == "i":
        transitions = build_scene_transitions(data)

        def plot_panel(fig, spec):
            plot_panel_i(fig, spec, {"transitions": transitions})
            fig.axes[-1].set_title(label, fontsize=8.0, fontweight="bold", pad=15)

        return plot_panel
    raise ValueError(f"Unknown panel: {letter}")


def save_combined(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.pdf", dpi=600)
    fig.savefig(f"{path}.png", dpi=400)
    plt.close(fig)


def render_single(
    out: Path,
    panels: set[str],
    selected_models: set[str],
    model_data: dict[str, dict],
) -> None:
    for letter in ("a", "d", "g", "i"):
        if letter not in panels:
            continue
        for model in PANEL_MODELS[letter]:
            if model not in selected_models:
                continue
            stem = out / f"panel_{letter}" / model
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_plot_pair(
                stem,
                PANEL_WIDTH_CM[letter],
                make_plotter(letter, model_data[model], MODELS[model]),
                letter,
            )


def render_combined(
    out: Path,
    panels: set[str],
    selected_models: set[str],
    model_data: dict[str, dict],
) -> None:
    for letter in ("a", "d", "g", "i"):
        if letter not in panels:
            continue
        panel_models = [
            model for model in PANEL_MODELS[letter] if model in selected_models
        ]
        if not panel_models:
            continue
        columns = min(3, len(panel_models))
        rows = math.ceil(len(panel_models) / columns)
        cell_width = PANEL_WIDTH_CM[letter] / 2.54
        cell_height = 4.8 / 2.54
        fig = plt.figure(
            figsize=(cell_width * columns, cell_height * rows),
            facecolor="white",
        )
        grid = fig.add_gridspec(
            rows,
            columns,
            left=0.055,
            right=0.99,
            bottom=0.08,
            top=0.90 if letter != "i" else 0.86,
            hspace=0.48 if letter != "i" else 0.64,
            wspace=0.22,
        )
        legend_handles = []
        legend_labels = []
        for index, model in enumerate(panel_models):
            row, column = divmod(index, columns)
            before = len(fig.axes)
            make_plotter(letter, model_data[model], MODELS[model])(
                fig, grid[row, column]
            )
            for ax in fig.axes[before:]:
                handles, labels = ax.get_legend_handles_labels()
                if not legend_handles and handles:
                    legend_handles, legend_labels = handles, labels
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=min(4, len(legend_labels)),
                frameon=False,
                fontsize=6.5,
            )
        save_combined(fig, out / "combined" / f"panel_{letter}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--global_name", default=DEFAULT_GLOBAL)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--panels", default="a,d,g,i")
    args = parser.parse_args()

    project = Path(args.project_root)
    panels = {panel.strip() for panel in args.panels.split(",") if panel.strip()}
    unknown_panels = panels - set(PANEL_MODELS)
    if unknown_panels:
        raise ValueError(f"Unknown panels: {','.join(sorted(unknown_panels))}")
    requested_models = [
        model.strip() for model in args.models.split(",") if model.strip()
    ]
    unknown_models = set(requested_models) - set(MODELS)
    if unknown_models:
        raise ValueError(f"Unknown models: {','.join(sorted(unknown_models))}")
    selected_models = set(requested_models)
    needed_models = {
        model
        for panel in panels
        for model in PANEL_MODELS[panel]
        if model in selected_models
    }

    configure_style()
    out = Path(__file__).resolve().parent / "appendix"
    model_data = {
        model: load_model_data(project / "results" / model, args.global_name)
        for model in MODELS
        if model in needed_models
    }
    render_single(out, panels, selected_models, model_data)
    render_combined(out, panels, selected_models, model_data)
    print(f"Wrote Figure 5 core appendix to {out}")


if __name__ == "__main__":
    main()
