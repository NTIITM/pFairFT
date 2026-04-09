#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def _load_csv(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["p_yes"] = pd.to_numeric(df["p_yes"], errors="coerce")
    df = df.dropna(subset=["p_yes"])
    df["source"] = source
    return df


def _compute_pair_gap(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = {"sample_id", "matched_id", "model", "decision_question_id", "p_yes"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    df2 = df.copy()
    df2["sample_id"] = pd.to_numeric(df2["sample_id"], errors="coerce")
    df2["matched_id"] = pd.to_numeric(df2["matched_id"], errors="coerce")
    df2["decision_question_id"] = pd.to_numeric(df2["decision_question_id"], errors="coerce")
    df2 = df2.dropna(subset=["sample_id", "matched_id", "decision_question_id"])

    df2["sample_id"] = df2["sample_id"].astype(int)
    df2["matched_id"] = df2["matched_id"].astype(int)
    df2["decision_question_id"] = df2["decision_question_id"].astype(int)

    other = df2[["model", "source", "sample_id", "p_yes"]].rename(
        columns={"sample_id": "matched_id", "p_yes": "matched_p_yes"}
    )
    merged = df2.merge(other, on=["model", "source", "matched_id"], how="inner")

    merged = merged[merged["sample_id"] < merged["matched_id"]].copy()

    merged["gap"] = (merged["p_yes"] - merged["matched_p_yes"]).abs()
    return merged[["model", "source", "decision_question_id", "sample_id", "matched_id", "gap"]]


def plot_gap_overall_bar(gap_df: pd.DataFrame, output_path: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    grouped = gap_df.groupby(["model", "source"])["gap"].mean().reset_index()

    models = [m for m in MODEL_ORDER if m in set(grouped["model"])]
    if not models:
        models = list(grouped["model"].unique())

    source_order = ["baseline", "exp4", "exp5"]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    x = list(range(len(models)))

    for idx, src in enumerate(source_order):
        vals = []
        for m in models:
            row = grouped[(grouped["model"] == m) & (grouped["source"] == src)]
            vals.append(float(row["gap"].values[0]) if len(row) else np.nan)
        ax.bar([i + idx * width for i in x], vals, width=width, label=src)

    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha="right")
    ax.set_ylabel("Mean |p_yes(a) - p_yes(b)|")
    ax.set_title("Fairness gap (prompt): baseline vs exp4 vs exp5")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gap_by_question_topk(gap_df: pd.DataFrame, output_path: str, topk: int = 10):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    gq = (
        gap_df.groupby(["source", "decision_question_id"])["gap"].mean().reset_index()
    )

    base = gq[gq["source"] == "baseline"].sort_values("gap", ascending=False)
    top_qids = base["decision_question_id"].head(topk).tolist()
    if not top_qids:
        print("No baseline question gaps found; skip question-level plot")
        return

    gq = gq[gq["decision_question_id"].isin(top_qids)]

    sources = ["baseline", "exp4", "exp5"]
    x = np.arange(len(top_qids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, src in enumerate(sources):
        vals = []
        for qid in top_qids:
            row = gq[(gq["source"] == src) & (gq["decision_question_id"] == qid)]
            vals.append(float(row["gap"].values[0]) if len(row) else np.nan)
        ax.bar(x + idx * width, vals, width=width, label=src)

    ax.set_xticks(x + width)
    ax.set_xticklabels([str(q) for q in top_qids], rotation=0)
    ax.set_xlabel("decision_question_id (top-k by baseline gap)")
    ax.set_ylabel("Mean |p_yes(a) - p_yes(b)|")
    ax.set_title(f"Question-level fairness gap (top {topk})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gap_by_question_all(gap_df: pd.DataFrame, output_path: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    gq = (
        gap_df.groupby(["source", "decision_question_id"])["gap"].mean().reset_index()
    )

    base = gq[gq["source"] == "baseline"].sort_values("gap", ascending=False)
    ordered_qids = base["decision_question_id"].tolist()
    if not ordered_qids:
        print("No baseline question gaps found; skip all-questions plot")
        return

    gq = gq[gq["decision_question_id"].isin(ordered_qids)]
    sources = ["baseline", "exp4", "exp5"]
    xs = np.arange(len(ordered_qids))

    fig, ax = plt.subplots(figsize=(12, 5))
    for src in sources:
        vals = []
        for qid in ordered_qids:
            row = gq[(gq["source"] == src) & (gq["decision_question_id"] == qid)]
            vals.append(float(row["gap"].values[0]) if len(row) else np.nan)
        ax.plot(xs, vals, label=src, linewidth=2)

    ax.set_xticks([])
    ax.set_xlabel("Question (ordered by baseline mean gap)")
    ax.set_ylabel("Mean |p_yes(a) - p_yes(b)|")
    ax.set_title("Question-level fairness gap (all questions)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _plot_single_model_axes_gap(ax, model_name: str, gq_model: pd.DataFrame):
    base = gq_model[gq_model["source"] == "baseline"].sort_values("gap", ascending=False)
    ordered_qids = base["decision_question_id"].tolist()
    if not ordered_qids:
        ax.set_title(f"{model_name}\n(no baseline data)", fontweight="bold")
        ax.set_xticks([])
        return []

    xs = list(range(len(ordered_qids)))
    sources = ["baseline", "exp4", "exp5"]
    colors = {
        "baseline": "tab:blue",
        "exp4": "tab:orange",
        "exp5": "tab:green",
    }

    handles = []
    for src in sources:
        sub = gq_model[gq_model["source"] == src]
        vals = []
        for qid in ordered_qids:
            row = sub[sub["decision_question_id"] == qid]
            vals.append(float(row["gap"].values[0]) if len(row) else np.nan)

        line, = ax.plot(
            xs,
            vals,
            label=src,
            color=colors.get(src, None),
            linewidth=2,
        )
        handles.append(line)

    ax.set_xlabel(MODEL_DISPLAY.get(model_name, model_name), fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    return handles


def plot_gap_qwen_and_llama_grids(gap_df: pd.DataFrame, out_qwen: str, out_llama: str):
    _set_font()

    gq = (
        gap_df.groupby(["model", "source", "decision_question_id"])["gap"]
        .mean()
        .reset_index()
    )

    # Qwen grid
    qwen_models = [m for m in QWEN_MODELS if m in set(gq["model"])]
    if qwen_models:
        fig_qwen, axes_qwen = plt.subplots(1, len(qwen_models), figsize=(12, 4), sharey=True)
        if len(qwen_models) == 1:
            axes_qwen = [axes_qwen]
        handles_q = []
        labels_q = []

        for ax, model in zip(axes_qwen, qwen_models):
            gq_model = gq[gq["model"] == model]
            if gq_model.empty:
                ax.set_title(f"{model}\n(no data)", fontweight="bold")
                ax.set_xticks([])
                continue

            h = _plot_single_model_axes_gap(ax, model, gq_model)
            if not handles_q and h:
                handles_q = h
                labels_q = [hh.get_label() for hh in h]

        if handles_q:
            fig_qwen.legend(
                handles_q,
                labels_q,
                loc="upper center",
                ncol=len(labels_q),
                bbox_to_anchor=(0.5, 0.98),
                fontsize=16,
            )
        if len(axes_qwen) > 0:
            axes_qwen[0].set_ylabel("Fairness Violation↓", fontweight="bold")
        fig_qwen.tight_layout(rect=[0.0, 0.10, 0.98, 0.90], w_pad=0.25)
        os.makedirs(os.path.dirname(out_qwen) or ".", exist_ok=True)
        fig_qwen.savefig(out_qwen, dpi=200)
        plt.close(fig_qwen)
        print(f"Saved: {out_qwen}")

    # Llama grid
    llama_models = [m for m in LLAMA_MODELS if m in set(gq["model"])]
    if llama_models:
        fig_llama, axes_llama = plt.subplots(1, len(llama_models), figsize=(12, 4), sharey=True)
        if len(llama_models) == 1:
            axes_llama = [axes_llama]
        handles_l = []
        labels_l = []

        for ax, model in zip(axes_llama, llama_models):
            gq_model = gq[gq["model"] == model]
            if gq_model.empty:
                ax.set_title(f"{model}\n(no data)", fontweight="bold")
                ax.set_xticks([])
                continue

            h = _plot_single_model_axes_gap(ax, model, gq_model)
            if not handles_l and h:
                handles_l = h
                labels_l = [hh.get_label() for hh in h]

        if handles_l:
            fig_llama.legend(
                handles_l,
                labels_l,
                loc="upper center",
                ncol=len(labels_l),
                bbox_to_anchor=(0.5, 0.98),
                fontsize=16,
            )
        if len(axes_llama) > 0:
            axes_llama[0].set_ylabel("Fairness Violation↓", fontweight="bold")
        fig_llama.tight_layout(rect=[0.0, 0.10, 0.98, 0.90], w_pad=0.25)
        os.makedirs(os.path.dirname(out_llama) or ".", exist_ok=True)
        fig_llama.savefig(out_llama, dpi=200)
        plt.close(fig_llama)
        print(f"Saved: {out_llama}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot exp16 fairness gap results (baseline vs exp4 vs exp5)."
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp1/per_sample_details_all_models.csv",
        help="Baseline per-sample CSV (from exp1).",
    )
    parser.add_argument("--exp4_csv", default="/home/common1/hwluo/project/pFairFT/exp16/per_sample_details_exp4.csv", type=str,  help="Exp4 per-sample CSV")
    parser.add_argument("--exp5_csv", default="/home/common1/hwluo/project/pFairFT/exp16/per_sample_details_exp5.csv", type=str,  help="Exp5 per-sample CSV")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Default: directory of --exp4_csv",
    )
    parser.add_argument(
        "--topk_questions",
        type=int,
        default=10,
        help="Top-k decision_question_id selected by baseline mean gap.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(args.exp4_csv))
    os.makedirs(out_dir, exist_ok=True)

    df_base = _load_csv(args.baseline_csv, "baseline")
    df_exp4 = _load_csv(args.exp4_csv, "exp4")
    df_exp5 = _load_csv(args.exp5_csv, "exp5")

    df_base = df_base[df_base["prompt_type"] == "prompt"]
    df_exp4 = df_exp4[df_exp4["prompt_type"] == "prompt"]
    df_exp5 = df_exp5[df_exp5["prompt_type"] == "prompt"]

    df_all = pd.concat([df_base, df_exp4, df_exp5], ignore_index=True)

    gap_df = _compute_pair_gap(df_all)

    plot_gap_overall_bar(
        gap_df,
        os.path.join(out_dir, "fairness_gap_overall_baseline_vs_exp4_vs_exp5.pdf"),
    )
    plot_gap_by_question_topk(
        gap_df,
        os.path.join(out_dir, f"fairness_gap_questions_top{args.topk_questions}.pdf"),
        topk=args.topk_questions,
    )
    plot_gap_by_question_all(
        gap_df,
        os.path.join(out_dir, "fairness_gap_questions_all.pdf"),
    )
    plot_gap_qwen_and_llama_grids(
        gap_df,
        os.path.join(out_dir, "qwen_models_fairness_gap_grid.pdf"),
        os.path.join(out_dir, "llama_models_fairness_gap_grid.pdf"),
    )


if __name__ == "__main__":
    main()
