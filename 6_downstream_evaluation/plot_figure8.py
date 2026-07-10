import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _set_font():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"

def _load_intervention_stats(
    csv_path: str,
) -> Dict[int, Dict[str, float]]:
    sample_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_id = int(row["sample_id"])
                matched_id_str = row.get("matched_id", "").strip()
                matched_id = int(matched_id_str) if matched_id_str else None
                decision_question_id_str = row["decision_question_id"].strip()
                decision_question_id = (
                    int(decision_question_id_str) if decision_question_id_str else None
                )
                p_yes = float(row["p_yes"])
                if decision_question_id is None:
                    continue
                sample_data.append(
                    {
                        "sample_id": sample_id,
                        "matched_id": matched_id,
                        "decision_question_id": decision_question_id,
                        "p_yes": p_yes,
                    }
                )
            except (ValueError, KeyError):
                continue

    data_by_id: Dict[int, dict] = {}
    for s in sample_data:
        data_by_id[int(s["sample_id"])] = {
            "decision_question_id": int(s["decision_question_id"]),
            "p_yes": float(s["p_yes"]),
            "matched_id": s["matched_id"],
        }

    from collections import defaultdict as dd2
    diffs_by_qid: Dict[int, List[float]] = dd2(list)
    processed_pairs = set()

    for sample_id, sample_info in data_by_id.items():
        matched_id = sample_info["matched_id"]
        if matched_id is None or int(matched_id) not in data_by_id:
            continue
        matched_id = int(matched_id)
        pair_key = tuple(sorted([sample_id, matched_id]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        qid = int(sample_info["decision_question_id"])
        matched_info = data_by_id[matched_id]
        if int(matched_info["decision_question_id"]) != qid:
            continue

        diff = abs(float(sample_info["p_yes"]) - float(matched_info["p_yes"]))
        diffs_by_qid[qid].append(diff)

    stats: Dict[int, Dict[str, float]] = {}
    for qid, diffs in diffs_by_qid.items():
        if not diffs:
            continue
        arr = np.array(diffs, dtype=np.float64)
        stats[qid] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)) if len(diffs) > 1 else 0.0,
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--pfairft_csv", type=str, required=True)
    parser.add_argument("--pfairft_kl_csv", type=str, required=True)
    parser.add_argument("--global_csv", type=str, required=True)
    parser.add_argument("--pfairft_kl_ce_csv", type=str, default=None)
    parser.add_argument("--debiased_prompt_csv", type=str, default=None)
    parser.add_argument("--inference_time_csv", type=str, default=None)
    parser.add_argument("--out_pdf", type=str, required=True)
    parser.add_argument("--model_label", type=str, default="Llama 3B", help="Model label for X-axis")
    args = parser.parse_args()

    _set_font()

    stats_b = _load_intervention_stats(args.baseline_csv)
    stats_p = _load_intervention_stats(args.pfairft_csv) if os.path.exists(args.pfairft_csv) else None
    stats_pkl = _load_intervention_stats(args.pfairft_kl_csv) if os.path.exists(args.pfairft_kl_csv) else None
    stats_g = _load_intervention_stats(args.global_csv) if os.path.exists(args.global_csv) else None
    stats_pklce = _load_intervention_stats(args.pfairft_kl_ce_csv) if args.pfairft_kl_ce_csv and os.path.exists(args.pfairft_kl_ce_csv) else None
    stats_debiased = _load_intervention_stats(args.debiased_prompt_csv) if args.debiased_prompt_csv and os.path.exists(args.debiased_prompt_csv) else None
    stats_inference = _load_intervention_stats(args.inference_time_csv) if args.inference_time_csv and os.path.exists(args.inference_time_csv) else None
    if not stats_b:
        raise ValueError(f"Baseline CSV produced no valid matched pairs: {args.baseline_csv}")

    # We sort by baseline's fairness violation descending
    ordered_qids = sorted(
        stats_b.keys(),
        key=lambda q: stats_b[q]["mean"],
        reverse=True,
    )
    xs = list(range(len(ordered_qids)))

    def extract(series_stats):
        if not series_stats:
            return np.full(len(xs), np.nan), np.full(len(xs), np.nan)
        means = [series_stats.get(q, {"mean": np.nan})["mean"] for q in ordered_qids]
        stds = [series_stats.get(q, {"std": np.nan})["std"] for q in ordered_qids]
        return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    means_b, stds_b = extract(stats_b)
    ax.plot(xs, means_b, label="baseline", color="tab:blue", linewidth=2.5)

    if stats_p is not None:
        means_p, _ = extract(stats_p)
        ax.plot(xs, means_p, label="PFairFT", color="tab:orange", linewidth=2)
        
    if stats_pkl is not None:
        means_pkl, _ = extract(stats_pkl)
        ax.plot(xs, means_pkl, label="PFairFT-KL", color="tab:green", linewidth=2)
        
    if stats_g is not None:
        means_g, _ = extract(stats_g)
        ax.plot(xs, means_g, label="Global", color="tab:red", linewidth=1.5)

    if stats_pklce is not None:
        means_pklce, _ = extract(stats_pklce)
        ax.plot(xs, means_pklce, label="PFairFT-KL-CE", color="black", linewidth=2.0)

    if stats_debiased is not None:
        means_debiased, _ = extract(stats_debiased)
        ax.plot(xs, means_debiased, label="Debiased Prompt", color="tab:purple", linewidth=1.5)

    if stats_inference is not None:
        means_inference, _ = extract(stats_inference)
        ax.plot(xs, means_inference, label="Inference Time", color="tab:brown", linewidth=1.5)

    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    ax.set_xlabel(f"{args.model_label} Samples", fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(-0.02, 0.55)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.15),
        fontsize=12,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    fig.savefig(args.out_pdf, dpi=200)
    plt.close(fig)
    metadata = {
        "model_label": args.model_label,
        "ordered_qids": ordered_qids,
        "inputs": {
            "baseline": args.baseline_csv,
            "pfairft": args.pfairft_csv,
            "pfairft_kl": args.pfairft_kl_csv,
            "global": args.global_csv,
            "pfairft_kl_ce": args.pfairft_kl_ce_csv,
            "debiased_prompt": args.debiased_prompt_csv,
            "inference_time": args.inference_time_csv,
        },
        "output_pdf": args.out_pdf,
    }
    with open(args.out_pdf + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Plot saved to {args.out_pdf}")

if __name__ == "__main__":
    main()
