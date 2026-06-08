import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _set_font() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"


def _load_gaps(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"index", "fact_p_yes", "cf_p_yes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    out = df[["index", "fact_p_yes", "cf_p_yes"]].copy()
    out["index"] = out["index"].astype(int)
    out["abs_gap"] = (out["fact_p_yes"] - out["cf_p_yes"]).abs()
    return out


def _aligned_gaps(
    baseline_csv: str,
    series_csvs: Dict[str, str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    baseline = _load_gaps(baseline_csv).rename(columns={"abs_gap": "baseline_gap"})
    merged = baseline[["index", "baseline_gap"]].copy()

    for label, csv_path in series_csvs.items():
        series = _load_gaps(csv_path).rename(columns={"abs_gap": label})
        merged = merged.merge(series[["index", label]], on="index", how="inner")

    if merged.empty:
        raise ValueError("No overlapping sample indices between baseline and branch CSVs.")

    merged = merged.sort_values("baseline_gap", ascending=False).reset_index(drop=True)
    return merged["baseline_gap"].to_numpy(dtype=np.float64), {
        label: merged[label].to_numpy(dtype=np.float64) for label in series_csvs
    }


def _existing_optional(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--pfairft_csv", type=str, required=True)
    parser.add_argument("--pfairft_kl_csv", type=str, default=None)
    parser.add_argument("--global_csv", type=str, default=None)
    parser.add_argument("--pfairft_kl_ce_csv", type=str, default=None)
    parser.add_argument("--out_pdf", type=str, required=True)
    parser.add_argument("--out_png", type=str, default=None)
    parser.add_argument("--model_label", type=str, default="JetMoE")
    parser.add_argument("--pfairft_label", type=str, default="PFairFT")
    args = parser.parse_args()

    _set_font()
    series: List[Tuple[str, str, str, float]] = [
        (args.pfairft_label, args.pfairft_csv, "tab:orange", 2.0),
    ]
    optional_series = [
        ("PFairFT-KL", _existing_optional(args.pfairft_kl_csv), "tab:green", 2.0),
        ("Global", _existing_optional(args.global_csv), "tab:red", 1.5),
        ("PFairFT-KL-CE", _existing_optional(args.pfairft_kl_ce_csv), "black", 2.0),
    ]
    series.extend(s for s in optional_series if s[1])

    baseline_gap, branch_gaps = _aligned_gaps(
        args.baseline_csv,
        {label: csv_path for label, csv_path, _, _ in series},
    )
    xs = np.arange(len(baseline_gap))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, baseline_gap, label="baseline", color="tab:blue", linewidth=2.5)
    for label, _, color, linewidth in series:
        ax.plot(xs, branch_gaps[label], label=label, color=color, linewidth=linewidth)

    ymax = max(
        [float(baseline_gap.max()), 0.01]
        + [float(gaps.max()) for gaps in branch_gaps.values()]
    )
    ax.set_ylim(-0.005, ymax * 1.18)
    ax.set_ylabel("Fairness Violation↓", fontweight="bold")
    ax.set_xlabel(f"{args.model_label} Resume Samples", fontweight="bold")
    ax.set_xticks([])
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
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {args.out_pdf}")
    if args.out_png:
        print(f"PNG saved to {args.out_png}")


if __name__ == "__main__":
    main()
