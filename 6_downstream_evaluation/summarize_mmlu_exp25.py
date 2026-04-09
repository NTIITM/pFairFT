#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp25_root", type=str, default="/home/common1/hwluo/project/pFairFT/exp25")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=[
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct",
            "Meta-Llama-3-8B-Instruct",
        ],
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp25/mmlu_summary_exp25.csv",
    )
    args = parser.parse_args()

    modes = ["baseline", "partial", "all"]

    rows: List[Dict[str, object]] = []
    for model in args.models:
        out_dir = os.path.join(args.exp25_root, f"results_{model}")
        row: Dict[str, object] = {"model": model}
        has_any = False

        for mode in modes:
            fp = os.path.join(out_dir, f"mmlu_{args.split}_{args.max_samples}_{mode}.json")
            js = _load_json(fp)
            if js is None:
                row[mode] = float("nan")
                continue
            row[mode] = _safe_float(js.get("accuracy"))
            has_any = True

        if has_any:
            # deltas
            b = float(row.get("baseline", float("nan")))
            p = float(row.get("partial", float("nan")))
            a = float(row.get("all", float("nan")))
            row["delta_partial_minus_baseline"] = p - b if not np.isnan(p) and not np.isnan(b) else float("nan")
            row["delta_all_minus_baseline"] = a - b if not np.isnan(a) and not np.isnan(b) else float("nan")
            rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "baseline",
                "partial",
                "all",
                "delta_partial_minus_baseline",
                "delta_all_minus_baseline",
            ]
        )
        for r in rows:
            def fmt(v) -> str:
                try:
                    fv = float(v)
                    return f"{fv:.6f}" if not np.isnan(fv) else "NaN"
                except Exception:
                    return "NaN"

            writer.writerow(
                [
                    r["model"],
                    fmt(r.get("baseline")),
                    fmt(r.get("partial")),
                    fmt(r.get("all")),
                    fmt(r.get("delta_partial_minus_baseline")),
                    fmt(r.get("delta_all_minus_baseline")),
                ]
            )

    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
