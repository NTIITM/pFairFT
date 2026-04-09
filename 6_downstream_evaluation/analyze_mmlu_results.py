#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 exp14/mmlu_results 下的 MMLU 评估 JSON，整理 baseline / exp4 / exp5
三者的 overall accuracy，对齐到同一行并导出 CSV / 终端表格。

约定的文件命名：
  - mmlu_baseline_${MODEL_NAME}_top100.json
  - mmlu_precision_fairness_${MODEL_NAME}_top100.json
  - mmlu_lora_${MODEL_NAME}_top100.json
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional


EXP_ROOT = "/home/common1/hwluo/project/pFairFT"
EXP14_DIR = os.path.join(EXP_ROOT, "exp14")
RESULT_DIR = os.path.join(EXP14_DIR, "mmlu_results")


@dataclass
class TripletMetrics:
    model_suffix: str  # 例如：Llama-3.2-1B-Instruct_top100
    baseline_acc: Optional[float] = None
    exp4_acc: Optional[float] = None
    exp5_acc: Optional[float] = None


def parse_filename(fname: str):
    """
    从文件名中解析 (family, model_suffix)。
    支持的模式：
      - mmlu_baseline_${MODEL_SUFFIX}.json
      - mmlu_precision_fairness_${MODEL_SUFFIX}.json
      - mmlu_lora_${MODEL_SUFFIX}.json
    """
    if not fname.startswith("mmlu_") or not fname.endswith(".json"):
        return None

    core = fname[len("mmlu_") : -len(".json")]

    if core.startswith("baseline_"):
        return "baseline", core[len("baseline_") :]
    if core.startswith("precision_fairness_"):
        return "exp4", core[len("precision_fairness_") :]
    if core.startswith("lora_"):
        return "exp5", core[len("lora_") :]

    return None


def load_overall_acc(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("overall_accuracy", 0.0))
    except Exception as e:
        print(f"[Warn] Failed to load {path}: {e}")
        return None


def main() -> None:
    if not os.path.isdir(RESULT_DIR):
        print(f"[Error] Result directory not found: {RESULT_DIR}")
        return

    triplets: Dict[str, TripletMetrics] = {}

    for fname in os.listdir(RESULT_DIR):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        family, model_suffix = parsed
        path = os.path.join(RESULT_DIR, fname)
        acc = load_overall_acc(path)
        if acc is None:
            continue

        if model_suffix not in triplets:
            triplets[model_suffix] = TripletMetrics(model_suffix=model_suffix)
        tm = triplets[model_suffix]

        if family == "baseline":
            tm.baseline_acc = acc
        elif family == "exp4":
            tm.exp4_acc = acc
        elif family == "exp5":
            tm.exp5_acc = acc

    if not triplets:
        print(f"[Info] No valid MMLU result JSONs found under: {RESULT_DIR}")
        return

    # 输出为 CSV
    csv_path = os.path.join(EXP14_DIR, "mmlu_triplet_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model_suffix,baseline_acc,exp4_acc,exp5_acc\n")
        for suffix in sorted(triplets.keys()):
            tm = triplets[suffix]
            b = "" if tm.baseline_acc is None else f"{tm.baseline_acc:.4f}"
            e4 = "" if tm.exp4_acc is None else f"{tm.exp4_acc:.4f}"
            e5 = "" if tm.exp5_acc is None else f"{tm.exp5_acc:.4f}"
            f.write(f"{suffix},{b},{e4},{e5}\n")

    # 在终端打印一个简洁表格
    print("MMLU overall accuracy comparison (baseline vs exp4 vs exp5)")
    print("-" * 80)
    header = f"{'Model (suffix)':40s} | {'Baseline':10s} | {'Exp4':10s} | {'Exp5':10s}"
    print(header)
    print("-" * 80)
    for suffix in sorted(triplets.keys()):
        tm = triplets[suffix]
        b = "-" if tm.baseline_acc is None else f"{tm.baseline_acc:.4f}"
        e4 = "-" if tm.exp4_acc is None else f"{tm.exp4_acc:.4f}"
        e5 = "-" if tm.exp5_acc is None else f"{tm.exp5_acc:.4f}"
        print(f"{suffix:40s} | {b:10s} | {e4:10s} | {e5:10s}")
    print("-" * 80)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

