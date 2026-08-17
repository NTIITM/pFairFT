#!/usr/bin/env python3
"""Build the nine-model Resume fairness/MMLU CE paper table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


MODEL_NAMES = (
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "DeepSeek-V2-Lite-Chat",
    "JetMoE-8B-Chat",
    "OLMoE-1B-7B-0924-Instruct",
)

DISPLAY_NAMES = {
    "Llama-3.2-1B-Instruct": "Llama-1B",
    "Llama-3.2-3B-Instruct": "Llama-3B",
    "Meta-Llama-3-8B-Instruct": "Llama-8B",
    "Qwen3-1.7B": "Qwen-1.7B",
    "Qwen3-4B": "Qwen-4B",
    "Qwen3-8B": "Qwen-8B",
    "DeepSeek-V2-Lite-Chat": "DeepSeek-V2-Lite",
    "JetMoE-8B-Chat": "JetMoE-8B",
    "OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B",
}

MODEL_CONFIGS = {
    "Llama-3.2-1B-Instruct": "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-1B-Instruct/config.json",
    "Llama-3.2-3B-Instruct": "/mnt/nfs/huggingface/LLM-Research/Llama-3.2-3B-Instruct/config.json",
    "Meta-Llama-3-8B-Instruct": "/mnt/nfs/huggingface/LLM-Research/Meta-Llama-3-8B-Instruct/config.json",
    "Qwen3-1.7B": "/mnt/nfs/huggingface/Qwen/Qwen3-1.7B/config.json",
    "Qwen3-4B": "/mnt/nfs/huggingface/Qwen/Qwen3-4B/config.json",
    "Qwen3-8B": "/mnt/nfs/huggingface/Qwen/Qwen3-8B/config.json",
    "DeepSeek-V2-Lite-Chat": "/mnt/nfs/huggingface/deepseek-ai/DeepSeek-V2-Lite-Chat/config.json",
    "JetMoE-8B-Chat": "/mnt/nfs/huggingface/jetmoe/jetmoe-8b-chat/config.json",
    "OLMoE-1B-7B-0924-Instruct": "/mnt/nfs/huggingface/allenai/OLMoE-1B-7B-0924-Instruct/config.json",
}

METHODS = (
    ("base", "Base"),
    ("debiased_prompt", "Debiased Prompt"),
    ("global", "Global"),
    ("inference_time", "Inference Time"),
    ("pfairft_kl", "PFairFT-KL"),
    ("pfairft", "PFairFT"),
)


def load_resume_gap(path: Path, expected_rows: int = 100) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    values = []
    seen = set()
    for row in rows:
        index = int(row["index"])
        if index in seen:
            raise ValueError(f"{path}: duplicate index {index}")
        seen.add(index)
        fact = float(row["fact_p_yes"])
        counterfactual = float(row["cf_p_yes"])
        if not (math.isfinite(fact) and math.isfinite(counterfactual)):
            raise ValueError(f"{path}: non-finite probability at index {index}")
        if not (0.0 <= fact <= 1.0 and 0.0 <= counterfactual <= 1.0):
            raise ValueError(f"{path}: probability outside [0, 1] at index {index}")
        values.append(abs(fact - counterfactual))
    return sum(values) / len(values)


def load_mmlu_ce(path: Path, expected_count: int = 1531) -> float:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    count = int(payload.get("count", payload.get("total", -1)))
    if count != expected_count:
        raise ValueError(f"{path}: expected count={expected_count}, found {count}")
    ce = float(payload["ce"])
    if not math.isfinite(ce):
        raise ValueError(f"{path}: CE is not finite")
    return ce


def _layer_from_key(key: str) -> int:
    parts = key.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Cannot resolve transformer layer from adapter key {key!r}") from error


def theoretical_lora_counts(
    adapter_path: Path,
    selected_heads_path: Path,
    num_attention_heads: int,
) -> tuple[int, float]:
    """Return full-adapter and selected-logical-head LoRA element counts."""
    layer_params: dict[int, int] = defaultdict(int)
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            layer_params[_layer_from_key(key)] += math.prod(shape)
    if not layer_params:
        raise ValueError(f"{adapter_path}: no LoRA tensors")

    with selected_heads_path.open(encoding="utf-8") as handle:
        selected = json.load(handle)
    selected_per_layer: dict[int, int] = defaultdict(int)
    for item in selected:
        selected_per_layer[int(item["layer"])] += 1

    precise = 0.0
    for layer, count in selected_per_layer.items():
        if layer not in layer_params:
            raise ValueError(f"{adapter_path}: selected layer {layer} has no LoRA tensors")
        precise += layer_params[layer] * count / num_attention_heads
    return sum(layer_params.values()), precise


def _num_attention_heads(config_path: Path) -> int:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    value = config.get("num_attention_heads", config.get("n_head"))
    if value is None:
        raise ValueError(f"{config_path}: missing attention head count")
    return int(value)


def _source_paths(project_root: Path, output_root: Path, model: str) -> dict:
    result_root = project_root / "results" / model
    table_root = output_root / model
    downstream = result_root / "downstream_evaluation"
    return {
        "resume": {
            "base": downstream / "resume_baseline_top100_summary_only_pkfair_3epoch_fresh.csv",
            "debiased_prompt": table_root / "resume_debiased_prompt.csv",
            "global": table_root / "resume_global.csv",
            "pfairft": downstream / "resume_pkfair_kl_pkfair_3epoch_fresh.csv",
            "pfairft_kl": downstream / "resume_pkfair_pkfair_3epoch_fresh.csv",
            "inference_time": table_root / "resume_inference_time.csv",
        },
        "mmlu": {
            "base": table_root / "mmlu_base.json",
            "debiased_prompt": table_root / "mmlu_base.json",
            "global": table_root / "mmlu_global.json",
            "pfairft": table_root / "mmlu_pfairft.json",
            "pfairft_kl": table_root / "mmlu_pfairft_kl.json",
            "inference_time": table_root / "mmlu_inference_time_teacher_forcing_all.json",
        },
        "heads": result_root
        / "sensitive_heads_moefreeze_top100_summary_only_current_ranking"
        / "selected_heads_elbow.json",
        "global_adapter": table_root / "global_adapter_path.txt",
        "pfairft_adapter": result_root
        / "pkfair_fairness_kl_yesno_summary_only_current_ranking_full_3epoch"
        / "final_model"
        / "adapter_model.safetensors",
    }


def _read_path(path_file: Path) -> Path:
    value = path_file.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{path_file}: empty path")
    return Path(value) / "adapter_model.safetensors"


def collect(project_root: Path, output_root: Path) -> dict:
    result = {
        "protocol": {
            "fairness_dataset": "Resume top-100 from current summary_only ranking",
            "dp": "mean(abs(fact_p_yes - cf_p_yes))",
            "capability_dataset": "cais/mmlu all validation",
            "mmlu_count": 1531,
            "inference_time_ce": "projection applied at every teacher-forcing position",
            "method_mapping": {
                "PFairFT": "fairness_kl",
                "PFairFT-KL": "fairness_kl_ce",
            },
            "checkpoint_policy": "reuse existing checkpoints; no head-mask-fix retraining",
            "parameter_count": "sum per-layer theoretical LoRA elements / logical heads * selected head count",
        },
        "models": {},
    }
    for model in MODEL_NAMES:
        paths = _source_paths(project_root, output_root, model)
        dp = {key: load_resume_gap(path) for key, path in paths["resume"].items()}
        ce = {key: load_mmlu_ce(path) for key, path in paths["mmlu"].items()}
        heads = _num_attention_heads(Path(MODEL_CONFIGS[model]))
        global_count, _ = theoretical_lora_counts(
            _read_path(paths["global_adapter"]), paths["heads"], heads
        )
        _, precise_count = theoretical_lora_counts(
            paths["pfairft_adapter"], paths["heads"], heads
        )
        result["models"][model] = {
            "display_name": DISPLAY_NAMES[model],
            "dp": dp,
            "ce": ce,
            "tuned_params": {
                "base": None,
                "debiased_prompt": None,
                "global": global_count,
                "pfairft": precise_count,
                "pfairft_kl": precise_count,
                "inference_time": None,
            },
            "sources": {
                group: {key: str(path.resolve()) for key, path in values.items()}
                for group, values in (("resume", paths["resume"]), ("mmlu", paths["mmlu"]))
            },
            "selected_head_count": len(json.loads(paths["heads"].read_text(encoding="utf-8"))),
        }
    return result


def _format_params(value: int | float | None) -> str:
    return "-" if value is None else f"{value / 1_000_000:.3f}M"


def write_csv(payload: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", *[label for _, label in METHODS]])
        for model in MODEL_NAMES:
            data = payload["models"][model]
            writer.writerow([data["display_name"], "DP", *[f'{data["dp"][key]:.6f}' for key, _ in METHODS]])
            writer.writerow([data["display_name"], "CE", *[f'{data["ce"][key]:.6f}' for key, _ in METHODS]])
            writer.writerow([data["display_name"], "Tuned Params.", *[_format_params(data["tuned_params"][key]) for key, _ in METHODS]])


def write_latex(payload: dict, path: Path) -> None:
    lines = [
        r"\begin{table*}[h!tbp]",
        r"\centering",
        r"\caption{Fairness performance (DP, lower is better) on the Resume Screening Dataset and capability preservation (MMLU validation CE, lower is better) across dense and mixture-of-experts model families.}",
        r"\label{tab:fine_tuning_resume}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{c r|cccccc}",
        r"\toprule",
        "& & " + " & ".join(rf"\textbf{{{label}}}" for _, label in METHODS) + r"\\",
        r"\midrule",
    ]
    for index, model in enumerate(MODEL_NAMES):
        data = payload["models"][model]
        rows = (
            (r"DP ($\downarrow$)", [f'{data["dp"][key]:.3f}' for key, _ in METHODS]),
            (r"CE ($\downarrow$)", [f'{data["ce"][key]:.3f}' for key, _ in METHODS]),
            ("Tuned Params.", [_format_params(data["tuned_params"][key]) for key, _ in METHODS]),
        )
        for row_index, (metric, values) in enumerate(rows):
            prefix = rf"\multirow{{3}}{{*}}{{{data['display_name']}}}" if row_index == 0 else ""
            lines.append(f"{prefix} & {metric} & " + " & ".join(values) + r"\\")
        if index != len(MODEL_NAMES) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_root", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else project_root / "results" / "table2_nine_models"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = collect(project_root, output_root)
    (output_root / "table2_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(payload, output_root / "table2_metrics.csv")
    write_latex(payload, output_root / "table2_nine_models.tex")
    print(f"Wrote audited table artifacts to {output_root}")


if __name__ == "__main__":
    main()
