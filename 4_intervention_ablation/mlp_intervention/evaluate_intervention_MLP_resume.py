#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate paired Resume fairness under MLP mean ablation.

功能：
- 使用在 Resume 样本上预先收集的 MLP 均值（white_emb/black_emb），统一 mean ablation：
  mean = (white_mean + black_mean) / 2
- 对 Resume top-k 样本构造 fact/counterfactual，并分别记录干预下的 p(yes)
- 输出与标准 Resume 评估兼容的 paired CSV

均值文件必须来自 collect_race_mean_MLPs_resume.py。
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入上层目录工具
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from probability import (  # type: ignore  # noqa: E402
    YES_CANDIDATES,
    NO_CANDIDATES,
    get_target_token_ids,
)
from prompt import (  # type: ignore  # noqa: E402
    add_yes_no_instruction,
    build_resume_prompt,
    format_prompt_for_model,
    resolve_model_type,
)
from util import (  # type: ignore  # noqa: E402
    compute_p_yes_from_logits_with_warning,
    get_input_device,
    get_model_config,
    extract_race_from_query,
    create_counterfactual_by_race,
)
from sampling import (  # type: ignore  # noqa: E402
    load_samples_by_csv_indices,
    sample_resume_data_by_race,
)
from hook import (  # type: ignore  # noqa: E402
    get_last_token_indices_safe,
    remove_intervention_hooks,
)
from model_adapter import get_model_adapter  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate p(yes) on Resume dataset under MLP-level negative intervention (mean ablation)."
        )
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="intervention_mlp_resume_results",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to write per-sample intervention results CSV.",
    )
    parser.add_argument(
        "--append_csv",
        action="store_true",
        help="Append to --csv_path instead of replacing it.",
    )
    parser.add_argument(
        "--sensitive_mlp_path",
        type=str,
        required=True,
        help="Path to selected_mlp_layers_elbow.json.",
    )
    parser.add_argument(
        "--mlp_embeddings_path",
        type=str,
        required=True,
        help="Path to Resume MLP mean embeddings pkl (from collect_race_mean_MLPs_resume.py).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
    )
    parser.add_argument("--sample_csv_path", type=str, default="")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument(
        "--resume_prompt_mode",
        choices=["summary_only", "category"],
        default="summary_only",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--intervention_type",
        type=str,
        default="mlp_negative",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load dataset (Resume)
    with open(args.dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of records.")

    if args.sample_csv_path:
        sampled_data, used_indices, _ = load_samples_by_csv_indices(
            dataset=dataset,
            csv_path=args.sample_csv_path,
            sample_size=args.sample_size,
        )
    else:
        sampled_data = sample_resume_data_by_race(
            data_records=dataset,
            max_samples=args.max_samples,
            balanced=args.balanced,
            random_sampling=args.random_sampling,
            seed=args.seed,
        )
        index_by_identity = {id(item): idx for idx, item in enumerate(dataset)}
        used_indices = [index_by_identity[id(item)] for item in sampled_data]

    fact_data: List[dict] = []
    for original_index, item in zip(used_indices, sampled_data):
        summary = item.get("summary", "")
        category = item.get("category", "")
        race = item.get("race", "")
        query = summary
        fact_data.append(
            {
                "id": int(original_index),
                "query": query,
                "summary": summary,
                "category": category,
                "race": race,
            }
        )

    # 2. Load model & tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto"
        if args.device == "cuda" and torch.cuda.is_available()
        else None,
        torch_dtype=(
            torch.float16
            if args.device == "cuda" and torch.cuda.is_available()
            else torch.float32
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_device = get_input_device(model, args.device)
    model_type = resolve_model_type(
        args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Using model_type: {model_type}")
    adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
    print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")

    # 3. Token IDs
    yes_ids = get_target_token_ids(tokenizer, YES_CANDIDATES)
    no_ids = get_target_token_ids(tokenizer, NO_CANDIDATES)
    if not yes_ids or not no_ids:
        raise ValueError("Could not find valid token IDs for yes/no candidates.")

    # 4. Load sensitive MLP layers & Resume means
    if not os.path.exists(args.sensitive_mlp_path):
        raise FileNotFoundError(f"sensitive_mlp_path not found: {args.sensitive_mlp_path}")
    if not os.path.exists(args.mlp_embeddings_path):
        raise FileNotFoundError(f"mlp_embeddings_path not found: {args.mlp_embeddings_path}")

    with open(args.sensitive_mlp_path, "r", encoding="utf-8") as f:
        selected_layers_data = f.read()
    import json as _json
    selected_layers_list = _json.loads(selected_layers_data)
    sensitive_layers = [int(d["layer"]) for d in selected_layers_list]

    with open(args.mlp_embeddings_path, "rb") as f:
        emb_data = pickle.load(f)

    white_embeddings = emb_data.get("white_emb", {})
    black_embeddings = emb_data.get("black_emb", {})

    white_emb: Dict[int, np.ndarray] = {int(k): v for k, v in white_embeddings.items()}
    black_emb: Dict[int, np.ndarray] = {int(k): v for k, v in black_embeddings.items()}

    config = get_model_config(model)
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    print(f"Model config: layers={num_layers}, hidden_size={hidden_size}")

    valid_layers: List[int] = []
    for l in sensitive_layers:
        if l in white_emb and l in black_emb and 0 <= l < num_layers:
            valid_layers.append(l)
    sensitive_layers = valid_layers
    if not sensitive_layers:
        raise ValueError("No valid sensitive MLP layers with Resume means.")

    # 5. Prepare paired factual/counterfactual prompts.
    prompt_pairs: List[Dict[str, str]] = []
    for item in fact_data:
        base_query = build_resume_prompt(
            summary=item.get("summary", ""),
            category=item.get("category", ""),
            mode=args.resume_prompt_mode,
        )
        fact_race = extract_race_from_query(base_query) or item.get("race", "") or "Unknown"
        paired_fact = {
            "query": base_query,
            "summary": item.get("summary", ""),
            "category": item.get("category", ""),
            "race": fact_race,
        }
        paired_cf = create_counterfactual_by_race(paired_fact)
        prompt_pairs.append(
            {
                "fact": add_yes_no_instruction(paired_fact["query"]),
                "cf": add_yes_no_instruction(paired_cf["query"]),
                "fact_race": fact_race,
                "cf_race": paired_cf.get("race", ""),
            }
        )

    # 6. Forward both sides with the same intervention.
    def evaluate_prompt(prompt: str, sample_idx: int) -> float:
        formatted_prompt = format_prompt_for_model(prompt, model_type)
        input_ids = tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(input_device)
        attention_mask = torch.ones_like(input_ids).to(input_device)

        last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
        output_pos = int(last_token_indices[0].item())

        prompt_hooks = []
        for l in sensitive_layers:
            mean_emb_np = (white_emb[l] + black_emb[l]) / 2.0
            mean_emb = (
                torch.from_numpy(mean_emb_np).float()
                if isinstance(mean_emb_np, np.ndarray)
                else mean_emb_np
            )
            hook = adapter.register_mlp_mean_replacement_hook(
                layer_idx=l,
                mean_embedding=mean_emb,
                output_pos=output_pos,
            )
            prompt_hooks.append(hook)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_row = outputs.logits[0, output_pos, :].float()

                p_yes = compute_p_yes_from_logits_with_warning(
                    logits_row=logits_row,
                    tokenizer=tokenizer,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    sample_idx=sample_idx,
                    show_warnings=True,
                    prefix="Intervention-MLP-Negative-Resume",
                )
                result = float(p_yes)
        finally:
            remove_intervention_hooks(prompt_hooks)

        del input_ids, outputs, logits_row
        return result

    fact_p_yes_results: List[float] = []
    cf_p_yes_results: List[float] = []
    for idx, pair in enumerate(tqdm(prompt_pairs, desc="Resume paired MLP intervention")):
        fact_p_yes_results.append(evaluate_prompt(pair["fact"], idx))
        cf_p_yes_results.append(evaluate_prompt(pair["cf"], idx))

    # 7. Save CSV
    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        mode = "a" if args.append_csv else "w"
        write_header = not args.append_csv or not os.path.exists(args.csv_path)
        with open(args.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "index",
                    "fact_p_yes",
                    "cf_p_yes",
                    "fact_race",
                    "cf_race",
                    "intervention_type",
                ])
            for item, pair, fact_p, cf_p in zip(
                fact_data, prompt_pairs, fact_p_yes_results, cf_p_yes_results
            ):
                if math.isnan(fact_p) or math.isnan(cf_p):
                    continue
                writer.writerow([
                    int(item["id"]),
                    fact_p,
                    cf_p,
                    pair["fact_race"],
                    pair["cf_race"],
                    args.intervention_type,
                ])

    metadata = {
        "model_path": args.model_path,
        "model_type": model_type,
        "adapter_family": adapter.family,
        "head_activation_kind": adapter.head_activation_kind,
        "mlp_surface": "routed_moe_block_output",
        "dataset": "resume",
        "evaluation": "paired_fact_counterfactual",
        "dataset_json_path": args.dataset_json_path,
        "sample_csv_path": args.sample_csv_path or None,
        "sample_size": len(fact_data),
        "sample_indices": [int(index) for index in used_indices],
        "resume_prompt_mode": args.resume_prompt_mode,
        "sensitive_mlp_path": args.sensitive_mlp_path,
        "mlp_embeddings_path": args.mlp_embeddings_path,
        "selected_layers": sensitive_layers,
        "intervention_type": args.intervention_type,
        "csv_path": args.csv_path or None,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done. Total fact/counterfactual pairs:", len(fact_data))


if __name__ == "__main__":
    main()
