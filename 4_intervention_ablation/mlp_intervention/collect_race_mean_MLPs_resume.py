#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect race-specific mean MLP activations (last token) for Resume dataset.

输出 pkl：
{
  "white_emb": {layer(int): np.ndarray[hidden_size]},
  "black_emb": {layer(int): np.ndarray[hidden_size]},
  "num_layers": int,
  "hidden_size": int,
  "total_samples": int,
  "white_count": int,
  "black_count": int,
  "dataset": "resume",
}

注意：这里的 race 分组依据 extract_race_from_query / 数据项的 race 字段。
"""

import argparse
import json
import os
import pickle
import tempfile
import shutil
from typing import Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from util import extract_race_from_query, get_model_config  # type: ignore  # noqa: E402
from sampling import (  # type: ignore  # noqa: E402
    load_samples_by_csv_indices,
    sample_resume_data_by_race,
)
from prompt import format_prompt_for_model, resolve_model_type, add_yes_no_instruction, build_category_prompt# type: ignore  # noqa: E402
from hook import get_last_token_indices_safe, get_mlp_last_token_activation_hook  # type: ignore  # noqa: E402
from cache import MLPDiskCache  # type: ignore  # noqa: E402
from model_adapter import get_model_adapter  # type: ignore  # noqa: E402


class ResumeDataset(Dataset):
    def __init__(self, data_records: List[dict], resume_prompt_mode: str):
        self.data = data_records
        self.resume_prompt_mode = resume_prompt_mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        query = item.get("summary", item.get("query", ""))
        race = item.get("race", "")
        category = item.get("category", "")
        if not race:
            race = extract_race_from_query(query) or "Unknown"
        query = (
            query
            if self.resume_prompt_mode == "summary_only"
            else build_category_prompt(query, category)
        )
        return {
            "index": idx,
            "prompt": add_yes_no_instruction(query),
            "race": race if race else "Unknown",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        default="",
        help="Follow the ranking CSV index order instead of sampling the dataset.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of ranking rows to use; <=0 uses the full ranking.",
    )
    parser.add_argument(
        "--resume_prompt_mode",
        choices=["summary_only", "category"],
        default="summary_only",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
    )
    parser.add_argument("--random_sampling", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
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
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="mlp_means_resume_")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()

        config = get_model_config(model)
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]

        model_type = resolve_model_type(args.model_type, model=model, tokenizer=tokenizer, model_path=args.model_path)
        adapter = get_model_adapter(model, model_type=args.model_type, model_path=args.model_path)
        print(f"Using adapter: {adapter.family} ({adapter.head_activation_kind})")
        device = adapter.get_input_embedding_module().weight.device

        with open(args.dataset_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
            raise ValueError("Dataset should be a list of records.")

        used_indices: List[int]
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

        # Keep only fields we need
        records: List[dict] = []
        for item, original_index in zip(sampled_data, used_indices):
            records.append(
                {
                    "summary": item.get("summary", ""),
                    "category": item.get("category", ""),
                    "race": item.get("race", ""),
                    "original_index": int(original_index),
                }
            )

        races_list: List[str] = []
        for item in records:
            q = item.get("summary", "")
            r = item.get("race", "")
            if not r:
                r = extract_race_from_query(q) or "Unknown"
            races_list.append(r if r else "Unknown")

        white_indices = [i for i, r in enumerate(races_list) if str(r).lower() == "white" or r == "White"]
        black_indices = [i for i, r in enumerate(races_list) if str(r).lower() == "black" or r == "Black"]

        n = len(records)
        cache = MLPDiskCache(n, num_layers, hidden_size, "fact_mlp", temp_dir)

        ds = ResumeDataset(records, args.resume_prompt_mode)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        batch_mlp_buffer: Dict[int, torch.Tensor] = {}
        hooks = []
        for l in range(num_layers):
            hooks.append(adapter.register_mlp_output_hook(l, batch_mlp_buffer))

        for batch in tqdm(dl, desc="Collecting MLP activations (resume)"):
            indices = batch["index"].numpy()
            prompts_formatted = [format_prompt_for_model(p, model_type) for p in batch["prompt"]]
            inputs = tokenizer(
                prompts_formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            last_token_indices = get_last_token_indices_safe(input_ids, attention_mask, tokenizer)
            batch_range = torch.arange(input_ids.shape[0], device=device)

            batch_mlp_buffer.clear()
            with torch.no_grad():
                _ = model(**inputs)

            for l in range(num_layers):
                if l not in batch_mlp_buffer:
                    continue
                act = batch_mlp_buffer[l]  # [B, Seq, Hidden]
                act_device = act.device
                batch_range_on_device = batch_range.to(act_device)
                last_token_indices_on_device = last_token_indices.to(act_device)
                last_act = act[batch_range_on_device, last_token_indices_on_device, :]  # [B, Hidden]
                cache.save_batch(indices, l, last_act.cpu().float().numpy())
                del batch_mlp_buffer[l]

        for h in hooks:
            h.remove()

        white_emb: Dict[int, np.ndarray] = {}
        black_emb: Dict[int, np.ndarray] = {}

        for l in range(num_layers):
            # MLPDiskCache.data shape: [N, L, Hidden]
            layer_act = cache.data[:, l, :]  # [Num_Samples, Hidden_Size]
            if white_indices:
                white_emb[l] = np.mean(layer_act[white_indices], axis=0)
            else:
                white_emb[l] = np.zeros(hidden_size, dtype=np.float32)
            if black_indices:
                black_emb[l] = np.mean(layer_act[black_indices], axis=0)
            else:
                black_emb[l] = np.zeros(hidden_size, dtype=np.float32)

        out = {
            "white_emb": white_emb,
            "black_emb": black_emb,
            "num_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "total_samples": int(n),
            "white_count": int(len(white_indices)),
            "black_count": int(len(black_indices)),
            "dataset": "resume",
            "model": args.model_path,
            "adapter_family": adapter.family,
            "head_activation_kind": adapter.head_activation_kind,
            "mlp_surface": (
                "routed_moe_block_output"
                if adapter.family in {"deepseek", "olmoe", "jetmoe", "qwen_moe"}
                else "dense_mlp_block_output"
            ),
            "dataset_json_path": args.dataset_json_path,
            "sample_csv_path": args.sample_csv_path or None,
            "sample_size": args.sample_size if args.sample_csv_path else len(records),
            "sample_indices": used_indices,
            "resume_prompt_mode": args.resume_prompt_mode,
        }

        with open(args.output_path, "wb") as f:
            pickle.dump(out, f)

        print("Saved:", args.output_path)
        print(f"White: {len(white_indices)}, Black: {len(black_indices)}, Total: {n}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
