#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 Resume 数据集上进行微调，通过构建成对的事实（fact）与反事实（counterfactual，翻转种族）文本，
直接在文本上进行继续预训练（Causal Language Modeling）。

训练逻辑：
- 使用 HuggingFace Transformers 加载 Causal LM。
- 采用 LoRA (PEFT) 进行参数高效微调。
- 对于每个样本，构建原始简历文本（fact）和翻转种族后的文本（counterfactual）。
- 使用与 PKFair 一致的 Resume Yes/No prompt 和标准交叉熵损失（Next Token Prediction）。
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from sampling import sample_resume_data_by_race, load_samples_by_csv_indices  # type: ignore
from util import create_counterfactual_by_race  # type: ignore
from model_adapter import get_model_adapter  # type: ignore
from prompt import (  # type: ignore
    add_yes_no_instruction,
    build_resume_prompt,
    format_prompt_for_model,
    resolve_model_type,
)

def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保可重复性。"""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def register_scattermoe_device_hooks(model: nn.Module) -> Tuple[int, int]:
    """Make JetMoE's ScatterMoE kernels and aux-loss reduction model-parallel safe."""
    expert_count = 0
    block_count = 0
    reduction_device = model.get_input_embeddings().weight.device

    def set_expert_device(_module: nn.Module, inputs: Tuple[object, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]) and inputs[0].is_cuda:
            torch.cuda.set_device(inputs[0].device)

    def gather_aux_loss(
        _module: nn.Module,
        _inputs: Tuple[object, ...],
        output: Tuple[object, ...],
    ) -> Tuple[object, ...]:
        if output and torch.is_tensor(output[-1]) and output[-1].device != reduction_device:
            return output[:-1] + (output[-1].to(reduction_device),)
        return output

    for module in model.modules():
        if module.__class__.__name__ == "ParallelExperts":
            if not getattr(module, "_pfairft_device_hook_registered", False):
                module.register_forward_pre_hook(set_expert_device)
                module._pfairft_device_hook_registered = True
                expert_count += 1
        elif module.__class__.__name__ == "JetMoEBlock":
            if not getattr(module, "_pfairft_aux_hook_registered", False):
                module.register_forward_hook(gather_aux_loss)
                module._pfairft_aux_hook_registered = True
                block_count += 1
    return expert_count, block_count



def build_fact_and_counterfactual_dataset(
    dataset_json_path: str,
    max_samples: int,
    balanced: bool,
    random_sampling: bool,
    seed: int,
    sample_csv_path: Optional[str] = None,
    sample_size: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    参考 exp2/analyze_race_sensitive_heads.py：
    - 从 Resume JSON 中采样（支持 CSV 驱动或常规采样）
    - 构建 fact_data（原始 query）
    - 构建 cf_data（翻转种族后的反事实 query）
    """
    print(f"Loading dataset from {dataset_json_path} ...")
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Resume dataset should be a list of records.")

    print(f"Total records in JSON: {len(dataset)}")

    # 采样数据：优先使用 CSV 驱动，否则使用常规采样
    if sample_csv_path:
        print(f"Sampling by CSV order: {sample_csv_path}")
        sampled_data, _, _ = load_samples_by_csv_indices(
            dataset=dataset,
            csv_path=sample_csv_path,
            sample_size=sample_size,
        )
        print(f"Sampled {len(sampled_data)} records from CSV (sample_size={sample_size}).")
    else:
        sampled_data = sample_resume_data_by_race(
            data_records=dataset,
            max_samples=max_samples,
            balanced=balanced,
            random_sampling=random_sampling,
            seed=seed,
        )
        print(f"Sampled {len(sampled_data)} records for fine-tuning.")

    fact_data: List[Dict] = []
    for item in sampled_data:
        summary = item.get("summary", "")
        race = item.get("race", "")



        query = summary
        fact_item = {
            "query": query,
            "summary": summary,
            "race": race,
            "ID": item.get("ID", 0),
        }
        fact_data.append(fact_item)

    print(f"Fact samples after filtering (have summary ): {len(fact_data)}")

    print("Creating counterfactual data (flipping race)...")
    cf_data: List[Dict] = []
    for fact_item in fact_data:
        cf_item = create_counterfactual_by_race(fact_item)
        cf_data.append(cf_item)

    min_len = min(len(fact_data), len(cf_data))
    fact_data = fact_data[:min_len]
    cf_data = cf_data[:min_len]

    print(f"Final paired samples (fact + cf): {min_len}")
    return fact_data, cf_data


class PairedDataset(Dataset):
    """事实与反事实配对数据集。

    每个元素返回一个 dict，包含：
    - fact_text: 原始样本的纯文本（这里直接使用 summary）
    - cf_text:   翻转种族后的反事实纯文本
    """

    def __init__(self, fact_data: List[Dict], cf_data: List[Dict]):
        assert len(fact_data) == len(
            cf_data
        ), "Fact and counterfactual data must have the same length"
        self.fact_data = fact_data
        self.cf_data = cf_data

    def __len__(self) -> int:
        return len(self.fact_data)

    def __getitem__(self, idx: int) -> Dict:
        fact_item = self.fact_data[idx]
        cf_item = self.cf_data[idx]
        return {
            "index": idx,
            "fact_text": fact_item.get("query", ""),
            "cf_text": cf_item.get("query", ""),
        }


def build_lm_features(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def compute_chunked_causal_lm_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_chunk_size: int = 32,
) -> torch.Tensor:
    """Compute full-prompt causal LM CE without upcasting all logits at once."""
    shift_labels = labels[..., 1:]
    valid_tokens = (shift_labels != -100).sum().clamp_min(1)
    loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)

    for start in range(0, shift_labels.size(1), token_chunk_size):
        end = min(start + token_chunk_size, shift_labels.size(1))
        chunk_logits = logits[:, start:end, :].reshape(-1, logits.size(-1)).float()
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        loss_sum = loss_sum + F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )

    return loss_sum / valid_tokens


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    model_type: str,
    resume_prompt_mode: str,
) -> Dict[str, float]:
    """Train one epoch with CE over aligned fact/counterfactual Yes/No prompts."""
    model.train()
    total_loss = 0.0
    num_micro_batches = 0
    num_optimizer_steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        raw_fact_texts = batch["fact_text"]
        raw_cf_texts = batch["cf_text"]

        combined_texts: List[str] = []
        for text in list(raw_fact_texts) + list(raw_cf_texts):
            base_prompt = build_resume_prompt(text, mode=resume_prompt_mode)
            instruction_prompt = add_yes_no_instruction(base_prompt)
            combined_texts.append(format_prompt_for_model(instruction_prompt, model_type))

        features = build_lm_features(
            tokenizer=tokenizer,
            texts=combined_texts,
            max_length=max_length,
            device=device,
        )

        outputs = model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
        )
        loss = compute_chunked_causal_lm_ce(outputs.logits, features["labels"])
        loss = loss / max(1, gradient_accumulation_steps)
        loss.backward()

        total_loss += loss.item() * max(1, gradient_accumulation_steps)
        num_micro_batches += 1

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            num_optimizer_steps += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item() * max(1, gradient_accumulation_steps):.4f}",
                "avg_loss": f"{total_loss / max(num_micro_batches, 1):.4f}",
            }
        )

    avg_loss = total_loss / max(num_micro_batches, 1)
    return {
        "loss": avg_loss,
        "num_micro_batches": num_micro_batches,
        "num_optimizer_steps": num_optimizer_steps,
    }


def save_json(data: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="KL-constrained fine-tuning on Resume fact + counterfactual data (supports LoRA)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path (HuggingFace / ModelScope).",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/data/resume/qwen_summaries_with_race.json",
        help="Resume dataset JSON path.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llama", "qwen", "deepseek", "olmoe", "jetmoe"],
        help="Model family used to choose architecture-specific LoRA target modules.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/common1/hwluo/project/pFairFT/exp5/finetune_output",
        help="Directory to save LoRA fine-tuned model and intermediate files.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of original records to sample before building fact/cf pairs.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced sampling by race (via sample_resume_data_by_race).",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Disable balanced sampling.",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        default=False,
        help="Use random sampling instead of sequential sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--sample_csv_path",
        type=str,
        default="",
        help="If set, sample by following the order of the CSV's `index` column (top rows first). "
        "This overrides --max_samples/--balanced/--random_sampling.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="When --sample_csv_path is set, take the first N rows' indices from the CSV. "
        "If <= 0, use all indices in the CSV.",
    )
    parser.add_argument(
        "--resume_prompt_mode",
        type=str,
        default="summary_only",
        choices=["summary_only", "category", "no_job_description"],
        help="Resume prompt body before the strict Yes/No instruction.",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Training type: 'lora' (default) or 'full' to fine-tune all parameters.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 precision.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 构建 fact + counterfactual 配对数据
    fact_data, cf_data = build_fact_and_counterfactual_dataset(
        dataset_json_path=args.dataset_json_path,
        max_samples=args.max_samples,
        balanced=args.balanced,
        random_sampling=args.random_sampling,
        seed=args.seed,
        sample_csv_path=args.sample_csv_path if args.sample_csv_path else None,
        sample_size=args.sample_size,
    )

    dataset = PairedDataset(fact_data, cf_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # 2. 自动选择设备和精度，支持多GPU训练
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
        print(f"Detected {num_gpus} GPUs. Will use device_map='auto' for model loading.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        use_cuda = torch.cuda.is_available()
        use_bf16_supported = use_cuda and torch.cuda.is_bf16_supported()
        if use_bf16_supported:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if use_cuda else torch.float32

    # 3. 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load directly onto the visible GPU(s); a later whole-model .to() is very
    # slow for sparse expert models whose parameters are backed by many shards.
    device_map = "auto" if torch.cuda.is_available() else None
    max_memory = None
    if device_map is not None and args.model_type.lower() == "jetmoe" and num_gpus > 1:
        # JetMoE's BF16 weights fit on one 24 GiB card, so the default auto
        # map does not shard it and leaves too little room for long samples.
        # A 9 GiB placement budget produces a 12/12 whole-block split on two
        # cards; JetMoEPreTrainedModel already protects JetMoEBlock from splits.
        max_memory = {i: "9GiB" for i in range(num_gpus)}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True
    )

    if args.model_type.lower() == "jetmoe" and num_gpus > 1:
        expert_hooks, block_hooks = register_scattermoe_device_hooks(model)
        print(
            f"Registered model-parallel hooks for {expert_hooks} ScatterMoE experts "
            f"and {block_hooks} JetMoE blocks."
        )

    # 如果使用 device_map，模型已经分布在多GPU上，不需要手动 to(device)
    if device_map is None:
        model.to(device)

    model_type = resolve_model_type(
        requested=args.model_type,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
    )
    print(f"Resolved model_type: {model_type}")

    # 根据 train_type 决定是否启用 LoRA
    if args.train_type == "lora":
        adapter = get_model_adapter(model, model_type=model_type, model_path=args.model_path)
        target_modules = adapter.lora_target_modules()
        print(f"Using LoRA target modules from model adapter: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        print("Using LoRA adapter. Trainable parameters:")
        model.print_trainable_parameters()
    elif args.train_type == "full":
        print("Training all model parameters (full fine-tuning).")
    else:
        raise ValueError(f"Unsupported train_type: {args.train_type}")

    # 如果使用 device_map，模型已经分布在多GPU上，不需要手动 to(device)
    # 但需要确保输入tensor在正确的设备上
    if device_map is None:
        model.to(device)
    else:
        # 使用 device_map 时，需要确定输入设备（通常是第一个GPU或embedding层所在设备）
        if hasattr(model, "get_input_embeddings"):
            try:
                embed_layer = model.get_input_embeddings()
                if hasattr(embed_layer, "weight"):
                    device = embed_layer.weight.device
                else:
                    device = next(model.parameters()).device
            except:
                device = next(model.parameters()).device
        else:
            device = next(model.parameters()).device
        print(f"Using device_map='auto'. Input tensors will be on device: {device}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # 4. 优化器与学习率调度器
    steps_per_epoch = math.ceil(
        len(dataloader) / max(1, args.gradient_accumulation_steps)
    )
    total_steps = max(1, steps_per_epoch * args.num_epochs)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # 5. 训练循环（KL 约束 fact / counterfactual）
    print("=" * 80)
    print("Starting KL-based training on fact/counterfactual pairs...")
    print("=" * 80)

    train_start_time = time.time()
    train_start_iso = datetime.now().isoformat()
    print(f"Training started at: {train_start_iso}")

    training_history: List[Dict] = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            tokenizer=tokenizer,
            max_length=args.max_length,
            model_type=model_type,
            resume_prompt_mode=args.resume_prompt_mode,
        )
        training_history.append({"epoch": epoch + 1, **epoch_metrics})
        print(f"  Loss: {epoch_metrics['loss']:.6f}")

    # 6. 保存最终模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # 7. 记录训练时间与配置
    train_end_time = time.time()
    train_end_iso = datetime.now().isoformat()
    train_duration = train_end_time - train_start_time

    timing_info = {
        "training_start_time": train_start_iso,
        "training_end_time": train_end_iso,
        "training_duration_seconds": train_duration,
        "training_duration_minutes": train_duration / 60.0,
        "training_duration_hours": train_duration / 3600.0,
        "total_train_samples": len(dataset),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "training_history": training_history,
        "config": {
            "model_path": args.model_path,
            "max_samples": args.max_samples,
            "balanced": args.balanced,
            "random_sampling": args.random_sampling,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "model_type": model_type,
            "resume_prompt_mode": args.resume_prompt_mode,
            "lora_target_modules": target_modules if args.train_type == "lora" else [],
        },
    }

    timing_json_path = os.path.join(args.output_dir, "training_timing.json")
    save_json(timing_info, timing_json_path)

    print("=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Final model saved to: {final_model_dir}")
    print(
        f"Total duration: {train_duration:.2f} seconds "
        f"({train_duration/60:.2f} minutes, {train_duration/3600:.2f} hours)"
    )
    print(f"Timing information saved to: {timing_json_path}")


if __name__ == "__main__":
    main()
