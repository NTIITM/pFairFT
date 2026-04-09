#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用的 MMLU 评估脚本。

功能：
- 加载一个 HF Causal LM（基座模型或已微调模型，如 exp5/exp4 产出的 final_model）
- 在 MMLU 上做选择题评估，计算整体 accuracy 以及各子任务 accuracy

依赖：
- transformers
- datasets

用法示例：

  python evaluate_mmlu.py \
    --model_path "/home/common1/hwluo/project/pFairFT/exp5/lora_Llama-3.2-1B-Instruct_top100/final_model" \
    --output_json "mmlu_Llama-3.2-1B-Instruct_top100.json"
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LETTERS = ["A", "B", "C", "D"]


@dataclass
class MMLUSample:
    question: str
    choices: List[str]
    answer: int  # index in [0, 1, 2, 3]
    subject: str


def build_mmlu_prompt(
    question: str,
    choices: List[str],
) -> str:
    """
    构造一个简单的英文 MMLU prompt：多选单选题，要求输出 A/B/C/D。
    """
    choices_block = "\n".join(
        f"{letter}. {text}" for letter, text in zip(CHOICE_LETTERS, choices)
    )
    prompt = (
        "You are a knowledgeable AI assistant. "
        "Please answer the following multiple-choice question by choosing one option.\n\n"
        f"Question: {question}\n"
        f"Options:\n{choices_block}\n\n"
        "Answer with the single letter of the correct option (A, B, C, or D)."
    )
    return prompt


def format_prompt_for_model(model_type: str, user_prompt: str) -> str:
    """
    为 Llama / Qwen 添加简单的对话包装，以便与现有 chat 格式兼容。
    这里不依赖项目根目录下的 prompt.py，避免循环依赖。
    """
    mt = (model_type or "llama").lower()
    if "qwen" in mt:
        return (
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant"
            f"<think>\n</think>\n\nAnswer: "
        )
    # 默认按 Llama3 格式处理
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    )


def infer_model_type(model_name_or_path: str) -> str:
    n = model_name_or_path.lower()
    if "qwen" in n:
        return "qwen"
    if "llama" in n:
        return "llama"
    return "llama"


def load_mmlu(split: str = "validation") -> List[MMLUSample]:
    """
    加载 MMLU 数据集（cais/mmlu），返回统一格式样本列表。
    默认使用 validation（dev）做评估；如果需要也可以切换到 test。
    """
    ds = load_dataset("cais/mmlu", "all", split=split)
    samples: List[MMLUSample] = []
    for row in ds:
        question = row["question"]
        choices = row["choices"]
        answer_letter = row["answer"]
        if isinstance(answer_letter, str):
            answer_idx = CHOICE_LETTERS.index(answer_letter)
        else:
            answer_idx = int(answer_letter)
        subject = row.get("subject", "unknown")
        samples.append(
            MMLUSample(
                question=question,
                choices=choices,
                answer=answer_idx,
                subject=subject,
            )
        )
    return samples


def get_choice_logit(
    model,
    tokenizer,
    prompt: str,
    choice_letter: str,
    device: torch.device,
    max_new_tokens: int = 2,
) -> float:
    """
    计算在给定 prompt 下，模型生成首个 token 为指定字母（A/B/C/D）的 logit。
    使用 greedy 解码的第一步 logits。

    当前版本中我们统一将模型加载到单一 device（例如 cuda:0 或 cpu），
    因此这里安全地将输入 tensor 也移动到该 device 上。
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]
    first_step_logits = logits[:, -1, :]  # [B, V]
    token_ids = tokenizer(choice_letter, add_special_tokens=False).input_ids
    # 只取第一个 token 对应的 logit
    if len(token_ids) == 0:
        raise ValueError(f"Tokenization for choice {choice_letter!r} returned empty ids.")
    token_id = token_ids[0]
    logit = first_step_logits[0, token_id].item()
    return logit


def evaluate_mmlu(
    model,
    tokenizer,
    samples: List[MMLUSample],
    device: torch.device,
    model_type: str,
    max_samples: int = -1,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float], int]:
    """
    在给定样本列表上进行 MMLU 评估。

    返回：
    - overall_accuracy
    - per_subject_accuracy: {subject: acc}
    - total_evaluated_samples
    """
    if max_samples > 0:
        samples = samples[:max_samples]

    correct = 0
    total = 0
    per_subject_stats: Dict[str, Dict[str, int]] = {}

    model.eval()

    for idx, sample in enumerate(samples):
        user_prompt = build_mmlu_prompt(sample.question, sample.choices)
        full_prompt = format_prompt_for_model(model_type, user_prompt)

        # 计算每个选项的 logit
        choice_logits = []
        for letter in CHOICE_LETTERS:
            logit = get_choice_logit(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                choice_letter=letter,
                device=device,
            )
            choice_logits.append(logit)

        pred_idx = int(torch.tensor(choice_logits).argmax().item())
        gold_idx = sample.answer

        is_correct = pred_idx == gold_idx
        if is_correct:
            correct += 1
        total += 1

        subj = sample.subject
        if subj not in per_subject_stats:
            per_subject_stats[subj] = {"correct": 0, "total": 0}
        if is_correct:
            per_subject_stats[subj]["correct"] += 1
        per_subject_stats[subj]["total"] += 1

        if verbose and (idx + 1) % 50 == 0:
            acc_so_far = correct / total if total > 0 else 0.0
            print(f"[Progress] {idx + 1}/{len(samples)} samples, acc={acc_so_far:.4f}")

    overall_acc = correct / total if total > 0 else 0.0
    per_subject_acc: Dict[str, float] = {}
    for subj, st in per_subject_stats.items():
        if st["total"] > 0:
            per_subject_acc[subj] = st["correct"] / st["total"]

    return overall_acc, per_subject_acc, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Causal LM on MMLU (multiple-choice).")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (e.g., exp5/..../final_model).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation (cuda or cpu).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max number of MMLU samples to evaluate; -1 means all.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which split of MMLU to use.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="mmlu_eval_results.json",
        help="Where to save evaluation results JSON.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print(f"Loading model from {args.model_path}")
    print("=" * 80)

    # 如果有多张 GPU，则使用 HuggingFace 的 device_map="auto" 进行多卡切分，
    # 以减小单卡显存占用，降低 OOM 风险。
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(
            f"Detected {torch.cuda.device_count()} CUDA devices, "
            "loading model with device_map='auto' for multi-GPU."
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        # 此时不再调用 model.to(device)，由 HF 根据 device_map 负责分布到多卡。
        # 下面的推理代码仍然将输入 tensor 放到主 device（例如 cuda:0），
        # HF 会在内部完成跨设备调度。
    else:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype, trust_remote_code=True
        )
        # 单卡或 CPU 的情况，依然保持原来的行为，将整个模型移动到指定 device。
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_type = infer_model_type(args.model_path)
    print(f"Inferred model_type: {model_type}")

    print("=" * 80)
    print(f"Loading MMLU ({args.split}) split...")
    print("=" * 80)
    samples = load_mmlu(split=args.split)
    print(f"Loaded {len(samples)} MMLU samples.")

    print("=" * 80)
    print("Running MMLU evaluation...")
    print("=" * 80)
    overall_acc, per_subject_acc, total_evaluated = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        model_type=model_type,
        max_samples=args.max_samples,
        verbose=True,
    )

    print("=" * 80)
    print(f"Overall MMLU accuracy: {overall_acc:.4f} (n={total_evaluated})")
    print("=" * 80)

    result = {
        "model_path": args.model_path,
        "split": args.split,
        "max_samples": args.max_samples,
        "total_evaluated": total_evaluated,
        "overall_accuracy": overall_acc,
        "per_subject_accuracy": per_subject_acc,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()

