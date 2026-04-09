import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import os
# Avoid tokenizer parallelism / background threads shutdown issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pandas as pd
import json
import argparse
import pickle
import copy
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

# Import local utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from util import get_model_config, get_input_device
from exp26.cache_utils import PreLogitsCache

def compute_kl_ce_batch(pre_logits, post_logits):
    """
    Compute KL(Post || Pre) and CE(Pre, Post).
    Logits shape: [seq_len, vocab_size]
    """
    log_p_post = F.log_softmax(post_logits, dim=-1)
    log_p_pre = F.log_softmax(pre_logits, dim=-1)
    p_post = F.softmax(post_logits, dim=-1)
    p_pre = F.softmax(pre_logits, dim=-1)

    kl = torch.sum(p_post * (log_p_post - log_p_pre), dim=-1)
    ce = -torch.sum(p_pre * log_p_post, dim=-1)
    return kl, ce

def make_all_tokens_debias_hook(
    h_idx: int,
    white_emb: torch.Tensor,
    black_emb: torch.Tensor,
    num_heads: int,
    head_dim: int,
    strength: float = 1.0,
):
    """Intervention hook applied to ALL token positions in the sequence."""
    diff = (white_emb - black_emb).to(torch.float32)
    d = diff / torch.norm(diff) if torch.norm(diff) > 1e-10 else torch.zeros_like(diff)
    # Target point b (neutral)
    b = 0.5 * (
        torch.sum(white_emb.to(torch.float32) * d)
        + torch.sum(black_emb.to(torch.float32) * d)
    )

    def hook_fn(module, inputs):
        inp = inputs[0]
        if not torch.is_tensor(inp):
            return inputs

        # o_proj pre-hook input is typically [B, S, hidden_size]
        bsz, slen, hidden = inp.shape
        expected_hidden = num_heads * head_dim

        # Handle config mismatch robustly by inferring head_dim from actual tensor
        if hidden != expected_hidden:
            if hidden % num_heads != 0:
                raise RuntimeError(
                    f"[exp26] Hidden size mismatch at o_proj: got hidden={hidden}, "
                    f"but num_heads={num_heads} does not divide it (expected {expected_hidden})."
                )
            effective_head_dim = hidden // num_heads
        else:
            effective_head_dim = head_dim

        # Ensure direction vector matches the effective head dim
        if d.numel() != effective_head_dim:
            d_eff = d[:effective_head_dim].contiguous() if d.numel() > effective_head_dim else F.pad(d, (0, effective_head_dim - d.numel()))
        else:
            d_eff = d

        inp2 = inp.clone()
        view = inp2.view(bsz, slen, num_heads, effective_head_dim)

        v = view[0, :, h_idx, :]  # [slen, effective_head_dim]
        d_dev = d_eff.to(v.device).to(v.dtype)
        b_dev = b.to(v.device).to(v.dtype)

        proj = torch.sum(v * d_dev, dim=-1, keepdim=True)  # [slen, 1]
        v_debiased = v - strength * (proj - b_dev) * d_dev
        view[0, :, h_idx, :] = v_debiased

        return (inp2,)

    return hook_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["baseline", "lora_ce", "lora_kl", "exp25_partial", "exp25_all", "exp4"],
        help="Run a single task per invocation to avoid model contamination.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="/home/common1/hwluo/project/pFairFT/exp26/results")
    parser.add_argument("--intervention_strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="/home/common1/hwluo/project/pFairFT/exp26/pre_logits_cache")
    parser.add_argument("--no_cache", action="store_true", help="Disable pre-logits caching")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip if individual result already exists
    individual_path = os.path.join(
        args.output_dir, f"exp26_{args.model_name}_{args.task}_openwebtext.csv"
    )
    if os.path.exists(individual_path):
        print(f"\n[SKIP] Task {args.task} for {args.model_name} already exists: {individual_path}")
        return
    
    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    device = get_input_device(model, "cuda")

    # Detect actual model configuration from model (Reference: exp2/analyze_race_sensitive_heads.py)
    print("Detecting actual model configuration from model...")
    config = get_model_config(model)
    num_heads, head_dim = config["num_heads"], config["head_dim"]

    def detect_config_hook(module, inputs, output):
        nonlocal num_heads, head_dim
        # output of o_proj is [B, S, hidden]
        # inputs[0] to o_proj is [B, S, num_heads * head_dim]
        # In Llama, the input to o_proj is the concatenated head outputs.
        inp = inputs[0]
        if torch.is_tensor(inp):
            hidden = inp.shape[-1]
            if hidden % num_heads == 0:
                detected_head_dim = hidden // num_heads
                if detected_head_dim != head_dim:
                    print(f"  Updating head_dim from {head_dim} to {detected_head_dim} based on o_proj input.")
                    head_dim = detected_head_dim

    # Run one dummy pass to trigger detection
    temp_hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(detect_config_hook)
    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**dummy_input)
        except Exception:
            pass
    temp_hook.remove()

    print(f"Final configuration: num_heads={num_heads}, head_dim={head_dim}")

    # Load OpenWebText
    print(f"Loading OpenWebText dataset (first {args.num_samples} samples)...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    texts = []
    it = iter(dataset)
    for _ in range(args.num_samples):
        try:
            texts.append(next(it)["text"])
        except StopIteration:
            break

    # Load Intervention Data
    sensitive_dir = f"/home/common1/hwluo/project/pFairFT/exp2_old/sensitive_heads_{args.model_name}_top100"
    with open(os.path.join(sensitive_dir, "results.pkl"), "rb") as f:
        emb_data = pickle.load(f)
    white_emb = emb_data.get("white_emb", {})
    black_emb = emb_data.get("black_emb", {})

    with open(os.path.join(sensitive_dir, "selected_heads_elbow.json"), "r") as f:
        partial_heads = [(h["layer"], h["head"]) for h in json.load(f)]
    all_heads = list(white_emb.keys())

    # Map one-task-per-run
    task_to_variant = {
        "baseline": {"name": "Baseline", "type": "baseline"},
        "lora_ce": {
            "name": "exp5_CE_Lora",
            "type": "lora",
            "path": f"/home/common1/hwluo/project/pFairFT/exp5_CE/lora_{args.model_name}_top100",
        },
        "lora_kl": {
            "name": "exp5_KL_Lora",
            "type": "lora",
            "path": f"/home/common1/hwluo/project/pFairFT/exp5_KL/lora_{args.model_name}_top100",
        },
        "exp25_partial": {
            "name": "exp25_Partial_Interv",
            "type": "intervention",
            "heads": partial_heads,
        },
        "exp25_all": {"name": "exp25_All_Interv", "type": "intervention", "heads": all_heads},
        "exp4": {
            "name": "exp4_Precision_Fairness",
            "type": "lora",
            "path": f"/home/common1/hwluo/project/pFairFT/exp4/precision_fairness_{args.model_name}_top100",
        },
    }

    var = task_to_variant[args.task]
    print(f"\nEvaluating single task: {var['name']} ({args.task})")

    # Base (teacher) model is always the raw model with NO hooks/adapters.
    base_model = model

    current_model = model
    hooks = []

    if var["type"] == "lora":
        # Check if adapter_config.json is in the path or final_model subfolder
        lora_path = os.path.abspath(var["path"])
        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            subfolder = os.path.join(lora_path, "final_model")
            if os.path.exists(os.path.join(subfolder, "adapter_config.json")):
                lora_path = subfolder
            else:
                raise FileNotFoundError(
                    f"adapter_config.json not found in {lora_path} or final_model/"
                )

        print(f"Loading LoRA adapter from: {lora_path}")
        # In this one-task-per-run mode, it's OK to attach adapter to the base model instance
        # because the process exits after finishing.
        current_model = PeftModel.from_pretrained(base_model, lora_path,
            trust_remote_code=True)
        current_model.eval()

    elif var["type"] == "intervention":
        for l, h in var["heads"]:
            if (l, h) in white_emb and (l, h) in black_emb:
                w_t = torch.from_numpy(white_emb[(l, h)]).float()
                b_t = torch.from_numpy(black_emb[(l, h)]).float()
                hook_fn = make_all_tokens_debias_hook(
                    h, w_t, b_t, num_heads, head_dim, args.intervention_strength
                )
                hooks.append(
                    base_model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_fn)
                )

    total_kl, total_ce, total_tokens = 0.0, 0.0, 0

    pre_cache = None
    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        pre_cache = PreLogitsCache(args.cache_dir, args.model_name)

    for sample_idx, text in enumerate(tqdm(texts, desc=var["name"])):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        if inputs["input_ids"].size(1) < 2:
            continue

        # Teacher (pre) logits are ALWAYS the baseline model w/ no hooks/adapters.
        # Cache per-sample to reuse across tasks for the same model.
        cached_pre = None
        if pre_cache is not None:
            cached_pre = pre_cache.get(
                sample_idx=sample_idx,
                max_length=args.max_length,
                input_ids=inputs["input_ids"][0],
            )

        with torch.no_grad():
            if cached_pre is not None:
                pre_logits = cached_pre.to(device).float()
            else:
                # Ensure no intervention hooks for the teacher pass
                if var["type"] == "intervention":
                    for hh in hooks:
                        hh.remove()

                pre_logits = base_model(**inputs).logits[0, :-1, :].float()

                # Re-register hooks for student pass if needed
                if var["type"] == "intervention":
                    hooks = []
                    for l, h in var["heads"]:
                        if (l, h) in white_emb and (l, h) in black_emb:
                            w_t = torch.from_numpy(white_emb[(l, h)]).float()
                            b_t = torch.from_numpy(black_emb[(l, h)]).float()
                            hook_fn = make_all_tokens_debias_hook(
                                h, w_t, b_t, num_heads, head_dim, args.intervention_strength
                            )
                            hooks.append(
                                base_model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook_fn)
                            )

                if pre_cache is not None:
                    pre_cache.save(
                        sample_idx=sample_idx,
                        max_length=args.max_length,
                        input_ids=inputs["input_ids"][0],
                        logits=pre_logits.cpu(),
                    )

            # Student (post) logits
            post_logits = current_model(**inputs).logits[0, :-1, :].float()

            kl, ce = compute_kl_ce_batch(pre_logits, post_logits)
            total_kl += kl.sum().item()
            total_ce += ce.sum().item()
            total_tokens += kl.size(0)

    # Always remove hooks even if an exception occurs to avoid exit-time crashes
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    result = {
        "Model": args.model_name,
        "Task": args.task,
        "Variant": var["name"],
        "Avg_KL": total_kl / total_tokens if total_tokens > 0 else 0.0,
        "Avg_CE": total_ce / total_tokens if total_tokens > 0 else 0.0,
        "Total_Tokens": total_tokens,
        "Num_Samples": len(texts),
        "Max_Length": args.max_length,
        "Intervention_Strength": args.intervention_strength,
    }

    # Save to individual CSV (for backup)
    individual_path = os.path.join(
        args.output_dir, f"exp26_{args.model_name}_{args.task}_openwebtext.csv"
    )
    pd.DataFrame([result]).to_csv(individual_path, index=False)

    # Append/Update master CSV
    master_path = os.path.join(args.output_dir, "exp26_all_results_openwebtext.csv")
    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        # Drop existing entry if same Model, Task, Variant to avoid duplicates
        mask = (master_df['Model'] == result['Model']) & \
               (master_df['Task'] == result['Task']) & \
               (master_df['Variant'] == result['Variant'])
        master_df = master_df[~mask]
        master_df = pd.concat([master_df, pd.DataFrame([result])], ignore_index=True)
    else:
        master_df = pd.DataFrame([result])
    
    master_df.to_csv(master_path, index=False)
    
    print(f"\nResult appended/updated in: {master_path}")
    print(f"Individual result saved to: {individual_path}")
    print(pd.DataFrame([result]).to_markdown(index=False))

    # Explicit cleanup to reduce exit-time thread/GIL issues
    try:
        # current_model is always defined; model is the baseline model
        if current_model is not model:
            del current_model
        del model
    except Exception:
        pass

    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return

if __name__ == "__main__":
    main()
