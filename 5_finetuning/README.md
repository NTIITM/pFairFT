# 5. Fine-Tuning (pFairFT 精准微调)

**Paper Section**: §6 — Precision Fair Fine-Tuning

## Purpose
Implement the **pFairFT** method: targeted LoRA injection on identified discriminatory heads with ACE (Affine Concept Editing) fairness constraints.

## Training Pipeline
1. Load fairness anchors from Component Identification results (`results.pkl`)
2. Inject LoRA adapters **only** on selected sensitive heads
3. Train with fairness loss: L = λ × L_f (project activations to neutral anchor)
4. Monitor KL divergence between fact/counterfactual outputs

## Scripts

| File | Origin | Description |
|------|--------|-------------|
| `finetune_precision_fairness.py` | exp4 | ★ Core: Precision LoRA + ACE fairness training |
| `finetune_global_lora.py` | exp5 | Baseline: Global LoRA fine-tuning (all heads) |
| `count_lora_and_precision_params.py` | exp4 | Parameter count comparison |
| `count_head_level_precision_params.py` | exp4 | Per-head parameter analysis |
| `evaluate_finetune_resume.py` | exp5 | Evaluate fine-tuned model on Resume |
| `visualize_finetune_results.py` | exp5 | Plot training curves |

## Usage
```bash
bash scripts/run_exp4_finetune.sh   # Precision fine-tuning
bash scripts/run_exp5_finetune.sh   # Global LoRA baseline
```
