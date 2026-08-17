# 5. Fine-Tuning (pFairFT 精准微调)

**Paper Section**: §6 — Precision Fair Fine-Tuning

## Purpose
Implement the **pFairFT** method: targeted LoRA injection on identified discriminatory heads with ACE (Affine Concept Editing) fairness constraints.

## Training Pipeline
1. Load fairness anchors from Component Identification results (`results.pkl`)
2. Inject LoRA adapters **only** on selected sensitive heads
3. Train selected comparison branches:
   - Global LoRA CE: CE on fact/counterfactual resume pairs across all LoRA target modules.
   - PFairFT: precise selected heads with affine fairness and KL (`fairness_kl`).
   - PFairFT-KL: precise selected heads with affine fairness, KL, and CE (`fairness_kl_ce`).
   - PFairFT-CE: precise selected heads with affine fairness and CE (`fairness_ce`).
4. Monitor branch-specific loss metrics and downstream fairness/utility evaluation.

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

Use the repository-level Llama 3 8B Figure workflow:

```bash
bash scripts/run_llama3_8b_figures.sh --stage figure5
```

Figure 5 requires validated Figure 1-4 stage manifests and trains every branch
for three epochs on the full Resume `summary_only` population.
