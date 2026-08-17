# 2. Component Identification (关键组件识别)

**Paper Section**: §4.1 — Causal Intervention for identifying discriminatory components

## Purpose
Use **Mean Ablation** (causal intervention) to identify race-sensitive attention heads and MLPs. This module answers: **"Which components cause discriminatory behavior?"**

## Key Algorithm
1. Collect activations on factual and counterfactual (race-swapped) inputs
2. For each head: patch its activation with counterfactual values while keeping others at factual values
3. Measure KL divergence between original and patched outputs
4. Use **Elbow Method** to select the most influential heads

## Scripts

| File | Origin | Description |
|------|--------|-------------|
| `analyze_race_sensitive_heads.py` | exp2 | ★ Core: Causal intervention to find sensitive heads |
| `analyze_race_sensitive_MLPs.py` | exp2 | Causal intervention for MLP layers |
| `analyze_topk_bias_with_cf.py` | exp2 | Top-K bias analysis with counterfactual |
| `compute_topk_mean_bias.py` | exp2 | Compute mean bias for top-K heads |
| `count_selected_heads_elbow.py` | exp2 | Elbow method for head selection |
| `evaluate_biased_sample.py` | exp2 | Identify most biased samples |
| `evaluate_intervention.py` | exp2 | Evaluate intervention effects |
| `evaluate_intervention_by_head_count.py` | exp9 | Vary number of intervened heads |
| `evaluate_intervention_by_head_count_random.py` | exp9 | Random baseline comparison |
| `intervention_all_model_resume.py` | exp9 | Cross-model intervention analysis |

## Usage
```bash
bash scripts/run_llama3_8b_figures.sh --stage figure1
bash scripts/run_llama3_8b_figures.sh --stage figure2
```
