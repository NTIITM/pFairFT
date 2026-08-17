# 3. Pattern Analysis (模式分析)

**Paper Section**: §4.2 — Analyzing functional roles of discriminatory components

## Purpose
Analyze **how** the identified components contribute to bias: attention patterns, head output influence on logits, MLP behavior, and debiased prompt mechanisms.

## Subdirectories

### `head_attention_pattern/` (exp11)
Analyze QK attention scores of sensitive heads on factual vs. counterfactual inputs.

### `head_logit_analysis/` (exp12, exp20)
Measure how each head's output influences the final yes/no logit probability via KL divergence.

### `mlp_analysis/` (exp13, exp20)
Analyze MLP input/output similarity and their contribution to biased output.

### `debiased_prompt_analysis/` (exp21)
Study head behavior under debiased prompts — why debiasing sometimes fails.

- `plot_debiased_prompt_head_l2.py`: plot head activation L2 norms before and after debiased prompting, highlighting selected sensitive heads.

### `model_comparison/` (exp23)
Compare head activation patterns between fine-tuned and baseline models.

- `compare_adapter_head_fairness_gap.py`: compute per-head mean absolute factual-vs-counterfactual p(yes) gaps for two adapters on one Discrim-Eval QID.
- `plot_adapter_head_fairness_gap.py`: plot those adapter head-level gap matrices with sensitive heads highlighted.

## Usage
```bash
bash scripts/run_llama3_8b_figures.sh --stage figure3
bash scripts/run_llama3_8b_figures.sh --stage figure5
```
