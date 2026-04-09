# 1. Bias Evaluation (偏见评估)

**Paper Section**: §3 — Evaluating discriminatory behavior across LLMs

## Purpose
Measure and quantify bias in LLMs using the Discrim-Eval benchmark and Resume datasets. This module answers: **"How biased is this model?"**

## Scripts

| File | Origin | Description |
|------|--------|-------------|
| `evaluate_bias_discrim.py` | exp1 | Evaluate models on Discrim-Eval for race bias |
| `evaluate_with_context.py` | exp1 | Evaluate with context prompts (Qwen) |
| `evaluate_with_context_llama.py` | exp1 | Evaluate with context prompts (LLaMA) |
| `create_dataset.py` | exp1 | Prepare Discrim-Eval paired datasets |
| `find_significant_debiased_question.py` | exp1 | Identify scenarios where debiased prompts fail |
| `compute_fact_race_prob_diff.py` | exp6 | Compute P(yes) gap between demographic groups |
| `compute_all_models_bias.py` | exp6 | Multi-model bias comparison |
| `evaluate_debiased_prompt.py` | exp24 | Evaluate debiased prompt effectiveness |
| `summarize_mean_abs_pyes_diff.py` | exp24 | Summarize fairness metrics |

## Usage
```bash
bash scripts/exp.sh
```
