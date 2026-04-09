# 6. Downstream Evaluation (下游评估)

**Paper Section**: §6.2 — Evaluating fairness-utility trade-off

## Purpose
Evaluate fine-tuned models on both **fairness benchmarks** (Discrim-Eval, Resume) and **capability benchmarks** (MMLU) to verify that bias mitigation does not degrade model utility.

## Scripts

| File | Origin | Description |
|------|--------|-------------|
| `evaluate_mmlu.py` | exp14 | MMLU benchmark evaluation |
| `analyze_mmlu_results.py` | exp14 | Analyze MMLU scores |
| `evaluate_models_discrim.py` | exp16 | Compare models on Discrim-Eval |
| `evaluate_resume_fairness_top100.py` | exp18 | Resume top-100 fairness (KL/CE/ours) |
| `summarize_mmlu_exp25.py` | exp25 | MMLU summary after intervention |
| `evaluate_mmlu_intervention.py` | exp25 | MMLU with ACE intervention |
| `compute_exp26_metrics.py` | exp26 | Compute comprehensive metrics |

## Usage
```bash
bash scripts/run_exp14.sh  # MMLU evaluation
bash scripts/run_exp16.sh  # Discrim-Eval comparison
```
