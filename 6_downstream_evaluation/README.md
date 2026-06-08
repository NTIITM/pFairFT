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

Use the repository-level standard driver for MOE downstream evaluation:

```bash
MODEL_NAME=JetMoE-8B-Chat \
MODEL_PATH=/mnt/nfs/models/JetMoE-8B-Chat \
MODEL_TYPE=jetmoe \
DRY_RUN=0 RUN_ALL=0 RUN_RESUME_EVAL=1 RUN_DISCRIM_EVAL=1 RUN_MMLU=1 RUN_PLOTS=1 \
bash scripts/run_moe_resume_standard.sh
```

`evaluate_models_discrim.py` appends to its CSV output, so fresh standard runs should
remove or rename the target CSV first. The standard driver removes fresh output paths
before writing them.
