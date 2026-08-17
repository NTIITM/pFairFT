# pFairFT: Precise Fairness Fine-Tuning

This repository implements the mechanistic interpretation and mitigation framework introduced in the paper **"Interpreting and Mitigating Discriminatory Behaviors in LLMs"**.

It consists of two major components:
- **Identify-then-Decide Analysis**: Localization of critical demographic-identifying Attention Heads and MLPs using causal intervention (Mean Ablation).
- **pFairFT (Precise Fairness Fine-Tuning) Trainer**: Efficient fairness alignment by applying LoRA fine-tuning strictly on the identified key components and employing an Affine Concept Editing (ACE) fairness constraint.

## Directory Structure

- `src/`
  - `identifier.py`: Contains `ComponentIdentifier` to run causal tracing.
  - `trainer.py`: Contains `PFairFTTrainer` for ACE-based targeted LoRA tuning.
  - `dataset.py`: Contains `CounterfactualDataset` for fact/counterfactual pairings.
  - Core utility scripts: `hook.py`, `probability.py`, `prompt.py`, `sampling.py`, `util.py`.
- `1_bias_evaluation/` through `6_downstream_evaluation/`: staged experiment code.
- `scripts/run_llama3_8b_figures.sh`: the only active experiment driver.
- `data/`: Resume and Discrim-Eval decision scenario datasets.

## Usage

Run commands from the repository root. The driver uses
`/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python` by default; select another
interpreter with `--python`. It validates the local Llama 3 8B checkpoint and
downloads `LLM-Research/Meta-Llama-3-8B-Instruct` from ModelScope when the
checkpoint is missing or incomplete.

Inspect the complete workflow without launching GPU experiments:

```bash
bash scripts/run_llama3_8b_figures.sh --stage figure1-figure5 --dry-run
```

Run all five experiments sequentially:

```bash
bash scripts/run_llama3_8b_figures.sh --stage figure1-figure5 --gpu 6,7
```

Alternatively, run one stage at a time. A later stage requires every earlier
stage to have a valid completion manifest; it never starts missing predecessors
implicitly.

```bash
bash scripts/run_llama3_8b_figures.sh --stage figure1 --gpu 6,7
bash scripts/run_llama3_8b_figures.sh --stage figure2 --gpu 6,7
bash scripts/run_llama3_8b_figures.sh --stage figure3 --gpu 6,7
bash scripts/run_llama3_8b_figures.sh --stage figure4 --gpu 6,7
bash scripts/run_llama3_8b_figures.sh --stage figure5 --gpu 6,7
```

The stages are:

1. `figure1`: Resume ranking and Head/MLP component identification.
2. `figure2`: Resume component intervention and head-count analysis.
3. `figure3`: MLP residual, logit, and token-attention mechanism analysis.
4. `figure4`: Discrim-Eval, COMPAS, and Adult intervention analysis.
5. `figure5`: three-epoch fine-tuning, transfer evaluation, context analysis, and activation geometry.

All new outputs are isolated under
`results/Meta-Llama-3-8B-Instruct-figures-v1`. Use `--result-root` to select a
different run. If a stage was interrupted or must be rebuilt, select it and pass
the same stage to `--force-stage`. This moves that stage and all downstream
outputs into the run's `stale/` directory before rebuilding:

```bash
bash scripts/run_llama3_8b_figures.sh \
  --stage figure3-figure5 \
  --force-stage figure3 \
  --gpu 6,7
```

Useful overrides:

```bash
bash scripts/run_llama3_8b_figures.sh \
  --stage figure1-figure5 \
  --python /path/to/python \
  --model-dir /path/to/Meta-Llama-3-8B-Instruct \
  --result-root /path/to/results \
  --gpu 0,1
```

The public method binding is intentionally fixed to the Figure 5 convention:

- `PFairFT` = `fairness_kl`
- `PFairFT-KL` = `fairness_kl_ce`
- `PFairFT-CE` = `fairness_ce`

Ranking, component identification, and training use Resume `summary_only` data.
Discrim-Eval is transfer-only. Formal random controls use seed 42, and mechanism
analyses use batch size 1.

The method is wrapped into OOP classes in `src`. To utilize the main methodologies, simply import the target module:

```python
from src.dataset import CounterfactualDataset
from src.identifier import ComponentIdentifier
from src.trainer import PFairFTTrainer

# 1. Load Data
dataset = CounterfactualDataset("data/...")

# 2. Identify Key heads & MLPs
identifier = ComponentIdentifier(model, tokenizer)
key_components = identifier.compute_importance_score(dataset)

# 3. Fine-tune utilizing ACE-based Fairness Constraint
trainer = PFairFTTrainer(model, tokenizer, target_components=key_components)
trainer.train_step(dataset)
```
