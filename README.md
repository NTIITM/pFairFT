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
- `scripts/run_moe_resume_standard.sh`: standard MOE resume-ranking workflow driver.
- `data/`: Resume and Discrim-Eval decision scenario datasets.

## Usage

For current MOE resume-transfer experiments, use the standard driver instead of old per-experiment shell scripts:

```bash
MODEL_NAME=JetMoE-8B-Chat \
MODEL_PATH=/mnt/nfs/models/JetMoE-8B-Chat \
MODEL_TYPE=jetmoe \
bash scripts/run_moe_resume_standard.sh
```

The driver defaults to `DRY_RUN=1`. Set `DRY_RUN=0` to execute, and use `RUN_RANKING`,
`RUN_HEADS`, `RUN_TRAIN`, `RUN_RESUME_EVAL`, `RUN_DISCRIM_EVAL`, `RUN_MMLU`, and
`RUN_PLOTS` to select phases. The standard comparison branches are baseline, Global
LoRA CE, PFairFT, PFairFT-KL, and PFairFT-KL-CE. PFairFT means precise selected heads
with affine fairness and CE; there is no separate PFairFT-CE branch.

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
