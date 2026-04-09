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
- `experiments/`: Historical and comprehensive experiment draft folders (`exp1` to `exp26`), showcasing iterative explorations on various models and datasets.
- `data/`: The `Resume` and `Anthropic` decision scenarios dataset.

## Usage
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
