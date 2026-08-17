# 4. Intervention Ablation (干预消融实验)

**Paper Section**: §4.2–4.3 — Validating component importance through intervention

## Purpose
Validate that the identified components are truly responsible for discrimination by performing targeted interventions and measuring fairness changes. This module answers: **"Does intervening on these components reduce bias?"**

## Subdirectories

### `head_intervention/` (exp3, exp8, exp10)
Direct head activation replacement — forward/reverse intervention on Discrim-Eval and Resume.

### `mlp_intervention/` (exp15)
Mean ablation on selected MLP layers, with elbow-point-based selection.

### `projection_intervention/` (exp17, exp25)
ACE-based projection intervention — project activations to fairness anchor at inference time.

## Usage
```bash
bash scripts/run_llama3_8b_figures.sh --stage figure2
bash scripts/run_llama3_8b_figures.sh --stage figure4
```
