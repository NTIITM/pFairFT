#!/usr/bin/env python
"""Shared cumulative-residual collection for layer-wise MLP probes."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from hook import get_last_token_indices_safe, remove_intervention_hooks
from prompt import format_prompt_for_model


InterventionFactory = Callable[[torch.Tensor], List[Any]]


def collect_next_mlp_inputs(
    model: Any,
    adapter: Any,
    tokenizer: Any,
    model_type: str,
    dataloader: Any,
    num_layers: int,
    prompt_key: str,
    intervention_factory: Optional[InterventionFactory] = None,
) -> Dict[int, np.ndarray]:
    """Collect layer-l cumulative residual probes at each sample's last token.

    For l < L-1 the probe is the input to layer (l+1)'s MLP/MOE block. For
    l == L-1 it is the input to the final normalization module.
    """
    chunks: Dict[int, List[np.ndarray]] = {layer: [] for layer in range(num_layers)}
    buffer: Dict[int, torch.Tensor] = {}
    collection_hooks = [
        adapter.register_next_mlp_input_hook(layer, buffer)
        for layer in range(num_layers)
    ]
    input_device = adapter.get_input_embedding_module().weight.device

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Collect cumulative residuals ({prompt_key})"):
                prompts = [format_prompt_for_model(p, model_type) for p in batch[prompt_key]]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value):
                        inputs[key] = value.to(input_device)

                attention_mask = inputs.get(
                    "attention_mask", torch.ones_like(inputs["input_ids"])
                )
                last_token_indices = get_last_token_indices_safe(
                    inputs["input_ids"], attention_mask, tokenizer
                )
                intervention_hooks = (
                    intervention_factory(last_token_indices)
                    if intervention_factory is not None
                    else []
                )
                buffer.clear()
                try:
                    model(**inputs)
                finally:
                    remove_intervention_hooks(intervention_hooks)

                rows = torch.arange(inputs["input_ids"].shape[0])
                for layer, states in buffer.items():
                    positions = last_token_indices.to(states.device)
                    last_states = states[rows.to(states.device), positions, :]
                    chunks[layer].append(last_states.cpu().float().numpy())
    finally:
        remove_intervention_hooks(collection_hooks)

    return {
        layer: np.concatenate(layer_chunks, axis=0)
        for layer, layer_chunks in chunks.items()
        if layer_chunks
    }
