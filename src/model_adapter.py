#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model architecture adapters for component-level interventions.

This module keeps architecture-specific module traversal and hook behavior out
of experiment scripts. The default adapter preserves the previous LLaMA/Qwen/
DeepSeek-style path where head activations are captured at self_attn.o_proj
input. OLMoE uses the same path. JetMoE does not expose a standard o_proj, so
its adapter uses the post-attention hidden state as a head-sliced proxy.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from hook import (
    create_config_detection_hook,
    get_activation_hook_for_intervention,
    get_patch_hook_modified,
)


RouterKey = Tuple[int, str]


def _model_type_from(model: Any, requested: str = "auto", model_path: str = "") -> str:
    requested_lower = requested.lower() if isinstance(requested, str) else ""
    if requested_lower and requested_lower != "auto":
        return requested_lower

    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None else None
    if isinstance(mt, str) and mt:
        return mt.lower()

    path_lower = model_path.lower() if isinstance(model_path, str) else ""
    for key in ("jetmoe", "olmoe", "deepseek", "qwen", "llama"):
        if key in path_lower:
            return key
    return "default"


def _unwrap_to_causal_lm(model: Any) -> Any:
    """Return the underlying causal LM when PEFT wraps it."""
    candidate = model
    if hasattr(candidate, "peft_config") and hasattr(candidate, "base_model") and hasattr(candidate.base_model, "model"):
        candidate = candidate.base_model.model
    return candidate


def _get_nested_attr(obj: Any, path: str) -> Optional[Any]:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _extract_first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor or tuple[tensor, ...], got {type(output)}")
    return output


def make_attention_output_activation_hook(
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    batch_activations_buffer: Dict[int, torch.Tensor],
) -> Callable:
    """Capture module output and split its hidden dimension into heads."""

    def hook(module, inputs, output):
        hidden_state = _extract_first_tensor(output)
        bsz, seqlen, hidden_dim = hidden_state.shape
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Layer {layer_idx}: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads})."
                )
            actual_head_dim = hidden_dim // num_heads
        else:
            actual_head_dim = head_dim
        batch_activations_buffer[layer_idx] = hidden_state.view(bsz, seqlen, num_heads, actual_head_dim)

    return hook


def make_attention_output_patch_hook(
    head_to_patch: int,
    fact_activation_batch: torch.Tensor,
    cf_activation_batch: torch.Tensor,
    last_token_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> Callable:
    """Patch a head slice in a module forward output.

    This is used for architectures such as JetMoE where there is no standard
    attention o_proj input to patch.
    """

    def hook(module, inputs, output):
        hidden_state = _extract_first_tensor(output)
        new_hidden = hidden_state.clone()
        bsz, seqlen, hidden_dim = new_hidden.shape
        expected_hidden = num_heads * head_dim
        if hidden_dim != expected_hidden:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Expected hidden_dim divisible by num_heads={num_heads}, got {hidden_dim}."
                )
            actual_head_dim = hidden_dim // num_heads
        else:
            actual_head_dim = head_dim

        heads_view = new_hidden.view(bsz, seqlen, num_heads, actual_head_dim)
        device = new_hidden.device
        batch_idxs = torch.arange(bsz, device=device)
        pos = last_token_indices.to(device)

        fact_vals = fact_activation_batch.to(device=device, dtype=new_hidden.dtype)
        cf_vals = cf_activation_batch.to(device=device, dtype=new_hidden.dtype)
        heads_view[batch_idxs, pos, :, :] = fact_vals[:, :, :actual_head_dim]
        heads_view[batch_idxs, pos, head_to_patch, :] = cf_vals[:, head_to_patch, :actual_head_dim]

        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook


def make_router_cache_hook(
    router_key: RouterKey,
    cache: Dict[int, Dict[RouterKey, Any]],
    get_batch_indices: Callable[[], Iterable[int]],
) -> Callable:
    """Cache router outputs for the current batch.

    Supports both tensor router logits (OLMoE/JetMoE) and the historical tuple
    format used by the DeepSeek-specific script.
    """

    def hook(module, args, output):
        if not args:
            return output
        batch_indices = list(get_batch_indices())
        if not batch_indices:
            return output

        inp = args[0]
        if inp.ndim >= 3:
            bsz, seq_len = inp.shape[:2]
        elif inp.ndim == 2:
            bsz = len(batch_indices)
            if inp.shape[0] % bsz != 0:
                return output
            seq_len = inp.shape[0] // bsz
        else:
            return output

        if torch.is_tensor(output):
            values = output.detach().view(bsz, seq_len, -1).cpu()
            for i, b_idx in enumerate(batch_indices):
                cache.setdefault(int(b_idx), {})[router_key] = values[i].clone()
            return output

        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
            idx = output[0].detach().view(bsz, seq_len, -1).cpu()
            wt = output[1].detach().view(bsz, seq_len, -1).cpu()
            for i, b_idx in enumerate(batch_indices):
                cache.setdefault(int(b_idx), {})[router_key] = (idx[i].clone(), wt[i].clone())
            return output

        return output

    return hook


def make_router_force_hook(
    router_key: RouterKey,
    cache: Dict[int, Dict[RouterKey, Any]],
    get_batch_indices: Callable[[], Iterable[int]],
) -> Callable:
    """Force router outputs from a cached fact pass when available."""

    def hook(module, args, output):
        if not args:
            return output
        batch_indices = list(get_batch_indices())
        if not batch_indices:
            return output
        inp = args[0]
        if inp.ndim >= 3:
            bsz, seq_len = inp.shape[:2]
        elif inp.ndim == 2:
            bsz = len(batch_indices)
            if inp.shape[0] % bsz != 0:
                return output
            seq_len = inp.shape[0] // bsz
        else:
            return output
        device = args[0].device

        if torch.is_tensor(output):
            forced = output.clone().view(bsz, seq_len, -1)
            for i, b_idx in enumerate(batch_indices):
                cached = cache.get(int(b_idx), {}).get(router_key)
                if torch.is_tensor(cached):
                    forced[i] = cached.to(device=device, dtype=forced.dtype)
            return forced.view_as(output)

        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
            topk_idx = output[0].view(bsz, seq_len, -1).clone()
            topk_wt = output[1].view(bsz, seq_len, -1).clone()
            for i, b_idx in enumerate(batch_indices):
                cached = cache.get(int(b_idx), {}).get(router_key)
                if isinstance(cached, tuple) and len(cached) == 2:
                    topk_idx[i] = cached[0].to(device=device, dtype=topk_idx.dtype)
                    topk_wt[i] = cached[1].to(device=device, dtype=topk_wt.dtype)
            new_output = (topk_idx.view_as(output[0]), topk_wt.view_as(output[1]))
            if len(output) > 2:
                new_output += output[2:]
            return new_output

        return output

    return hook


class ModelArchitectureAdapter:
    family = "default"
    head_activation_kind = "o_proj_input"

    def __init__(self, model: Any, model_type: str = "auto", model_path: str = ""):
        self.model = model
        self.model_type = model_type
        self.model_path = model_path

    @property
    def causal_lm(self) -> Any:
        return _unwrap_to_causal_lm(self.model)

    def get_layers(self):
        lm = self.causal_lm
        for path in ("model.layers", "transformer.h", "layers"):
            layers = _get_nested_attr(lm, path)
            if layers is not None:
                return layers
        raise ValueError(f"Cannot find decoder layers for model type {type(self.model)}")

    def get_config(self) -> Dict[str, Any]:
        cfg = getattr(self.causal_lm, "config", getattr(self.model, "config", None))
        layers = self.get_layers()
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden_size is None:
            attn = self.get_attention_module(0)
            q_proj = getattr(attn, "q_proj", None)
            hidden_size = getattr(q_proj, "in_features", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size.")

        num_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", None)
            or getattr(self.get_attention_module(0), "num_heads", None)
        )
        head_dim = (
            getattr(cfg, "head_dim", None)
            or getattr(cfg, "v_head_dim", None)
            or getattr(cfg, "kv_channels", None)
            or getattr(self.get_attention_module(0), "head_dim", None)
        )
        if num_heads is None and head_dim is not None:
            num_heads = hidden_size // head_dim
        if head_dim is None and num_heads is not None:
            head_dim = hidden_size // num_heads
        if num_heads is None or head_dim is None:
            raise ValueError("Cannot infer num_heads/head_dim.")

        return {
            "num_layers": len(layers),
            "hidden_size": int(hidden_size),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
        }

    def get_attention_module(self, layer_idx: int) -> Any:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "self_attention"):
            return layer.self_attention
        raise ValueError(f"Cannot find attention module in layer {layer_idx}.")

    def get_head_activation_module(self, layer_idx: int) -> Any:
        attn = self.get_attention_module(layer_idx)
        if hasattr(attn, "o_proj"):
            return attn.o_proj
        raise ValueError(f"Cannot find self_attn.o_proj in layer {layer_idx}.")

    def get_lm_head_weight(self) -> torch.Tensor:
        lm = self.causal_lm
        if hasattr(lm, "lm_head") and hasattr(lm.lm_head, "weight"):
            return lm.lm_head.weight
        if hasattr(lm, "get_output_embeddings"):
            embeddings = lm.get_output_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                return embeddings.weight
        raise ValueError("Cannot find lm_head/output embedding weight.")

    def get_lm_head_module(self) -> Any:
        lm = self.causal_lm
        if hasattr(lm, "lm_head"):
            return lm.lm_head
        if hasattr(lm, "get_output_embeddings"):
            embeddings = lm.get_output_embeddings()
            if embeddings is not None:
                return embeddings
        raise ValueError("Cannot find lm_head/output embedding module.")

    @staticmethod
    def _align_head_dim(head_activations: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = head_activations.shape[-1]
        if current_dim == target_dim:
            return head_activations
        if current_dim > target_dim:
            return head_activations[..., :target_dim].contiguous()
        pad_width = target_dim - current_dim
        return torch.nn.functional.pad(head_activations, (0, pad_width))

    def project_head_activations_to_logits(
        self,
        layer_idx: int,
        head_idx: int,
        head_activations: torch.Tensor,
        num_heads: int,
        head_dim: int,
        lm_head_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project one head's captured activations to vocabulary logits.

        Default models capture the input to `self_attn.o_proj`, so a head is
        first projected through the matching `o_proj` slice and then through
        the output embedding. MOE adapters can override this when their hook
        captures a different activation surface.
        """
        del head_dim  # The actual projection width is determined from weight shape.
        o_proj_weight = self.get_head_activation_module(layer_idx).weight
        o_proj_device = o_proj_weight.device
        o_proj_dtype = o_proj_weight.dtype

        if o_proj_weight.shape[1] % num_heads != 0:
            raise ValueError(
                f"Layer {layer_idx}: o_proj input dim {o_proj_weight.shape[1]} is not divisible by num_heads={num_heads}."
            )
        effective_head_dim = o_proj_weight.shape[1] // num_heads
        start = head_idx * effective_head_dim
        end = (head_idx + 1) * effective_head_dim

        if getattr(o_proj_weight, "is_meta", False):
            acts = self._align_head_dim(
                head_activations.to(dtype=o_proj_dtype),
                effective_head_dim,
            )
            full_input = torch.zeros(
                acts.shape[0],
                o_proj_weight.shape[1],
                device=acts.device,
                dtype=o_proj_dtype,
            )
            full_input[:, start:end] = acts
            hidden = self.get_head_activation_module(layer_idx)(full_input)
            return self.get_lm_head_module()(hidden)

        acts = self._align_head_dim(
            head_activations.to(device=o_proj_device, dtype=o_proj_dtype),
            effective_head_dim,
        )
        hidden = acts @ o_proj_weight[:, start:end].t()

        if lm_head_weight is not None and not getattr(lm_head_weight, "is_meta", False):
            lm_head_on_device = lm_head_weight.to(device=o_proj_device, dtype=o_proj_dtype)
            return hidden @ lm_head_on_device.t()
        return self.get_lm_head_module()(hidden)

    def register_config_detection_hook(self, buffer: Dict[str, Any]):
        return self.get_head_activation_module(0).register_forward_hook(create_config_detection_hook(buffer))

    def register_head_activation_hook(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        batch_activations_buffer: Dict[int, torch.Tensor],
    ):
        module = self.get_head_activation_module(layer_idx)
        return module.register_forward_hook(
            get_activation_hook_for_intervention(layer_idx, num_heads, head_dim, batch_activations_buffer)
        )

    def register_head_patch_hook(
        self,
        layer_idx: int,
        head_idx: int,
        fact_layer_tensor: torch.Tensor,
        cf_layer_tensor: torch.Tensor,
        last_token_indices: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ):
        module = self.get_head_activation_module(layer_idx)
        hook_fn = get_patch_hook_modified(
            head_idx, fact_layer_tensor, cf_layer_tensor, last_token_indices, num_heads, head_dim
        )
        return module.register_forward_pre_hook(hook_fn)

    def lora_target_modules(self) -> List[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def head_mask_layer_key(self, layer_idx: int) -> str:
        return f"layers.{layer_idx}.self_attn"

    def router_modules_for_freeze(self) -> List[Tuple[RouterKey, Any]]:
        modules = []
        for layer_idx, layer in enumerate(self.get_layers()):
            gate = _get_nested_attr(layer, "mlp.gate")
            if gate is not None:
                modules.append(((layer_idx, "mlp"), gate))
        return modules


class OlmoeAdapter(ModelArchitectureAdapter):
    family = "olmoe"


class JetMoeAdapter(ModelArchitectureAdapter):
    family = "jetmoe"
    head_activation_kind = "attention_output"

    def get_layers(self):
        lm = self.causal_lm
        layers = _get_nested_attr(lm, "model.layers")
        if layers is not None:
            return layers
        return super().get_layers()

    def get_attention_module(self, layer_idx: int) -> Any:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "self_attention"):
            return layer.self_attention
        return super().get_attention_module(layer_idx)

    def get_head_activation_module(self, layer_idx: int) -> Any:
        return self.get_attention_module(layer_idx)

    def project_head_activations_to_logits(
        self,
        layer_idx: int,
        head_idx: int,
        head_activations: torch.Tensor,
        num_heads: int,
        head_dim: int,
        lm_head_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project JetMoE attention-output head slices directly to logits.

        JetMoE does not expose a standard attention `o_proj`. Its exp23 head
        activations are captured after attention, already in the residual
        hidden-state basis, so each head slice maps to the corresponding
        columns of the output embedding.
        """
        del layer_idx, head_dim
        if lm_head_weight is None:
            lm_head_weight = self.get_lm_head_weight()

        lm_head_dtype = lm_head_weight.dtype
        hidden_dim = lm_head_weight.shape[1]
        if hidden_dim % num_heads != 0:
            raise ValueError(f"lm_head hidden dim {hidden_dim} is not divisible by num_heads={num_heads}.")

        effective_head_dim = hidden_dim // num_heads
        start = head_idx * effective_head_dim
        end = (head_idx + 1) * effective_head_dim
        if not getattr(lm_head_weight, "is_meta", False):
            acts = self._align_head_dim(
                head_activations.to(device=lm_head_weight.device, dtype=lm_head_dtype),
                effective_head_dim,
            )
            lm_head_on_device = lm_head_weight.to(device=acts.device, dtype=lm_head_dtype)
            return acts @ lm_head_on_device[:, start:end].t()

        acts = self._align_head_dim(
            head_activations.to(dtype=lm_head_dtype),
            effective_head_dim,
        )
        hidden = torch.zeros(
            acts.shape[0],
            hidden_dim,
            device=acts.device,
            dtype=lm_head_dtype,
        )
        hidden[:, start:end] = acts
        return self.get_lm_head_module()(hidden)

    def register_config_detection_hook(self, buffer: Dict[str, Any]):
        def hook(module, inputs, output):
            hidden_state = _extract_first_tensor(output)
            hidden_dim = hidden_state.shape[-1]
            cfg = self.get_config()
            buffer["hidden_dim"] = hidden_dim
            buffer["num_heads"] = cfg["num_heads"]
            buffer["head_dim"] = hidden_dim // cfg["num_heads"]

        return self.get_head_activation_module(0).register_forward_hook(hook)

    def register_head_activation_hook(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        batch_activations_buffer: Dict[int, torch.Tensor],
    ):
        module = self.get_head_activation_module(layer_idx)
        return module.register_forward_hook(
            make_attention_output_activation_hook(layer_idx, num_heads, head_dim, batch_activations_buffer)
        )

    def register_head_patch_hook(
        self,
        layer_idx: int,
        head_idx: int,
        fact_layer_tensor: torch.Tensor,
        cf_layer_tensor: torch.Tensor,
        last_token_indices: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ):
        module = self.get_head_activation_module(layer_idx)
        hook_fn = make_attention_output_patch_hook(
            head_idx, fact_layer_tensor, cf_layer_tensor, last_token_indices, num_heads, head_dim
        )
        return module.register_forward_hook(hook_fn)

    def lora_target_modules(self) -> List[str]:
        # JetMoE uses custom ParallelExperts for query/output projections; PEFT
        # LoRA can safely target standard Linear modules such as kv_proj. Router
        # LoRA is left out by default because it changes expert routing policy.
        return ["kv_proj"]

    def head_mask_layer_key(self, layer_idx: int) -> str:
        return f"layers.{layer_idx}.self_attention"

    def router_modules_for_freeze(self) -> List[Tuple[RouterKey, Any]]:
        modules = []
        for layer_idx, layer in enumerate(self.get_layers()):
            attn_router = _get_nested_attr(layer, "self_attention.experts.router.layer")
            mlp_router = _get_nested_attr(layer, "mlp.router.layer")
            if attn_router is not None:
                modules.append(((layer_idx, "attn"), attn_router))
            if mlp_router is not None:
                modules.append(((layer_idx, "mlp"), mlp_router))
        return modules


def get_model_adapter(model: Any, model_type: str = "auto", model_path: str = "") -> ModelArchitectureAdapter:
    family = _model_type_from(model, model_type, model_path)
    if "jetmoe" in family:
        return JetMoeAdapter(model, model_type=model_type, model_path=model_path)
    if "olmoe" in family:
        return OlmoeAdapter(model, model_type=model_type, model_path=model_path)
    return ModelArchitectureAdapter(model, model_type=model_type, model_path=model_path)
