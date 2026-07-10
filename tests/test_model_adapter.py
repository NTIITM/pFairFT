import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hook import get_mlp_last_token_patch_hook
from model_adapter import JetMoeAdapter, ModelArchitectureAdapter


class IdentityAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, x):
        return self.o_proj(x)


class DenseLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = IdentityAttention(hidden_size)
        self.mlp = nn.Identity()

    def forward(self, x):
        x = self.self_attn(x)
        return self.mlp(x)


class DenseBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden_size)
        self.layers = nn.ModuleList(
            [DenseLayer(hidden_size) for _ in range(num_layers)]
        )
        self.norm = nn.Identity()


class DenseLM(nn.Module):
    def __init__(self, hidden_size: int = 4, num_layers: int = 2):
        super().__init__()
        self.model = DenseBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, 8, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


class JetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Identity()
        self.mlp = nn.Identity()


class JetLM(DenseLM):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([JetLayer(), JetLayer()])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_dense_head_mean_replacement_uses_o_proj_input():
    model = DenseLM()
    adapter = ModelArchitectureAdapter(model)
    hook = adapter.register_head_mean_replacement_hook(
        layer_idx=0,
        head_idx=1,
        mean_embedding=torch.tensor([9.0, 8.0]),
        output_pos=torch.tensor([1]),
        num_heads=2,
        head_dim=2,
    )
    try:
        x = torch.zeros(1, 3, 4)
        output = model.model.layers[0].self_attn(x)
    finally:
        hook.remove()
    torch.testing.assert_close(output[0, 1], torch.tensor([0.0, 0.0, 9.0, 8.0]))


def test_jet_head_mean_replacement_uses_attention_output():
    model = JetLM()
    adapter = JetMoeAdapter(model)
    hook = adapter.register_head_mean_replacement_hook(
        layer_idx=0,
        head_idx=0,
        mean_embedding=torch.tensor([3.0, 4.0]),
        output_pos=2,
        num_heads=2,
        head_dim=2,
    )
    try:
        x = torch.zeros(1, 3, 4)
        output = model.model.layers[0].self_attention(x)
    finally:
        hook.remove()
    torch.testing.assert_close(output[0, 2], torch.tensor([3.0, 4.0, 0.0, 0.0]))


def test_next_mlp_input_surface_and_final_norm_surface():
    model = DenseLM()
    adapter = ModelArchitectureAdapter(model)
    buffer = {}
    hooks = [adapter.register_next_mlp_input_hook(i, buffer) for i in range(2)]
    try:
        x = torch.randn(1, 3, 4)
        for layer in model.model.layers:
            x = layer(x)
        model.model.norm(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert set(buffer) == {0, 1}
    torch.testing.assert_close(buffer[0], x)
    torch.testing.assert_close(buffer[1], x)


def test_residual_projection_matches_lm_head_without_norm():
    model = DenseLM()
    adapter = ModelArchitectureAdapter(model)
    x = torch.randn(2, 4)
    torch.testing.assert_close(
        adapter.project_residual_to_logits(x, apply_final_norm=False),
        model.lm_head(x),
    )


def test_mlp_mean_replacement_supports_per_sample_positions():
    model = DenseLM()
    adapter = ModelArchitectureAdapter(model)
    hook = adapter.register_mlp_mean_replacement_hook(
        layer_idx=0,
        mean_embedding=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        output_pos=torch.tensor([0, 2]),
    )
    try:
        x = torch.zeros(2, 3, 4)
        output = model.model.layers[0].mlp(x)
    finally:
        hook.remove()
    torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(output[1, 2], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(output[0, 1], torch.zeros(4))


def test_jet_debias_projection_uses_attention_output_surface():
    model = JetLM()
    adapter = JetMoeAdapter(model)
    hook = adapter.register_head_debias_projection_hook(
        layer_idx=0,
        head_idx=0,
        group1_embedding=torch.tensor([1.0, 0.0]),
        group2_embedding=torch.tensor([-1.0, 0.0]),
        combined_std=None,
        output_pos=1,
        intervention_strength=1.0,
        num_heads=2,
        head_dim=2,
        use_std=False,
    )
    try:
        x = torch.zeros(1, 3, 4)
        x[0, 1, :2] = torch.tensor([2.0, 3.0])
        output = model.model.layers[0].self_attention(x)
    finally:
        hook.remove()
    torch.testing.assert_close(output[0, 1], torch.tensor([0.0, 3.0, 0.0, 0.0]))


def test_mlp_replacement_hooks_preserve_tuple_aux_output():
    hidden = torch.zeros(2, 3, 4)
    aux = torch.tensor(0.25)
    mean_hook = __import__("hook").make_mlp_intervention_hook_mean_replacement(
        layer_idx=0,
        mean_embedding=torch.ones(4),
        output_pos=torch.tensor([0, 2]),
    )
    mean_output = mean_hook(None, (), (hidden, aux))
    assert isinstance(mean_output, tuple)
    assert mean_output[1] is aux
    torch.testing.assert_close(mean_output[0][0, 0], torch.ones(4))
    torch.testing.assert_close(mean_output[0][1, 2], torch.ones(4))

    patch_hook = get_mlp_last_token_patch_hook(
        cf_batch_tensor=torch.full((2, 4), 3.0),
        last_token_indices=torch.tensor([1, 2]),
    )
    patched_output = patch_hook(None, (), (hidden, aux))
    assert patched_output[1] is aux
    torch.testing.assert_close(patched_output[0][0, 1], torch.full((4,), 3.0))
    torch.testing.assert_close(patched_output[0][1, 2], torch.full((4,), 3.0))
