import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finetune_precision_fairness",
    ROOT / "5_finetuning" / "finetune_precision_fairness.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeLoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 2):
        super().__init__()
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(in_features, rank, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, out_features, bias=False)}
        )


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = FakeLoraLinear(8, 8)
        self.k_proj = FakeLoraLinear(8, 4)
        self.v_proj = FakeLoraLinear(8, 4)
        self.o_proj = FakeLoraLinear(8, 8)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])


def test_gqa_head_mask_uses_projection_axes_and_freezes_unselected_layers():
    model = FakeModel()
    masks = MODULE.create_head_masks(
        [{"layer": 0, "head": 1}],
        num_heads=4,
        head_dim=2,
        num_key_value_heads=2,
    )

    for param in model.parameters():
        param.data.fill_(1.0)
        param.grad = torch.ones_like(param)

    MODULE._apply_head_grad_mask_(model, masks)

    attn = model.layers[0].self_attn
    torch.testing.assert_close(
        attn.q_proj.lora_B["default"].weight.grad[:, 0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        attn.o_proj.lora_A["default"].weight.grad[0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        attn.k_proj.lora_B["default"].weight.grad[:, 0],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    assert torch.all(attn.q_proj.lora_A["default"].weight.grad == 1)
    assert torch.all(attn.o_proj.lora_B["default"].weight.grad == 1)
    assert all(param.grad is None for param in model.layers[1].parameters())

    MODULE._enforce_head_param_mask_(model, masks)
    torch.testing.assert_close(
        attn.q_proj.lora_B["default"].weight[:, 0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        attn.o_proj.lora_A["default"].weight[0],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        attn.v_proj.lora_B["default"].weight[:, 0],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lora_b = getattr(model.layers[1].self_attn, projection).lora_B["default"]
        assert torch.count_nonzero(lora_b.weight) == 0
