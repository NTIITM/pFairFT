import csv
import importlib.util
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


table = load_module("table2_builder", "6_downstream_evaluation/build_table2_nine_models.py")
standard_mmlu = load_module("standard_mmlu", "6_downstream_evaluation/evaluate_mmlu_ce.py")
intervention_mmlu = load_module(
    "intervention_mmlu",
    "4_intervention_ablation/projection_intervention/evaluate_mmlu_intervention.py",
)
hooks = load_module("intervention_hooks", "src/hook.py")
model_adapter = load_module("model_adapter_under_test", "src/model_adapter.py")


def test_mmlu_prompt_is_identical_for_standard_and_intervention_ce():
    choices = ["one", "two", "three", "four"]
    assert standard_mmlu.build_mmlu_prompt("question", choices) == intervention_mmlu.build_mmlu_prompt(
        "question", choices
    )


def test_intervention_head_layout_uses_runtime_activation_shape(monkeypatch):
    monkeypatch.setattr(
        intervention_mmlu,
        "get_model_config",
        lambda _model: {"num_heads": 32, "head_dim": 64},
    )

    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return Inputs(input_ids=torch.ones((1, 2), dtype=torch.long))

    class Hook:
        def remove(self):
            pass

    class Adapter:
        def register_config_detection_hook(self, detected):
            detected.update(num_heads=16, head_dim=128)
            return Hook()

    class Model:
        def __call__(self, **_inputs):
            return object()

    assert intervention_mmlu._detect_head_config(
        Model(), Adapter(), Tokenizer(), "cpu", "llama", "prompt"
    ) == (16, 128)


def test_default_projection_hook_intervenes_at_all_teacher_forcing_positions():
    hook = hooks.make_intervention_hook_debias_projection(
        0,
        1,
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        None,
        None,
        1.0,
        2,
        2,
        use_std=False,
    )
    original = torch.tensor([[[9.0, 9.0, 1.0, 2.0], [8.0, 8.0, -3.0, 4.0]]])
    (projected,) = hook(None, (original,))
    assert torch.equal(projected[0, :, 0,], original[0, :, 0])
    assert torch.equal(projected[0, :, 1], original[0, :, 1])
    assert torch.equal(projected[0, :, 2], torch.zeros(2))
    assert torch.equal(projected[0, :, 3], original[0, :, 3])


def test_attention_output_projection_intervenes_at_all_teacher_forcing_positions():
    hook = model_adapter.make_attention_output_debias_projection_hook(
        0,
        0,
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        None,
        None,
        1.0,
        2,
        2,
        use_std=False,
    )
    original = torch.tensor([[[1.0, 2.0, 9.0, 9.0], [-3.0, 4.0, 8.0, 8.0]]])
    projected = hook(None, (), original)
    assert torch.equal(projected[0, :, 0], torch.zeros(2))
    assert torch.equal(projected[0, :, 1], original[0, :, 1])
    assert torch.equal(projected[0, :, 2:], original[0, :, 2:])


def test_intervention_scope_is_all_teacher_forcing_positions():
    assert intervention_mmlu.INTERVENTION_SCOPE == "all_teacher_forcing_positions"


def test_resume_gap_validates_rows_indices_and_probabilities(tmp_path):
    path = tmp_path / "resume.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "fact_p_yes", "cf_p_yes"])
        writer.writerow([1, 0.8, 0.2])
        writer.writerow([2, 0.4, 0.3])
    assert table.load_resume_gap(path, expected_rows=2) == pytest.approx(0.35)


def test_theoretical_counts_use_logical_heads_and_selected_layers(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 8),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 2),
            "base_model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.zeros(2, 8),
            "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.zeros(8, 2),
        },
        str(adapter),
    )
    selected = tmp_path / "heads.json"
    selected.write_text(json.dumps([{"layer": 0, "head": 1}, {"layer": 1, "head": 2}]))
    global_count, precise_count = table.theoretical_lora_counts(adapter, selected, 4)
    assert global_count == 64
    assert precise_count == 16


def test_display_mapping_matches_current_figure5_binding():
    assert table.METHODS[3:6] == (
        ("inference_time", "Inference Time"),
        ("pfairft_kl", "PFairFT-KL"),
        ("pfairft", "PFairFT"),
    )
