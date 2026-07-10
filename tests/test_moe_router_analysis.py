import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "3_pattern_analysis"
    / "mlp_analysis"
    / "analyze_moe_router_resume.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_moe_router_resume", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_router_metrics_support_logits_and_topk_tuples():
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    js, overlap = MODULE._router_metrics(logits, logits.clone(), top_k=2)
    np.testing.assert_allclose(js, 0.0, atol=1e-7)
    np.testing.assert_allclose(overlap, 1.0)

    indices = torch.tensor([[0, 2], [1, 2]])
    weights = torch.tensor([[0.7, 0.3], [0.8, 0.2]])
    js, overlap = MODULE._router_metrics(
        (indices, weights), (indices.clone(), weights.clone()), top_k=2
    )
    np.testing.assert_allclose(js, 0.0, atol=1e-7)
    np.testing.assert_allclose(overlap, 1.0)


def test_frozen_router_hook_replaces_only_last_token():
    key = (0, "mlp")
    fact = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    output = torch.zeros(2 * 4, 4)
    hook = MODULE._make_last_token_force_hook(
        key=key,
        fact_router={key: fact},
        fact_positions=torch.tensor([2, 1]),
        cf_positions=torch.tensor([3, 2]),
        batch_size=2,
        seq_len=4,
    )
    replaced = hook(None, (torch.zeros(2, 4, 8),), output).view(2, 4, 4)
    torch.testing.assert_close(replaced[0, 3], fact[0, 2])
    torch.testing.assert_close(replaced[1, 2], fact[1, 1])
    assert torch.count_nonzero(replaced[0, :3]) == 0
    assert torch.count_nonzero(replaced[1, [0, 1, 3]]) == 0
