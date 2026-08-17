import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


activation = _load_module(
    "figure5_activation_geometry",
    "3_pattern_analysis/model_comparison/analyze_figure5_activation_geometry.py",
)
plot = _load_module("figure5_plot", "nmi_plot/figure5/plot_figure5.py")


def test_geometry_uses_two_base_fitted_components_orthogonal_to_sensitive_direction():
    rng = np.random.default_rng(42)
    direction = np.asarray([1.0, 0.0, 0.0])
    base = rng.normal(size=(40, 3))
    pfairft = rng.normal(size=(40, 3))

    projected, metadata = activation.geometry(base, pfairft, direction, anchor=0.25)

    assert metadata["direction_pc1_dot"] == pytest.approx(0.0, abs=1e-10)
    assert metadata["direction_pc2_dot"] == pytest.approx(0.0, abs=1e-10)
    assert metadata["pc1_pc2_dot"] == pytest.approx(0.0, abs=1e-10)
    assert np.asarray(metadata["orthogonal_pc1"]) @ direction == pytest.approx(0.0, abs=1e-10)
    assert np.asarray(metadata["orthogonal_pc2"]) @ direction == pytest.approx(0.0, abs=1e-10)
    for key in ("orthogonal_pc1", "orthogonal_pc2", "sensitive_residual"):
        assert projected["base"][key].shape == projected["pfairft"][key].shape == (40,)
    assert projected["base"]["sensitive_residual"] == pytest.approx(base[:, 0] - 0.25)
    center = np.asarray(metadata["base_center"])
    pc1 = np.asarray(metadata["orthogonal_pc1"])
    pc2 = np.asarray(metadata["orthogonal_pc2"])
    assert projected["pfairft"]["orthogonal_pc1"] == pytest.approx((pfairft - center) @ pc1)
    assert projected["pfairft"]["orthogonal_pc2"] == pytest.approx((pfairft - center) @ pc2)


def test_head_selection_requires_both_separation_and_anchor_improvement():
    heads = [(1, 1), (2, 2)]
    ranks = {(1, 1): 2, (2, 2): 1}
    directions = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    anchors = np.zeros(2)
    groups = np.asarray(["Black", "Black", "White", "White"])
    base = np.zeros((4, 2, 2))
    pfairft = np.zeros_like(base)
    base[:, 0, 0] = [-2.0, -1.0, 1.0, 2.0]
    pfairft[:, 0, 0] = [-0.3, 0.2, -0.2, 0.3]
    base[:, 1, 0] = [-1.0, -0.8, 0.8, 1.0]
    pfairft[:, 1, 0] = [-4.0, -3.0, -2.0, -1.0]

    selected, rows = activation.select_head(
        heads, ranks, directions, anchors, base, pfairft, groups
    )

    assert selected == 0
    assert next(row for row in rows if row["selected"])["eligible"] == 1
    assert next(row for row in rows if row["head"] == 2)["eligible"] == 0


def test_head_selection_prefers_orthogonal_information_preservation():
    rng = np.random.default_rng(7)
    heads = [(1, 1), (2, 2)]
    ranks = {(1, 1): 2, (2, 2): 1}
    directions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchors = np.zeros(2)
    groups = np.asarray(["Black"] * 20 + ["White"] * 20)
    base = rng.normal(size=(40, 2, 3))
    base[:20, :, 0] -= 2.0
    base[20:, :, 0] += 2.0
    pfairft = base.copy()
    pfairft[:, :, 0] = np.tile([-0.05, 0.05], (40, 1))
    pfairft[:, 1, 1:] = rng.normal(scale=0.03, size=(40, 2))

    selected, rows = activation.select_head(
        heads, ranks, directions, anchors, base, pfairft, groups
    )

    assert selected == 0
    selected_row = next(row for row in rows if row["selected"])
    assert selected_row["eligible"] == 1
    assert selected_row["orthogonal_preservation_score"] > next(
        row["orthogonal_preservation_score"] for row in rows if row["head"] == 2
    )


def test_head_selection_prioritizes_small_shared_plane_center_shift():
    rng = np.random.default_rng(11)
    heads = [(1, 1), (2, 2)]
    ranks = {(1, 1): 2, (2, 2): 1}
    directions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchors = np.zeros(2)
    groups = np.asarray(["Black"] * 20 + ["White"] * 20)
    base = rng.normal(scale=0.3, size=(40, 2, 3))
    base[:20, :, 0] -= 1.0
    base[20:, :, 0] += 1.0
    pfairft = base.copy()
    pfairft[:, :, 0] = np.tile([-0.02, 0.02], (40, 1))
    pfairft[:, 1, 1] += 2.0

    selected, rows = activation.select_head(
        heads, ranks, directions, anchors, base, pfairft, groups
    )

    assert selected == 0
    first = next(row for row in rows if row["head"] == 1)
    second = next(row for row in rows if row["head"] == 2)
    assert first["eligible"] == second["eligible"] == 1
    assert first["display_center_shift_standardized"] < second[
        "display_center_shift_standardized"
    ]


def test_scene_transitions_use_70_scene_means_and_fixed_base_boundaries():
    downstream = {name: {} for name in ("original", "global", "pfairft")}
    for qid in range(70):
        downstream["original"][qid] = {"mean": float(qid), "std": 0.0, "pairs": 18}
        downstream["global"][qid] = {"mean": float(qid) / 2.0, "std": 0.0, "pairs": 18}
        downstream["pfairft"][qid] = {"mean": 0.0, "std": 0.0, "pairs": 18}

    result = plot.build_scene_transitions(downstream)

    assert result["base_counts"] == {"High": 24, "Medium": 23, "Low": 23}
    assert result["thresholds"] == {"low_medium": 22.5, "medium_high": 45.5}
    assert len(result["rows"]) == 70
    assert sum(result["matrices"]["global"].values()) == 70
    assert sum(result["matrices"]["pfairft"].values()) == 70
    assert result["matrices"]["pfairft"][("High", "Low")] == 24
    assert result["matrices"]["pfairft"][("Medium", "Low")] == 23
    assert result["matrices"]["pfairft"][("Low", "Low")] == 23


def test_sankey_segments_fill_node_interval_exactly():
    interval = (0.2, 0.2 + 70 * 0.0095)
    segments = plot._segments(
        interval, {"High": 24, "Medium": 23, "Low": 23}
    )

    assert segments["Low"][0] == pytest.approx(interval[0])
    assert segments["High"][1] == pytest.approx(interval[1])
