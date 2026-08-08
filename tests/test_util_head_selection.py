import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util import get_sensitive_heads_sorted_by_heatmap


def _results() -> dict:
    keys = {(0, 0): np.zeros(1), (0, 1): np.zeros(1), (0, 2): np.zeros(1)}
    return {
        "heatmap": np.asarray([[0.9, 0.4, 0.2]]),
        "white_emb": keys,
        "black_emb": keys,
        "elbow_score": 0.4,
    }


def test_default_elbow_score_filters_candidates() -> None:
    assert get_sensitive_heads_sorted_by_heatmap(_results()) == [(0, 0), (0, 1)]


def test_explicit_threshold_overrides_elbow_score() -> None:
    assert get_sensitive_heads_sorted_by_heatmap(_results(), min_score=0.8) == [(0, 0)]
