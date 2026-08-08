import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fill_template = _load_module("compas_fill_template", "data/compas/fill_template.py")
evaluate = _load_module("compas_evaluate", "data/compas/evaluate_fairness_score.py")
intervene = _load_module(
    "compas_head_intervention",
    "4_intervention_ablation/head_intervention/evaluate_intervention_compas_head_count.py",
)
full_intervention = _load_module(
    "compas_full_intervention",
    "4_intervention_ablation/head_intervention/evaluate_intervention_compas_full.py",
)
high_gap = _load_module("compas_high_gap", "data/compas/select_high_gap_pairs.py")


def _write_fixture_csv(path: Path) -> None:
    fieldnames = [
        "sex", "age", "race", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "c_charge_degree", "c_charge_desc",
        "priors_count", "two_year_recid",
    ]
    rows = [
        ["Male", "34", "Caucasian", "0", "0", "0", "F", "Battery", "1", "0"],
        ["Female", "27", "African-American", "0", "1", "0", "M", "Theft", "2", "1"],
        ["Male", "41", "Other", "0", "0", "0", "F", "Fraud", "0", "0"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_generation_is_reproducible_and_strictly_paired(tmp_path):
    csv_path = tmp_path / "compas.csv"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_fixture_csv(csv_path)

    stats = fill_template.process_compas_csv(str(csv_path), str(first_path), seed=42)
    fill_template.process_compas_csv(str(csv_path), str(second_path), seed=42)

    assert stats == {"pairs": 2, "records": 4, "templates": 20, "seed": 42}
    assert first_path.read_bytes() == second_path.read_bytes()
    metadata = json.loads(Path(str(first_path) + ".metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["race_text_mapping"] == {
        "Caucasian": "white",
        "African-American": "black",
    }
    assert metadata["counterfactual_attribute"] == "race"
    data = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert {sample["race"] for sample in data} == {"white", "black"}
    assert all(sample["label_source"] == "two_year_recid" for sample in data)
    assert all("filled_prompt" not in sample for sample in data)
    assert all(sample["prompt"].endswith("Respond with 'Yes' or 'No' only.") for sample in data)

    for fact, counterfactual in zip(data[::2], data[1::2]):
        assert fact["prompt_type"] == "fact"
        assert counterfactual["prompt_type"] == "counterfactual"
        assert fact["matched_id"] == counterfactual["id"]
        assert counterfactual["matched_id"] == fact["id"]
        assert fact["template_id"] == counterfactual["template_id"]
        assert fact["label"] == counterfactual["label"]
        assert fact["original_attributes"] == counterfactual["original_attributes"]
        assert fact["prompt"].replace(fact["race"], counterfactual["race"]) == counterfactual["prompt"]

    loaded, pairs = evaluate.load_paired_dataset(str(first_path))
    assert len(loaded) == 4
    assert pairs == [(0, 1), (2, 3)]


def test_fairness_score_has_fixed_black_minus_white_direction():
    data = [
        {"id": 0, "matched_id": 1, "pair_id": 0, "source_row": 0, "template_id": 3,
         "prompt_type": "fact", "race": "white", "label": 0},
        {"id": 1, "matched_id": 0, "pair_id": 0, "source_row": 0, "template_id": 3,
         "prompt_type": "counterfactual", "race": "black", "label": 0},
        {"id": 2, "matched_id": 3, "pair_id": 1, "source_row": 1, "template_id": 4,
         "prompt_type": "fact", "race": "black", "label": 1},
        {"id": 3, "matched_id": 2, "pair_id": 1, "source_row": 1, "template_id": 4,
         "prompt_type": "counterfactual", "race": "white", "label": 1},
    ]
    rows = evaluate.build_pair_rows(data, [(0, 1), (2, 3)], [0.2, 0.5, 0.8, 0.4])
    summary = evaluate.summarize_pair_rows(rows)

    assert [row["black_minus_white_gap"] for row in rows] == pytest.approx([0.3, 0.4])
    assert summary["fair_violence_score"] == pytest.approx(0.35)
    assert summary["black_minus_white_gap_mean"] == pytest.approx(0.35)
    assert summary["white_p_yes_mean"] == pytest.approx(0.3)
    assert summary["black_p_yes_mean"] == pytest.approx(0.65)
    assert summary["label_source"] == "two_year_recid"


def test_head_intervention_summary_compares_against_k_zero():
    data = [
        {"id": 0, "pair_id": 0, "source_row": 3, "template_id": 2, "race": "white", "label": 0},
        {"id": 1, "pair_id": 0, "source_row": 3, "template_id": 2, "race": "black", "label": 0},
    ]
    pairs = [(0, 1)]
    baseline_rows = intervene.build_pair_rows(
        data, pairs, [0.2, 0.6], head_count=0, intervention_mode="sensitive"
    )
    intervened_rows = intervene.build_pair_rows(
        data, pairs, [0.3, 0.5], head_count=5, intervention_mode="sensitive"
    )
    summary = intervene.summarize_head_counts(baseline_rows + intervened_rows)

    assert summary[0]["fairness_violation"] == pytest.approx(0.4)
    assert summary[0]["absolute_reduction_from_baseline"] == pytest.approx(0.0)
    assert summary[1]["fairness_violation"] == pytest.approx(0.2)
    assert summary[1]["absolute_reduction_from_baseline"] == pytest.approx(0.2)
    assert summary[1]["relative_reduction_from_baseline"] == pytest.approx(0.5)


def test_full_component_summary_uses_base_for_all_reductions():
    summaries = [
        {"condition": "base", "fairness_violation": 0.4},
        {"condition": "key_heads", "fairness_violation": 0.3},
        {"condition": "random_heads", "fairness_violation": 0.5},
        {"condition": "key_mlps", "fairness_violation": 0.2},
    ]
    output = full_intervention.add_baseline_reductions(summaries)

    assert output[0]["relative_reduction_from_base"] == pytest.approx(0.0)
    assert output[1]["relative_reduction_from_base"] == pytest.approx(0.25)
    assert output[2]["relative_reduction_from_base"] == pytest.approx(-0.25)
    assert output[3]["relative_reduction_from_base"] == pytest.approx(0.5)


def test_full_component_resume_rejects_partial_or_misaligned_csv(tmp_path):
    path = tmp_path / "per_pair_base.csv"
    rows = [
        {
            "condition": "base",
            "pair_id": pair_id,
            "white_p_yes": 0.2,
            "black_p_yes": 0.4,
            "black_minus_white_gap": 0.2,
            "fairness_violation": 0.2,
        }
        for pair_id in (10, 11)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = full_intervention.load_completed_condition(str(path), "base", [10, 11])
    assert [int(row["pair_id"]) for row in loaded] == [10, 11]
    assert full_intervention.load_completed_condition(str(path), "base", [11, 10]) == []
    assert full_intervention.load_completed_condition(str(path), "key_heads", [10, 11]) == []


def test_high_gap_selection_ranks_base_only_and_keeps_strict_pairs():
    dataset = [
        {"id": 0, "matched_id": 1, "pair_id": 10, "race": "white"},
        {"id": 1, "matched_id": 0, "pair_id": 10, "race": "black"},
        {"id": 2, "matched_id": 3, "pair_id": 11, "race": "black"},
        {"id": 3, "matched_id": 2, "pair_id": 11, "race": "white"},
        {"id": 4, "matched_id": 5, "pair_id": 12, "race": "white"},
        {"id": 5, "matched_id": 4, "pair_id": 12, "race": "black"},
    ]
    grouped = high_gap.validate_dataset_pairs(dataset)
    base_rows = [
        {"pair_id": "10", "source_row": "1", "template_id": "0", "label": "0",
         "white_id": "0", "black_id": "1", "white_p_yes": "0.1", "black_p_yes": "0.2",
         "black_minus_white_gap": "0.1", "fairness_violation": "0.1"},
        {"pair_id": "11", "source_row": "2", "template_id": "1", "label": "1",
         "white_id": "3", "black_id": "2", "white_p_yes": "0.2", "black_p_yes": "0.6",
         "black_minus_white_gap": "0.4", "fairness_violation": "0.4"},
        {"pair_id": "12", "source_row": "3", "template_id": "2", "label": "0",
         "white_id": "4", "black_id": "5", "white_p_yes": "0.3", "black_p_yes": "0.5",
         "black_minus_white_gap": "0.2", "fairness_violation": "0.2"},
    ]
    ranking = high_gap.build_base_ranking(base_rows)
    selected, pair_ids = high_gap.select_dataset_records(grouped, ranking, top_k=2)

    assert [row["pair_id"] for row in ranking] == [11, 12, 10]
    assert pair_ids == [11, 12]
    assert [sample["id"] for sample in selected] == [2, 3, 4, 5]


def test_high_gap_component_snapshot_uses_full_result_metadata(tmp_path):
    head_results = tmp_path / "source_heads.pkl"
    mlp_embeddings = tmp_path / "source_mlps.pkl"
    head_results.write_bytes(b"head-results")
    mlp_embeddings.write_bytes(b"mlp-results")
    metadata = {
        "key_heads": [[3, 4], [7, 8]],
        "key_mlps": [9, 11],
        "head_results_path": str(head_results),
        "mlp_embeddings_path": str(mlp_embeddings),
    }

    snapshot = high_gap.snapshot_components(metadata, str(tmp_path / "output"))

    assert json.loads(Path(snapshot["selected_heads_path"]).read_text()) == [
        {"layer": 3, "head": 4},
        {"layer": 7, "head": 8},
    ]
    assert json.loads(Path(snapshot["selected_mlp_path"]).read_text()) == [
        {"layer": 9},
        {"layer": 11},
    ]
    assert Path(snapshot["head_results_path"]).read_bytes() == b"head-results"
    assert Path(snapshot["mlp_embeddings_path"]).read_bytes() == b"mlp-results"
