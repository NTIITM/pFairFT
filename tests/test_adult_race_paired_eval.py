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


fill_template = _load_module(
    "adult_fill_template", "data/adult_datasets/fill_template.py"
)
evaluate = _load_module(
    "adult_evaluate", "data/adult_datasets/evaluate_fairness_score.py"
)
select = _load_module(
    "adult_select", "data/adult_datasets/select_high_gap_pairs.py"
)
intervene = _load_module(
    "adult_intervene",
    "4_intervention_ablation/head_intervention/evaluate_intervention_adult_race.py",
)
merge = _load_module(
    "adult_merge", "data/adult_datasets/merge_baseline_shards.py"
)


FIELDNAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def _write_fixture(path: Path) -> None:
    rows = [
        ["39", "State-gov", "1", "Bachelors", "13", "Never-married", "Clerk", "Not-in-family", "White", "Male", "0", "0", "40", "United-States", "<=50K"],
        ["28", "Private", "2", "Masters", "14", "Married", "Engineer", "Wife", "Black", "Female", "0", "0", "45", "Canada", ">50K"],
        ["31", "Private", "3", "HS-grad", "9", "Single", "Sales", "Own-child", "Asian-Pac-Islander", "Female", "0", "0", "20", "India", "<=50K"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)


def test_generation_is_reproducible_strict_and_race_only(tmp_path):
    csv_path = tmp_path / "adult.csv"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_fixture(csv_path)

    stats = fill_template.process_adult_csv(str(csv_path), str(first_path), seed=42)
    fill_template.process_adult_csv(str(csv_path), str(second_path), seed=42)

    assert stats == {
        "source_rows": 3,
        "eligible_rows": 2,
        "excluded_rows": 1,
        "pairs": 2,
        "records": 4,
        "templates": 10,
        "seed": 42,
    }
    assert first_path.read_bytes() == second_path.read_bytes()
    metadata = json.loads(Path(str(first_path) + ".metadata.json").read_text())
    assert metadata["counterfactual_attribute"] == "race"
    assert metadata["evaluation_protocol"] == "yes_no_income_gt_50k_v1"
    assert metadata["excluded_race_counts"] == {"Asian-Pac-Islander": 1}
    data = json.loads(first_path.read_text())
    assert {sample["race"] for sample in data} == {"white", "black"}
    assert all("answer" not in sample for sample in data)

    for fact, counterfactual in zip(data[::2], data[1::2]):
        assert fact["prompt_type"] == "fact"
        assert counterfactual["prompt_type"] == "counterfactual"
        assert fact["matched_id"] == counterfactual["id"]
        assert counterfactual["matched_id"] == fact["id"]
        assert fact["template_id"] == counterfactual["template_id"]
        assert fact["label"] == counterfactual["label"]
        assert fact["original_attributes"] == counterfactual["original_attributes"]
        fact_race = fact["race"].title()
        counterfactual_race = counterfactual["race"].title()
        assert fact["prompt"].replace(fact_race, counterfactual_race) == counterfactual["prompt"]
        assert fact["prompt"].endswith("Respond with 'Yes' or 'No' only.")

    loaded, pairs = evaluate.load_paired_dataset(str(first_path))
    assert len(loaded) == 4
    assert pairs == [(0, 1), (2, 3)]
    sliced, sliced_pairs = evaluate.load_paired_dataset(
        str(first_path), pair_start=1, pair_end=2
    )
    assert [sample["id"] for sample in sliced] == [2, 3]
    assert sliced_pairs == [(2, 3)]


def test_fairness_violation_uses_fixed_black_minus_white_p_yes():
    data = [
        {"id": 0, "matched_id": 1, "pair_id": 0, "source_row": 0, "template_id": 1, "prompt_type": "fact", "race": "white", "label": 0},
        {"id": 1, "matched_id": 0, "pair_id": 0, "source_row": 0, "template_id": 1, "prompt_type": "counterfactual", "race": "black", "label": 0},
        {"id": 2, "matched_id": 3, "pair_id": 1, "source_row": 1, "template_id": 2, "prompt_type": "fact", "race": "black", "label": 1},
        {"id": 3, "matched_id": 2, "pair_id": 1, "source_row": 1, "template_id": 2, "prompt_type": "counterfactual", "race": "white", "label": 1},
    ]
    rows = evaluate.build_pair_rows(data, [(0, 1), (2, 3)], [0.2, 0.5, 0.8, 0.4])
    summary = evaluate.summarize_pair_rows(rows)

    assert [row["black_minus_white_gap"] for row in rows] == pytest.approx([0.3, 0.4])
    assert [row["fairness_violation"] for row in rows] == pytest.approx([0.3, 0.4])
    assert summary["fairness_violation"] == pytest.approx(0.35)
    assert summary["white_p_yes_mean"] == pytest.approx(0.3)
    assert summary["black_p_yes_mean"] == pytest.approx(0.65)


def test_baseline_ranking_is_stable_and_intervention_summary_uses_base():
    base_rows = [
        {"condition": "base", "pair_id": "10", "source_row": "1", "template_id": "0", "label": "0", "white_id": "0", "black_id": "1", "white_p_yes": "0.1", "black_p_yes": "0.3", "black_minus_white_gap": "0.2", "fairness_violation": "0.2"},
        {"condition": "base", "pair_id": "11", "source_row": "2", "template_id": "1", "label": "1", "white_id": "2", "black_id": "3", "white_p_yes": "0.2", "black_p_yes": "0.6", "black_minus_white_gap": "0.4", "fairness_violation": "0.4"},
        {"condition": "base", "pair_id": "9", "source_row": "3", "template_id": "2", "label": "0", "white_id": "4", "black_id": "5", "white_p_yes": "0.2", "black_p_yes": "0.4", "black_minus_white_gap": "0.2", "fairness_violation": "0.2"},
    ]
    ranking = select.build_base_ranking(base_rows)
    assert [row["pair_id"] for row in ranking] == [11, 9, 10]

    summaries = [
        {"condition": "base", "fairness_violation": 0.4},
        {"condition": "key_heads", "fairness_violation": 0.3},
        {"condition": "random_heads", "fairness_violation": 0.5},
        {"condition": "key_mlps", "fairness_violation": 0.2},
    ]
    output = intervene.add_baseline_reductions(summaries)
    assert output[1]["relative_reduction_from_base"] == pytest.approx(0.25)
    assert output[2]["relative_reduction_from_base"] == pytest.approx(-0.25)
    assert output[3]["relative_reduction_from_base"] == pytest.approx(0.5)
    assert intervene._head_counts(10) == [0, 5, 10]
    assert intervene._head_counts(31) == [0, 5, 10, 15, 20, 25]


def test_shard_merge_accepts_requested_model_type_for_resumed_checkpoint(tmp_path):
    shard_dirs = []
    for pair_id, model_type in ((0, None), (1, "olmoe")):
        shard_dir = tmp_path / f"shard_{pair_id}"
        shard_dir.mkdir()
        metadata = {
            "status": "complete",
            "dataset_path": "/data/adult.json",
            "model_path": "/models/olmoe",
            "model_name": "OLMoE",
            "requested_model_type": "olmoe",
            "evaluation_protocol": "yes_no_income_gt_50k_v1",
            "batch_size": 1,
            "pairs": 1,
            "pair_start": pair_id,
            "pair_end": pair_id + 1,
        }
        if model_type is not None:
            metadata["model_type"] = model_type
        (shard_dir / "metadata.json").write_text(json.dumps(metadata))
        with (shard_dir / "per_pair_base.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pair_id",
                    "label",
                    "white_p_yes",
                    "black_p_yes",
                    "black_minus_white_gap",
                    "fairness_violation",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "label": pair_id,
                    "white_p_yes": 0.1,
                    "black_p_yes": 0.2,
                    "black_minus_white_gap": 0.1,
                    "fairness_violation": 0.1,
                }
            )
        shard_dirs.append(str(shard_dir))

    output_dir = tmp_path / "merged"
    metadata = merge.merge_shards(shard_dirs, str(output_dir), expected_pairs=2)

    assert metadata["status"] == "complete"
    assert metadata["model_type"] == "olmoe"
    assert metadata["shard_ranges"] == [[0, 1], [1, 2]]
