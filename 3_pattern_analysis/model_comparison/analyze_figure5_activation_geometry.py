#!/usr/bin/env python3
"""Build panel-h activation geometry for Base and PFairFT."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hook import get_last_token_indices_safe
from model_adapter import get_model_adapter
from prompt import add_yes_no_instruction, build_resume_prompt, format_prompt_for_model, resolve_model_type
from util import flip_race_in_text


GROUPS = ("Black", "White")
CONDITIONS = ("base", "pfairft")
GEOMETRY_SCHEMA_VERSION = 2


class PromptDataset(Dataset):
    def __init__(self, records: list[dict[str, str | int]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str | int]:
        return self.records[index]


def normalize_group(value: str) -> str:
    text = str(value).strip().lower()
    if text == "black":
        return "Black"
    if text == "white":
        return "White"
    raise ValueError(f"Unsupported sensitive group: {value!r}")


def opposite_group(group: str) -> str:
    return "White" if group == "Black" else "Black"


def load_resume_records(dataset_path: Path, ranking_csv: Path, sample_size: int) -> list[dict]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    with ranking_csv.open("r", encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))
    if sample_size <= 0 or len(ranking) < sample_size:
        raise ValueError(f"Need at least {sample_size} ranked Resume rows, found {len(ranking)}")

    records = []
    for row in ranking[:sample_size]:
        source_index = int(row["index"])
        item = data[source_index]
        group = normalize_group(item["race"])
        prompt = add_yes_no_instruction(
            build_resume_prompt(
                summary=item["summary"],
                category=item.get("category", ""),
                mode="summary_only",
            )
        )
        flipped = flip_race_in_text(prompt)
        pair_id = f"resume-{source_index}"
        records.extend(
            [
                {
                    "sample_id": f"{pair_id}-fact",
                    "matched_id": f"{pair_id}-counterfactual",
                    "pair_id": pair_id,
                    "group": group,
                    "decision_question_id": "",
                    "prompt": prompt,
                },
                {
                    "sample_id": f"{pair_id}-counterfactual",
                    "matched_id": f"{pair_id}-fact",
                    "pair_id": pair_id,
                    "group": opposite_group(group),
                    "decision_question_id": "",
                    "prompt": flipped,
                },
            ]
        )
    counts = {group: sum(row["group"] == group for row in records) for group in GROUPS}
    if counts != {"Black": sample_size, "White": sample_size}:
        raise ValueError(f"Resume counterfactual pairing is not balanced: {counts}")
    return records


def load_discrim_records(dataset_path: Path) -> list[dict]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = [
        {
            "sample_id": str(item["id"]),
            "matched_id": str(item["matched_id"]),
            "pair_id": "-".join(map(str, sorted((int(item["id"]), int(item["matched_id"]))))),
            "group": normalize_group(item["race"]),
            "decision_question_id": str(item["decision_question_id"]),
            "prompt": add_yes_no_instruction(item["prompt"]),
        }
        for item in data
    ]
    counts = {group: sum(row["group"] == group for row in records) for group in GROUPS}
    if len(records) != 2520 or counts != {"Black": 1260, "White": 1260}:
        raise ValueError(f"Unexpected Discrim-Eval population: rows={len(records)}, groups={counts}")
    return records


def load_heads(heads_dir: Path) -> tuple[list[tuple[int, int]], dict, dict[tuple[int, int], int]]:
    selected = json.loads((heads_dir / "selected_heads_elbow.json").read_text(encoding="utf-8"))
    heads = [(int(row["layer"]), int(row["head"])) for row in selected]
    if len(heads) != 52 or len(set(heads)) != 52:
        raise ValueError(f"Expected 52 unique selected heads, found {len(set(heads))}")
    with (heads_dir / "results.pkl").open("rb") as handle:
        results = pickle.load(handle)
    rank_array = np.asarray(results["rank_array"])
    ranks = {head: int(rank_array[head]) for head in heads}
    return heads, results, ranks


def directions_and_anchors(heads: list[tuple[int, int]], results: dict) -> tuple[np.ndarray, np.ndarray]:
    directions, anchors = [], []
    for head in heads:
        white = np.asarray(results["white_emb"][head], dtype=np.float64).reshape(-1)
        black = np.asarray(results["black_emb"][head], dtype=np.float64).reshape(-1)
        direction = white - black
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError(f"Degenerate sensitive direction for head {head}")
        direction /= norm
        directions.append(direction)
        anchors.append(float((white @ direction + black @ direction) / 2.0))
    matrix = np.stack(directions)
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6):
        raise ValueError("Sensitive directions are not unit normalized")
    return matrix, np.asarray(anchors, dtype=np.float64)


def load_model(base_model_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()
    model_type = resolve_model_type(
        "auto", model=model, tokenizer=tokenizer, model_path=base_model_path
    )
    architecture = get_model_adapter(model, model_type=model_type, model_path=base_model_path)
    return model, architecture, tokenizer, model_type


def collect_activations(
    model,
    architecture,
    tokenizer,
    model_type: str,
    records: list[dict],
    heads: list[tuple[int, int]],
    batch_size: int,
    description: str,
) -> np.ndarray:
    config = model.config
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size // num_heads)
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for output_index, (layer, head) in enumerate(heads):
        by_layer[layer].append((output_index, head))

    buffer: dict[int, torch.Tensor] = {}
    hooks = [
        architecture.register_head_activation_hook(layer, num_heads, head_dim, buffer)
        for layer in sorted(by_layer)
    ]
    output = np.empty((len(records), len(heads), head_dim), dtype=np.float32)
    dataloader = DataLoader(PromptDataset(records), batch_size=batch_size, shuffle=False)
    input_device = architecture.get_input_embedding_module().weight.device
    offset = 0
    try:
        for batch in tqdm(dataloader, desc=description):
            prompts = [format_prompt_for_model(value, model_type) for value in batch["prompt"]]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(input_device)
            last = get_last_token_indices_safe(
                inputs["input_ids"], inputs.get("attention_mask"), tokenizer
            )
            buffer.clear()
            with torch.no_grad():
                model(**inputs)
            size = len(prompts)
            for layer, layer_heads in by_layer.items():
                values = buffer[layer]
                rows = torch.arange(size, device=values.device)
                acts = values[rows, last.to(values.device), :, :].detach().cpu().float().numpy()
                for output_index, head in layer_heads:
                    output[offset : offset + size, output_index, :] = acts[:, head, :]
            offset += size
    finally:
        for hook in hooks:
            hook.remove()
    if offset != len(records) or not np.isfinite(output).all():
        raise ValueError(f"Incomplete or non-finite activation collection for {description}")
    return output


def group_metrics(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    black = values[groups == "Black"]
    white = values[groups == "White"]
    pooled_var = (float(np.var(black)) + float(np.var(white))) / 2.0
    separation = abs(float(np.mean(white)) - float(np.mean(black))) / np.sqrt(pooled_var + 1e-12)
    anchor_distance = (abs(float(np.mean(white))) + abs(float(np.mean(black)))) / 2.0
    return separation, anchor_distance


def orthogonal_centered(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered - np.outer(centered @ direction, direction)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    cross = first.T @ second
    numerator = float(np.sum(cross * cross))
    first_norm = float(np.linalg.norm(first.T @ first, ord="fro"))
    second_norm = float(np.linalg.norm(second.T @ second, ord="fro"))
    denominator = first_norm * second_norm
    return numerator / denominator if denominator > 1e-12 else 0.0


def preservation_metrics(
    base: np.ndarray, pfairft: np.ndarray, direction: np.ndarray
) -> dict[str, float]:
    base_center = np.mean(base, axis=0, keepdims=True)
    base_orthogonal = (base - base_center) - np.outer(
        (base - base_center) @ direction, direction
    )
    _, singular_values, vt = np.linalg.svd(base_orthogonal, full_matrices=False)
    components = vt[:2].T
    base_plane = base_orthogonal @ components
    pfairft_centered = pfairft - base_center
    pfairft_orthogonal = pfairft_centered - np.outer(
        pfairft_centered @ direction, direction
    )
    pfairft_plane = pfairft_orthogonal @ components
    base_variance = float(np.sum((base_plane - np.mean(base_plane, axis=0)) ** 2))
    pfairft_variance = float(
        np.sum((pfairft_plane - np.mean(pfairft_plane, axis=0)) ** 2)
    )
    variance_ratio = pfairft_variance / base_variance if base_variance > 1e-12 else 0.0
    variance_score = (
        float(np.exp(-abs(np.log(variance_ratio)))) if variance_ratio > 1e-12 else 0.0
    )
    base_distances = pdist(base_plane, metric="euclidean")
    pfairft_distances = pdist(pfairft_plane, metric="euclidean")
    if np.std(base_distances) <= 1e-12 or np.std(pfairft_distances) <= 1e-12:
        correlation = -1.0
    else:
        correlation = float(spearmanr(base_distances, pfairft_distances).statistic)
        if not np.isfinite(correlation):
            correlation = -1.0
    cka = linear_cka(
        base_plane - np.mean(base_plane, axis=0),
        pfairft_plane - np.mean(pfairft_plane, axis=0),
    )
    total_variance = float(np.sum(singular_values**2))
    base_plane_fraction = (
        float(np.sum(singular_values[:2] ** 2) / total_variance)
        if total_variance > 1e-12
        else 0.0
    )
    preservation_score = float((cka + variance_score + (correlation + 1.0) / 2.0) / 3.0)
    return {
        "orthogonal_linear_cka": cka,
        "orthogonal_variance_ratio": variance_ratio,
        "orthogonal_variance_score": variance_score,
        "orthogonal_distance_spearman": correlation,
        "orthogonal_preservation_score": preservation_score,
        "base_pc12_variance_fraction": base_plane_fraction,
    }


def display_plane_metrics(
    base: np.ndarray,
    pfairft: np.ndarray,
    direction: np.ndarray,
    anchor: float,
) -> dict[str, float]:
    center = np.mean(base, axis=0)
    centered = base - center
    orthogonal = centered - np.outer(centered @ direction, direction)
    _, _, vt = np.linalg.svd(orthogonal, full_matrices=False)
    pc1 = vt[0] - float(vt[0] @ direction) * direction
    pc1_norm = float(np.linalg.norm(pc1))
    if pc1_norm <= 1e-12:
        basis = np.eye(direction.size)
        candidates = basis - np.outer(basis @ direction, direction)
        pc1 = candidates[int(np.argmax(np.linalg.norm(candidates, axis=1)))]
        pc1_norm = float(np.linalg.norm(pc1))
    pc1 /= pc1_norm
    base_plane = np.column_stack((base @ direction - anchor, centered @ pc1))
    pfairft_plane = np.column_stack(
        (pfairft @ direction - anchor, (pfairft - center) @ pc1)
    )
    base_covariance = np.cov(base_plane, rowvar=False)
    base_scale = float(np.sqrt(np.trace(base_covariance)))
    center_shift = float(
        np.linalg.norm(np.mean(pfairft_plane, axis=0) - np.mean(base_plane, axis=0))
        / (base_scale + 1e-12)
    )
    base_variance = float(np.trace(base_covariance))
    pfairft_variance = float(np.trace(np.cov(pfairft_plane, rowvar=False)))
    variance_ratio = pfairft_variance / base_variance if base_variance > 1e-12 else 0.0
    return {
        "display_center_shift_standardized": center_shift,
        "display_variance_ratio": variance_ratio,
    }


def select_head(
    heads: list[tuple[int, int]],
    ranks: dict[tuple[int, int], int],
    directions: np.ndarray,
    anchors: np.ndarray,
    base: np.ndarray,
    pfairft: np.ndarray,
    groups: np.ndarray,
) -> tuple[int, list[dict]]:
    rows = []
    for index, (layer, head) in enumerate(heads):
        base_x = base[:, index, :] @ directions[index] - anchors[index]
        pfairft_x = pfairft[:, index, :] @ directions[index] - anchors[index]
        base_sep, base_anchor = group_metrics(base_x, groups)
        pfairft_sep, pfairft_anchor = group_metrics(pfairft_x, groups)
        preservation = preservation_metrics(
            base[:, index, :], pfairft[:, index, :], directions[index]
        )
        display = display_plane_metrics(
            base[:, index, :],
            pfairft[:, index, :],
            directions[index],
            float(anchors[index]),
        )
        separation_ratio = pfairft_sep / base_sep if base_sep > 1e-12 else np.inf
        anchor_ratio = pfairft_anchor / base_anchor if base_anchor > 1e-12 else np.inf
        eligible = separation_ratio <= 0.5 and anchor_ratio <= 0.5
        rows.append(
            {
                "layer": layer,
                "head": head,
                "sensitivity_rank": ranks[(layer, head)],
                "base_standardized_separation": base_sep,
                "pfairft_standardized_separation": pfairft_sep,
                "separation_reduction": base_sep - pfairft_sep,
                "base_anchor_distance": base_anchor,
                "pfairft_anchor_distance": pfairft_anchor,
                "anchor_distance_reduction": base_anchor - pfairft_anchor,
                "separation_ratio": separation_ratio,
                "anchor_distance_ratio": anchor_ratio,
                **preservation,
                **display,
                "eligible": int(eligible),
                "selected": 0,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise ValueError(
            "No selected head reduces both sensitive separation and fairness-anchor "
            "distance by at least 50%"
        )
    winner = sorted(
        eligible_rows,
        key=lambda row: (
            row["display_center_shift_standardized"],
            -row["orthogonal_preservation_score"],
            -row["orthogonal_linear_cka"],
            -row["orthogonal_distance_spearman"],
            -row["separation_reduction"],
            row["sensitivity_rank"],
            row["layer"],
            row["head"],
        ),
    )[0]
    winner["selected"] = 1
    selected_index = heads.index((winner["layer"], winner["head"]))
    rows.sort(key=lambda row: (0 if row["selected"] else 1, row["sensitivity_rank"]))
    return selected_index, rows


def geometry(
    base: np.ndarray,
    pfairft: np.ndarray,
    direction: np.ndarray,
    anchor: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    center = np.mean(base, axis=0)
    centered = base - center
    orthogonal = centered - np.outer(centered @ direction, direction)
    _, singular_values, vt = np.linalg.svd(orthogonal, full_matrices=False)
    components = []
    for row in vt[:2]:
        component = row - float(row @ direction) * direction
        for previous in components:
            component -= float(component @ previous) * previous
        component /= np.linalg.norm(component)
        largest = int(np.argmax(np.abs(component)))
        if component[largest] < 0:
            component *= -1.0
        components.append(component)
    pc1, pc2 = components
    total_variance = float(np.sum(singular_values**2))
    explained = (
        [float(value**2 / total_variance) for value in singular_values[:2]]
        if total_variance > 0
        else [0.0, 0.0]
    )
    projected = {}
    for label, values in (("base", base), ("pfairft", pfairft)):
        projected[label] = {
            "orthogonal_pc1": (values - center) @ pc1,
            "orthogonal_pc2": (values - center) @ pc2,
            "sensitive_residual": values @ direction - anchor,
        }
    metadata = {
        "base_center": center.tolist(),
        "orthogonal_pc1": pc1.tolist(),
        "orthogonal_pc2": pc2.tolist(),
        "explained_variance_ratio": explained,
        "pc1_pc2_explained_variance_ratio": float(sum(explained)),
        "direction_pc1_dot": float(direction @ pc1),
        "direction_pc2_dot": float(direction @ pc2),
        "pc1_pc2_dot": float(pc1 @ pc2),
    }
    return projected, metadata


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geometry_csv(
    path: Path,
    records: list[dict],
    projected: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = []
    for condition in CONDITIONS:
        coordinates = projected[condition]
        for index, record in enumerate(records):
            rows.append(
                {
                    "condition": condition,
                    "sample_id": record["sample_id"],
                    "matched_id": record["matched_id"],
                    "pair_id": record["pair_id"],
                    "group": record["group"],
                    "decision_question_id": record["decision_question_id"],
                    "orthogonal_pc1": float(coordinates["orthogonal_pc1"][index]),
                    "orthogonal_pc2": float(coordinates["orthogonal_pc2"][index]),
                    "sensitive_residual": float(coordinates["sensitive_residual"][index]),
                }
            )
    write_csv(
        path,
        [
            "condition",
            "sample_id",
            "matched_id",
            "pair_id",
            "group",
            "decision_question_id",
            "orthogonal_pc1",
            "orthogonal_pc2",
            "sensitive_residual",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--pfairft_adapter", required=True)
    parser.add_argument("--heads_dir", required=True)
    parser.add_argument("--resume_dataset", required=True)
    parser.add_argument("--resume_ranking_csv", required=True)
    parser.add_argument("--discrim_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heads, head_results, ranks = load_heads(Path(args.heads_dir))
    directions, anchors = directions_and_anchors(heads, head_results)
    resume_records = load_resume_records(
        Path(args.resume_dataset), Path(args.resume_ranking_csv), args.resume_sample_size
    )
    discrim_records = load_discrim_records(Path(args.discrim_dataset))

    activations: dict[str, dict[str, np.ndarray]] = {"resume": {}, "discrim": {}}
    for condition, adapter_path in (("base", None), ("pfairft", args.pfairft_adapter)):
        print(f"Loading {condition}: {adapter_path or args.base_model_path}")
        model, architecture, tokenizer, model_type = load_model(args.base_model_path, adapter_path)
        activations["resume"][condition] = collect_activations(
            model,
            architecture,
            tokenizer,
            model_type,
            resume_records,
            heads,
            args.batch_size,
            f"{condition} Resume",
        )
        activations["discrim"][condition] = collect_activations(
            model,
            architecture,
            tokenizer,
            model_type,
            discrim_records,
            heads,
            args.batch_size,
            f"{condition} Discrim-Eval",
        )
        del model, architecture, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resume_groups = np.asarray([row["group"] for row in resume_records])
    selected_index, scores = select_head(
        heads,
        ranks,
        directions,
        anchors,
        activations["resume"]["base"],
        activations["resume"]["pfairft"],
        resume_groups,
    )
    selected_head = heads[selected_index]
    write_csv(output / "head_scores.csv", list(scores[0]), scores)
    np.savez_compressed(
        output / "resume_candidate_activations.npz",
        base=activations["resume"]["base"],
        pfairft=activations["resume"]["pfairft"],
        layers=np.asarray([head[0] for head in heads], dtype=np.int16),
        heads=np.asarray([head[1] for head in heads], dtype=np.int16),
        group=resume_groups,
        sample_id=np.asarray([str(row["sample_id"]) for row in resume_records]),
    )

    geometry_metadata = {}
    for domain, records in (("resume", resume_records), ("discrim", discrim_records)):
        selected_activations = {
            condition: activations[domain][condition][:, selected_index, :]
            for condition in CONDITIONS
        }
        projected, domain_metadata = geometry(
            selected_activations["base"].astype(np.float64),
            selected_activations["pfairft"].astype(np.float64),
            directions[selected_index],
            float(anchors[selected_index]),
        )
        write_geometry_csv(output / f"{domain}_geometry.csv", records, projected)
        np.savez_compressed(
            output / f"{domain}_selected_activations.npz",
            base=selected_activations["base"],
            pfairft=selected_activations["pfairft"],
            sample_id=np.asarray([str(row["sample_id"]) for row in records]),
            matched_id=np.asarray([str(row["matched_id"]) for row in records]),
            pair_id=np.asarray([str(row["pair_id"]) for row in records]),
            group=np.asarray([str(row["group"]) for row in records]),
            decision_question_id=np.asarray(
                [str(row["decision_question_id"]) for row in records]
            ),
        )
        geometry_metadata[domain] = domain_metadata

    selected_row = next(row for row in scores if row["selected"])
    metadata = {
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "base_model_path": str(Path(args.base_model_path).resolve()),
        "pfairft_adapter": str(Path(args.pfairft_adapter).resolve()),
        "heads_dir": str(Path(args.heads_dir).resolve()),
        "head_selection_domain": "resume",
        "head_selection_rule": (
            "among Resume heads reducing both standardized sensitive-group separation "
            "and fairness-anchor distance by at least 50%, maximize information "
            "preservation by first minimizing standardized common-center drift in the "
            "shared d/orthogonal-PC1 display plane, then use centered linear CKA, "
            "plane-variance retention, pairwise-distance Spearman correlation, and "
            "sensitivity rank as successive tie-breakers"
        ),
        "selected_head": {
            "layer": selected_head[0],
            "head": selected_head[1],
            **{key: value for key, value in selected_row.items() if key not in {"layer", "head"}},
        },
        "sensitive_direction": directions[selected_index].tolist(),
        "fairness_anchor": float(anchors[selected_index]),
        "population": {
            "resume": {
                "rows_per_condition": len(resume_records),
                **{
                    group.lower(): sum(row["group"] == group for row in resume_records)
                    for group in GROUPS
                },
            },
            "discrim": {
                "rows_per_condition": len(discrim_records),
                **{
                    group.lower(): sum(row["group"] == group for row in discrim_records)
                    for group in GROUPS
                },
            },
        },
        "batch_size": args.batch_size,
        "geometry": geometry_metadata,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected Layer {selected_head[0]}, Head {selected_head[1]}")
    print(f"Saved activation geometry to {output}")


if __name__ == "__main__":
    main()
