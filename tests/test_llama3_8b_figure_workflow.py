from __future__ import annotations

import subprocess
import json
import pickle
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_llama3_8b_figures.sh"
PYTHON = "/home/common1/hwluo/anaconda3/envs/GRPOV/bin/python"


def run_driver(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(DRIVER), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_scripts_directory_has_one_active_entrypoint() -> None:
    entries = sorted(path.name for path in (ROOT / "scripts").iterdir())
    assert entries == ["run_llama3_8b_figures.sh"]
    assert DRIVER.stat().st_mode & 0o111


def test_stage_is_required_and_range_is_forward() -> None:
    missing = run_driver()
    assert missing.returncode == 2
    assert "--stage NAME" in missing.stdout

    reversed_range = run_driver("--stage", "figure4-figure2", "--dry-run")
    assert reversed_range.returncode == 2
    assert "stage range must be forward" in reversed_range.stdout


def test_later_stage_fails_without_validated_predecessors(tmp_path: Path) -> None:
    result = run_driver(
        "--stage",
        "figure2",
        "--dry-run",
        "--python",
        PYTHON,
        "--model-dir",
        str(tmp_path / "model"),
        "--result-root",
        str(tmp_path / "results"),
    )
    assert result.returncode == 2
    assert "figure2 requires a validated figure1 stage" in result.stdout


def test_complete_dry_run_has_five_figures_and_fixed_method_binding(tmp_path: Path) -> None:
    result = run_driver(
        "--stage",
        "figure1-figure5",
        "--dry-run",
        "--python",
        PYTHON,
        "--model-dir",
        str(tmp_path / "model"),
        "--result-root",
        str(tmp_path / "results"),
    )
    assert result.returncode == 0, result.stdout
    for number in range(1, 6):
        assert f"===== Running figure{number} =====" in result.stdout
    assert "python-snippet modelscope.snapshot_download" in result.stdout
    assert "--loss_type fairness_kl " in result.stdout
    assert "--loss_type fairness_kl_ce " in result.stdout
    assert "--loss_type fairness_ce " in result.stdout
    assert "--target_qids 40 12 94" in result.stdout


def test_public_method_mapping_is_not_swapped() -> None:
    plot_source = (ROOT / "nmi_plot/figure5/plot_figure5.py").read_text(encoding="utf-8")
    appendix_source = (ROOT / "nmi_plot/figure5/make_core_appendix.py").read_text(
        encoding="utf-8"
    )
    manifest_source = (ROOT / "nmi_plot/figure5/prepare_figure5_data.py").read_text(
        encoding="utf-8"
    )
    assert '("pfairft", "PFairFT", "tab:green")' in plot_source
    assert '("pfairft_kl", "PFairFT-KL", "tab:orange")' in plot_source
    assert '("pfairft_ce", "PFairFT-CE", "tab:cyan")' in plot_source
    assert '"PFairFT": "fairness_kl"' in manifest_source
    assert '"PFairFT-KL": "fairness_kl_ce"' in manifest_source
    assert '"PFairFT-CE": "fairness_ce"' in manifest_source
    legacy_pfairft = '"pfairft": downstream / "discrim_pkfair_kl_pkfair_3epoch_fresh.csv"'
    legacy_pfairft_kl = '"pfairft_kl": downstream / "discrim_pkfair_pkfair_3epoch_fresh.csv"'
    for source in (plot_source, appendix_source):
        assert legacy_pfairft in source
        assert legacy_pfairft_kl in source
    assert 'else "discrim_pkfair_kl_pkfair_3epoch_fresh.csv"' in manifest_source
    assert 'else "discrim_pkfair_pkfair_3epoch_fresh.csv"' in manifest_source


def test_llama_mlp_metadata_uses_dense_block_surface() -> None:
    metadata_writers = [
        "2_component_identification/analyze_race_sensitive_MLPs.py",
        "4_intervention_ablation/head_intervention/evaluate_intervention_compas_full.py",
        "4_intervention_ablation/mlp_intervention/collect_race_mean_MLPs_resume.py",
        "4_intervention_ablation/mlp_intervention/evaluate_intervention_MLP_discrim_eval.py",
        "4_intervention_ablation/mlp_intervention/evaluate_intervention_MLP_resume.py",
    ]
    for relative_path in metadata_writers:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert 'else "dense_mlp_block_output"' in source, relative_path


def test_single_model_figure1_renderer_writes_nonblank_outputs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    heads_dir = root / "figure1/components/heads"
    mlp_dir = root / "figure1/components/mlps/identification"
    selected_dir = root / "figure1/components/mlps/selected"
    heads_dir.mkdir(parents=True)
    mlp_dir.mkdir(parents=True)
    selected_dir.mkdir(parents=True)

    heatmap = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float64).reshape(32, 32)
    with (heads_dir / "results.pkl").open("wb") as handle:
        pickle.dump({"heatmap": heatmap, "elbow_idx": 4, "elbow_rank": 5}, handle)
    with (mlp_dir / "results_mlp.pkl").open("wb") as handle:
        pickle.dump({"layer_kl_scores": np.linspace(0.0, 0.3, 32)}, handle)
    (heads_dir / "selected_heads_elbow.json").write_text(
        json.dumps([{"layer": 31, "head": head} for head in range(5)]),
        encoding="utf-8",
    )
    (selected_dir / "selected_mlp_layers_elbow.json").write_text(
        json.dumps([{"layer": 30}, {"layer": 31}]), encoding="utf-8"
    )

    output = root / "figure1/figures"
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "nmi_plot/llama3_8b/plot_figures.py"),
            "--figure",
            "1",
            "--result-root",
            str(root),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "figure1.pdf").stat().st_size > 0
    image = mpimg.imread(output / "figure1.png")
    assert image.size > 0 and float(np.std(image)) > 0.01
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure"] == 1
    assert manifest["selected_head_count"] == 5
    assert len(manifest["sources"]) == 4
