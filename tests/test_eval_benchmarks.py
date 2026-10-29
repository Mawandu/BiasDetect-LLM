"""Tests de experiments/eval_benchmarks.py sur des donnees synthetiques (aucun reseau)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.eval.metrics import pairwise_accuracy
from experiments import eval_benchmarks as eb


DATASETS = ("stereoset_test", "crows", "sbic")
P_SYSTEMS = ("P_fusion_minilm", "P_neu_minilm", "P_sym")
ALL_SYSTEMS = ("B0_majority",) + P_SYSTEMS
P_SEEDS = (13, 21, 42, 7, 99)


def _frame(dataset, seed, n_pairs=8):
    rows = []
    for group in range(n_pairs):
        if dataset == "sbic":
            group_id = f"sbic::{group}"
            pair_id = None
            for label in (0, 1):
                rows.append({
                    "text": f"sbic text {group} {label}",
                    "label": label,
                    "category": "none",
                    "group_id": group_id,
                    "source": "sbic",
                })
        else:
            group_id = (
                f"crows::{group}" if dataset == "crows"
                else f"stereoset_test::{group // 2}"
            )
            pair_id = f"{dataset}::p{group}"
            for label in (0, 1):
                text = f"{dataset} text {group} {label}"
                if dataset == "stereoset_test":
                    text += f" seed {seed}"
                rows.append({
                    "text": text,
                    "label": label,
                    "category": "origin",
                    "group_id": group_id,
                    "pair_id": pair_id,
                    "source": dataset,
                })
    return pd.DataFrame(rows)


def _write_preds(tmp_path, systems=ALL_SYSTEMS, p_seeds=P_SEEDS, n_pairs=8,
                 seed=0):
    preds = tmp_path / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for system in systems:
        seeds = p_seeds if system.startswith("P_") else (13,)
        for dataset in DATASETS:
            for eval_seed in seeds:
                frame = _frame(dataset, eval_seed, n_pairs=n_pairs)
                if system == "P_fusion_minilm":
                    scores = frame["label"].to_numpy(dtype=float) * 0.8 + 0.1
                else:
                    scores = rng.normal(size=len(frame))
                frame = frame.assign(score=scores, threshold=0.5)
                frame.to_csv(
                    preds / f"{system}__{dataset}__s{eval_seed}.csv",
                    index=False,
                )
    return preds


def _three_pair_frame():
    return pd.DataFrame({
        "label": [0, 1, 0, 1, 0, 1],
        "group_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
        "pair_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
    })


def test_pair_accuracy_weights_duplicate_groups():
    frame = _three_pair_frame()
    pairs = eb._pair_indices(frame)
    scores = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.1])
    counts = np.array([2, 2, 1, 1, 0, 0])

    assert eb._pair_accuracy_selected(pairs, scores, counts) == pytest.approx(
        2 / 3
    )


def test_pair_accuracy_matches_reference_without_duplicates():
    frame = _three_pair_frame()
    pairs = eb._pair_indices(frame)
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    counts = np.ones(len(frame), dtype=int)

    assert eb._pair_accuracy_selected(
        pairs, scores, counts
    ) == pytest.approx(
        pairwise_accuracy(frame.assign(group_id=frame["pair_id"]), scores)
    )


def test_pair_accuracy_matches_reference_with_ties():
    frame = _three_pair_frame()
    pairs = eb._pair_indices(frame)
    scores = np.array([0.5, 0.5, 0.2, 0.8, 0.7, 0.3])
    counts = np.ones(len(frame), dtype=int)

    assert eb._pair_accuracy_selected(
        pairs, scores, counts
    ) == pytest.approx(
        pairwise_accuracy(frame.assign(group_id=frame["pair_id"]), scores)
    )


def test_auroc_and_macro_f1_match_sklearn_with_ties():
    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 100)
    scores = rng.integers(0, 4, size=len(y)).astype(float)

    assert eb._auroc(y, scores) == pytest.approx(roc_auc_score(y, scores))
    predictions = scores >= 2
    assert eb._macro_f1(y, predictions) == pytest.approx(
        f1_score(y, predictions, average="macro")
    )


def test_compute_metrics_perfect_auroc_and_pair_accuracy():
    frame = pd.DataFrame({
        "label": [0, 0, 1, 1],
        "group_id": ["g1", "g1", "g2", "g2"],
    })
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    values = eb.compute_metrics(frame, scores, 0.5)

    assert values["auroc"] == pytest.approx(1.0)
    assert values["macro_f1"] == pytest.approx(1.0)
    assert "pair_acc" not in values

    pairs = pd.DataFrame({
        "label": [0, 1, 0, 1],
        "group_id": ["g1", "g1", "g2", "g2"],
        "pair_id": ["p1", "p1", "p2", "p2"],
    })
    pair_values = eb.compute_metrics(
        pairs, np.array([0.1, 0.9, 0.8, 0.2]), 0.5
    )
    assert pair_values["pair_acc"] == pytest.approx(0.5)


def test_random_scores_auroc_near_half():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "label": np.array([0, 1] * 200),
        "group_id": [f"g{i}" for i in range(400)],
    })
    values = eb.compute_metrics(frame, rng.normal(size=400), 0.5)

    assert abs(values["auroc"] - 0.5) < 0.1


def test_run_outputs_info_and_holm(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=8)
    out = tmp_path / "out"

    metrics, paired, info = eb.run(preds, out, n_boot=10, seed=13)

    assert (out / "benchmark_metrics.csv").exists()
    assert (out / "benchmark_paired.csv").exists()
    info_files = list(out.glob("benchmark_eval_INFO_*.json"))
    assert len(info_files) == 1
    assert json.loads(info_files[0].read_text()) == info

    assert {
        "system", "dataset", "seed", "metric", "estimate", "ci_low",
        "ci_high", "sd_across_seeds", "n", "n_clusters", "n_seeds",
        "n_boot", "n_boot_rejected",
    } <= set(metrics.columns)

    crows_fusion = metrics[
        (metrics["system"] == "P_fusion_minilm")
        & (metrics["dataset"] == "crows")
        & (metrics["metric"] == "auroc")
        & (metrics["seed"].astype(str) == "mean")
    ]
    assert len(crows_fusion) == 1
    assert float(crows_fusion.iloc[0]["estimate"]) == pytest.approx(1.0)
    assert int(crows_fusion.iloc[0]["n_seeds"]) == len(P_SEEDS)
    assert np.isfinite(float(crows_fusion.iloc[0]["sd_across_seeds"]))

    crows_seed = metrics[
        (metrics["system"] == "P_fusion_minilm")
        & (metrics["dataset"] == "crows")
        & (metrics["metric"] == "auroc")
        & (metrics["seed"].astype(str) == "13")
    ]
    assert len(crows_seed) == 1
    assert float(crows_fusion.iloc[0]["n"]) == float(crows_seed.iloc[0]["n"])
    assert float(
        crows_fusion.iloc[0]["n_clusters"]
    ) == float(crows_seed.iloc[0]["n_clusters"])

    b0_crows = metrics[
        (metrics["system"] == "B0_majority") & (metrics["dataset"] == "crows")
    ]
    assert set(b0_crows["seed"].astype(str)) == {"13"}

    sbic_metrics = set(metrics.loc[metrics["dataset"] == "sbic", "metric"])
    assert "pair_acc" not in sbic_metrics

    assert set(paired["metric"]) == {"auroc", "pair_acc"}
    auroc_pairs = paired[paired["metric"] == "auroc"]
    assert len(auroc_pairs) == 6
    assert auroc_pairs["p_holm"].notna().all()
    assert (auroc_pairs["p"] > 0).all()

    pair_pairs = paired[paired["metric"] == "pair_acc"]
    assert set(pair_pairs["dataset"]) == {"stereoset_test", "crows"}
    assert pair_pairs["p_holm"].isna().all()

    assert {
        "versions", "seed", "n_boot", "git_commit", "inputs",
        "identical_items_across_seeds",
    } <= set(info)
    assert "approx_independent_seeds" not in info
    assert "approx_independent_seeds_by_dataset" not in info
    assert info["seed"] == 13
    assert info["n_boot"] == 10
    assert info["identical_items_across_seeds"]["stereoset_test"] is False
    assert info["identical_items_across_seeds"]["crows"] is True
    assert info["identical_items_across_seeds"]["sbic"] is True
    assert info["n_inputs"] == len(ALL_SYSTEMS) * len(DATASETS) + (
        len(P_SYSTEMS) * len(DATASETS) * (len(P_SEEDS) - 1)
    )
    assert len(info["inputs"]) == info["n_inputs"]


def test_one_bootstrap_draw_per_replicate_and_dataset(monkeypatch, tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    real = eb.bootstrap_prompt_ids
    calls = []

    def wrapped(prompt_ids, rng):
        result = real(prompt_ids, rng)
        calls.append(tuple(pd.unique(prompt_ids)))
        return result

    monkeypatch.setattr(eb, "bootstrap_prompt_ids", wrapped)
    eb.run(preds, tmp_path / "out", n_boot=4, seed=13)

    counts = {"crows": 0, "sbic": 0, "stereoset_test": 0}
    for prompt_ids in calls:
        counts[str(prompt_ids[0]).split("::")[0]] += 1

    assert counts["crows"] == 4
    assert counts["sbic"] == 4
    assert counts["stereoset_test"] == 4


def test_same_draw_multiplicity_for_each_stereoset_seed(monkeypatch, tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    real = eb.draw_cluster_indices
    calls = []

    def wrapped(drawn_prompts, groups):
        result = real(drawn_prompts, groups)
        calls.append((np.asarray(drawn_prompts).copy(), groups, result))
        return result

    monkeypatch.setattr(eb, "draw_cluster_indices", wrapped)
    eb.run(preds, tmp_path / "out", n_boot=3, seed=13)

    stereo_draws = [
        drawn for drawn, groups, _ in calls
        if str(next(iter(groups))).startswith("stereoset_test::")
    ]
    assert len(stereo_draws) == 3 * len(P_SEEDS)
    for replicate in range(3):
        chunk = stereo_draws[
            replicate * len(P_SEEDS):(replicate + 1) * len(P_SEEDS)
        ]
        assert all(np.array_equal(chunk[0], drawn) for drawn in chunk[1:])


def test_draw_cluster_indices_keeps_whole_groups(monkeypatch, tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    real = eb.draw_cluster_indices
    calls = []

    def wrapped(drawn_prompts, groups):
        result = real(drawn_prompts, groups)
        calls.append((groups, result))
        return result

    monkeypatch.setattr(eb, "draw_cluster_indices", wrapped)
    eb.run(preds, tmp_path / "out", n_boot=3, seed=13)

    assert calls
    for groups, result in calls:
        selected = set(result.tolist())
        for rows in groups.values():
            rows = set(rows.tolist())
            assert rows <= selected or rows.isdisjoint(selected)


def test_pair_id_group_id_mismatch_errors(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    path = preds / "P_sym__crows__s13.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "group_id"] = "crows::999"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="pair_id"):
        eb.run(preds, tmp_path / "out", n_boot=5, seed=13)


def test_outputs_refuse_overwrite_before_evaluate(monkeypatch, tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    out = tmp_path / "out"
    out.mkdir()
    (out / "benchmark_metrics.csv").write_text("existing\n")
    called = []
    real = eb.evaluate

    def wrapped(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(eb, "evaluate", wrapped)
    with pytest.raises(FileExistsError, match="benchmark_metrics.csv"):
        eb.run(preds, out, n_boot=5, seed=13)
    assert not called


def test_overwrite_allows_existing_outputs(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=6)
    out = tmp_path / "out"

    eb.run(preds, out, n_boot=5, seed=13)
    eb.run(preds, out, n_boot=5, seed=13, overwrite=True)

    assert (out / "benchmark_metrics.csv").exists()
    assert (out / "benchmark_paired.csv").exists()


def test_expect_seeds_rejects_unexpected_p_seeds(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=4, p_seeds=(13, 21))

    with pytest.raises(ValueError, match="expect-seeds"):
        eb.run(preds, tmp_path / "out", n_boot=3, seed=13)


def test_expect_seeds_option_accepts_matching_p_seeds(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=4, p_seeds=(13, 21))

    metrics, _, info = eb.run(
        preds, tmp_path / "out", n_boot=3, seed=13,
        expect_seeds=(13, 21),
    )

    assert len(metrics) > 0
    assert info["seeds_by_system"]["P_sym"] == [13, 21]


def test_p_system_seed_mismatch_errors(tmp_path):
    preds = _write_preds(tmp_path, n_pairs=4)
    for dataset in DATASETS:
        (preds / f"P_sym__{dataset}__s99.csv").unlink()

    with pytest.raises(ValueError, match="seeds incohérentes entre les systèmes P"):
        eb.run(preds, tmp_path / "out", n_boot=3, seed=13)


def test_bootstrap_p_value_never_zero():
    assert eb._bootstrap_p_value(np.array([0.1, 0.2, 0.3])) > 0
    assert eb._bootstrap_p_value(np.array([-0.1, -0.2, -0.3])) > 0
    assert eb._bootstrap_p_value(np.array([0.0, 0.0])) == pytest.approx(1.0)
