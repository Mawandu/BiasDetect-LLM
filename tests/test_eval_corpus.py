"""Tests de experiments/eval_corpus.py sur des donnees synthetiques (aucun reseau)."""
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import eval_corpus as ec


def _write_corpus(tmp_path, n_prompts=24, per_prompt=8, labels=None,
                  core=True, refusal=False, n_words=20):
    n = n_prompts * per_prompt
    if labels is None:
        labels = np.array([0, 1] * (n // 2), dtype=int)
    labels = np.asarray(labels)
    corpus = pd.DataFrame({
        "id": [f"id{i}" for i in range(n)],
        "prompt_id": [f"p{i // per_prompt}" for i in range(n)],
        "is_core": core,
        "refusal": refusal,
        "gold_raw": labels,
        "gold_reviewed": labels,
        "A1_round1": labels,
        "A2_round1": labels,
        "n_words": n_words,
        "text": ["some text"] * n,
    })
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / "biasdetect_corpus.csv"
    corpus.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (corpus_dir / "SHA256SUMS").write_text(f"{digest}  biasdetect_corpus.csv\n")
    return corpus, path


def _write_preds(tmp_path, corpus, corpus_path,
                 systems=("P_fusion_minilm", "P_neu_minilm", "P_sym"),
                 seeds=(13,), rng_seed=0, score_scale=1.0, score_shift=0.0):
    preds = tmp_path / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    for system in systems:
        for mode in ("full", "sentmean"):
            for seed in seeds:
                scores = rng.normal(size=len(corpus)) * score_scale + score_shift
                pd.DataFrame({
                    "id": corpus["id"],
                    "prompt_id": corpus["prompt_id"],
                    "score": scores,
                    "threshold": 0.5,
                }).to_csv(preds / f"{system}__{mode}__s{seed}.csv", index=False)
    (preds / "RUN_INFO.json").write_text(json.dumps({
        "systems": list(systems),
        "seeds": list(seeds),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }))
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(preds.glob("*.csv"))
    ]
    (preds / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return preds


def _rewrite_sums(preds):
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(preds.glob("*.csv"))
    ]
    (preds / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


def _write_run_info(preds, name, systems, seeds, corpus_path):
    (preds / name).write_text(json.dumps({
        "systems": list(systems),
        "seeds": list(seeds),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }))


def test_perfect_and_inverted_auroc():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert ec.compute_metrics(y, scores, 0.5)["auroc"] == pytest.approx(1.0)
    assert ec.compute_metrics(y, 1.0 - scores, 0.5)["auroc"] == pytest.approx(0.0)


def test_draw_cluster_indices_keeps_whole_prompts():
    prompt_ids = np.repeat(["p0", "p1", "p2", "p3"], 10)
    groups = {
        prompt: np.flatnonzero(prompt_ids == prompt)
        for prompt in pd.unique(prompt_ids)
    }
    drawn = np.array(["p0", "p2", "p2"])
    idx = ec.draw_cluster_indices(drawn, groups)

    assert idx.size == 30
    for prompt in np.unique(prompt_ids[idx]):
        actual = set(idx[prompt_ids[idx] == prompt])
        expected = set(np.flatnonzero(prompt_ids == prompt))
        assert actual == expected


def test_bootstrap_uses_subset_prompts(monkeypatch, tmp_path):
    per_prompt = 4
    n_prompts = 6
    core = np.array([True] * (3 * per_prompt) + [False] * (3 * per_prompt))
    labels = np.array([0, 1] * (n_prompts * per_prompt // 2), dtype=int)
    corpus, corpus_path = _write_corpus(
        tmp_path, n_prompts=n_prompts, per_prompt=per_prompt,
        labels=labels, core=core,
    )
    preds = _write_preds(tmp_path, corpus, corpus_path)
    real = ec.bootstrap_prompt_ids
    calls = []

    def wrapped(prompt_ids, rng):
        result = real(prompt_ids, rng)
        calls.append((tuple(pd.unique(prompt_ids)), tuple(result)))
        return result

    monkeypatch.setattr(ec, "bootstrap_prompt_ids", wrapped)
    ec.run(
        preds, corpus_path, tmp_path / "out", n_boot=5, expect_seeds=(13,)
    )

    core_set = {"p0", "p1", "p2"}
    other_set = {"p3", "p4", "p5"}
    all_set = core_set | other_set
    assert len(calls) == 6 * 5
    for inputs, draws in calls:
        assert set(inputs) in (core_set, other_set, all_set)
        if set(inputs) == core_set:
            assert len(draws) == 3
            assert set(draws) <= core_set
        elif set(inputs) == other_set:
            assert len(draws) == 3
            assert set(draws) <= other_set
        else:
            assert len(draws) == 6


def test_run_bootstrap_uses_complete_prompt_groups(monkeypatch, tmp_path):
    per_prompt = 4
    n_prompts = 6
    core = np.array([True] * (3 * per_prompt) + [False] * (3 * per_prompt))
    corpus, corpus_path = _write_corpus(
        tmp_path, n_prompts=n_prompts, per_prompt=per_prompt, core=core,
    )
    preds = _write_preds(tmp_path, corpus, corpus_path)
    real = ec.draw_cluster_indices
    calls = []

    def wrapped(drawn_prompts, groups):
        result = real(drawn_prompts, groups)
        calls.append((tuple(drawn_prompts), groups, result))
        return result

    monkeypatch.setattr(ec, "draw_cluster_indices", wrapped)
    ec.run(preds, corpus_path, tmp_path / "out", n_boot=5, expect_seeds=(13,))

    assert calls
    prompt_ids = corpus["prompt_id"].to_numpy()
    subset_positions = {}
    for subset, mask in ec._subsets(corpus).items():
        subset_prompt_ids = prompt_ids[mask]
        subset_positions[subset] = {
            prompt: np.flatnonzero(subset_prompt_ids == prompt)
            for prompt in pd.unique(subset_prompt_ids)
        }
    for drawn, groups, result in calls:
        expected = set()
        for prompt in drawn:
            if prompt in groups:
                expected.update(groups[prompt].tolist())
        assert set(result.tolist()) == expected

        matching = [
            positions for positions in subset_positions.values()
            if set(positions) == set(groups)
        ]
        assert matching
        for prompt, rows in groups.items():
            assert set(rows.tolist()) == set(matching[0][prompt].tolist())


def test_permuted_labels_auroc_near_half(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 96, dtype=int)
    rng.shuffle(labels)
    corpus, corpus_path = _write_corpus(tmp_path, labels=labels)
    preds = _write_preds(tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99))
    out = tmp_path / "out"

    metrics, paired, info = ec.run(preds, corpus_path, out, n_boot=30, seed=13)

    rows = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "auroc")
    ]
    assert len(rows) == 6
    assert abs(float(rows["estimate"].mean()) - 0.5) < 0.1
    assert {"system", "mode", "subset", "gold", "metric", "estimate",
            "ci_low", "ci_high", "n_boot_rejected",
            "n_prompts_subset", "n_prompts_boot"} <= set(metrics.columns)
    assert "n_dropped" not in metrics.columns
    assert "n_prompts" not in metrics.columns
    assert (rows["n_prompts_subset"] == 24).all()
    assert (rows["n_prompts_boot"] == 24).all()
    assert {"comparison", "mode", "difference", "ci_low", "ci_high",
            "p", "p_holm", "n_boot_rejected",
            "n_prompts_subset", "n_prompts_boot"} <= set(paired.columns)
    assert any(column.startswith("mcnemar_p_s") for column in paired.columns)
    assert paired["p_holm"].notna().all()
    assert len(paired) == 4

    assert (out / "corpus_metrics.csv").exists()
    assert (out / "corpus_paired.csv").exists()
    info_files = list(out.glob("corpus_eval_INFO_*.json"))
    assert len(info_files) == 1
    on_disk = json.loads(info_files[0].read_text())
    assert on_disk == info
    assert {"versions", "seed", "n_boot", "corpus_sha256", "git_commit"} <= set(on_disk)
    assert on_disk["seed"] == 13
    assert on_disk["n_boot"] == 30


def test_permuted_gold_auroc_200_games():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    base = np.array([0, 1] * 100, dtype=int)
    values = [
        ec.compute_metrics(rng.permutation(base), scores, 0.5)["auroc"]
        for _ in range(200)
    ]
    assert 0.45 <= float(np.mean(values)) <= 0.55


def test_holm_known_values():
    pvalues = np.array([0.01, 0.04, 0.03, 0.02])
    assert np.allclose(ec.holm(pvalues), [0.04, 0.06, 0.06, 0.06])


def test_checksum_and_missing_score_errors(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)
    out = tmp_path / "out"

    (corpus_path.parent / "SHA256SUMS").write_text(
        "0" * 64 + "  biasdetect_corpus.csv\n"
    )
    with pytest.raises(ValueError, match="sha256"):
        ec.run(preds, corpus_path, out, n_boot=10, expect_seeds=(13,))
    assert not out.exists()

    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (corpus_path.parent / "SHA256SUMS").write_text(
        f"{digest}  biasdetect_corpus.csv\n"
    )
    (preds / "P_sym__sentmean__s13.csv").unlink()
    with pytest.raises(FileNotFoundError, match="manquant"):
        ec.run(preds, corpus_path, out, n_boot=10, expect_seeds=(13,))
    assert not out.exists()


def test_run_info_detects_absent_system(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path,
        systems=("P_fusion_minilm", "P_neu_minilm"),
    )
    info = json.loads((preds / "RUN_INFO.json").read_text())
    info["systems"] = ["P_fusion_minilm", "P_neu_minilm", "P_sym"]
    (preds / "RUN_INFO.json").write_text(json.dumps(info))

    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="manquant"):
        ec.run(preds, corpus_path, out, n_boot=10, expect_seeds=(13,))
    assert not out.exists()


def test_run_info_corpus_hash_mismatch(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)
    info = json.loads((preds / "RUN_INFO.json").read_text())
    info["corpus_sha256"] = "0" * 64
    (preds / "RUN_INFO.json").write_text(json.dumps(info))

    with pytest.raises(ValueError, match="hash du corpus"):
        ec.run(preds, corpus_path, tmp_path / "out", n_boot=10, expect_seeds=(13,))


def test_score_sha256sums_mismatch(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)
    path = preds / "P_sym__full__s13.csv"
    frame = pd.read_csv(path)
    frame["score"] = frame["score"] + 1.0
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="sha256"):
        ec.run(preds, corpus_path, tmp_path / "out", n_boot=10, expect_seeds=(13,))


def test_score_sha256sums_missing(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)
    (preds / "SHA256SUMS").unlink()

    with pytest.raises(FileNotFoundError, match="SHA256SUMS"):
        ec.run(preds, corpus_path, tmp_path / "out", n_boot=10, expect_seeds=(13,))


def test_eval_info_timestamp_and_no_overwrite(tmp_path, monkeypatch):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)
    out = tmp_path / "out"

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(ec, "datetime", FixedDateTime)
    ec.run(preds, corpus_path, out, n_boot=5, expect_seeds=(13,))

    info_path = out / "corpus_eval_INFO_2026-01-02_03-04-05.json"
    assert info_path.exists()
    with pytest.raises(FileExistsError, match="INFO"):
        ec.run(preds, corpus_path, out, n_boot=5, expect_seeds=(13,))


def test_ece_out_of_range_written_as_na(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=6, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, score_scale=0.0, score_shift=2.0
    )
    metrics, _, _ = ec.run(
        preds, corpus_path, tmp_path / "out", n_boot=5, expect_seeds=(13,)
    )

    rows = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "ece")
    ]
    assert (rows["estimate"] == "n/a").all()
    assert (rows["ci_low"] == "n/a").all()
    assert (rows["ci_high"] == "n/a").all()

    auroc = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "auroc")
    ]
    assert auroc["estimate"].notna().all()
    assert ec.compute_metrics([0, 1, 0, 1], [2.0, 2.0, 2.0, 2.0], 0.5)["ece"] == "n/a"


def test_single_class_subset_returns_nan(tmp_path):
    n = 48
    core = np.array([True] * 24 + [False] * 24)
    labels = np.array([0, 1] * 12 + [1] * 24, dtype=int)
    corpus, corpus_path = _write_corpus(
        tmp_path, n_prompts=8, per_prompt=6, labels=labels, core=core
    )
    preds = _write_preds(tmp_path, corpus, corpus_path)
    out = tmp_path / "out"

    metrics, _, _ = ec.run(preds, corpus_path, out, n_boot=10, expect_seeds=(13,))

    rows = metrics[
        (metrics["subset"] == "other") & (metrics["gold"] == "gold_raw")
    ]
    assert len(rows) == 3 * 2 * 3
    assert rows["estimate"].isna().all()
    assert rows["ci_low"].isna().all()
    assert rows["ci_high"].isna().all()
    assert (rows["n_boot_rejected"] == 10).all()

    main = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "auroc")
    ]
    assert main["estimate"].notna().all()


def test_multiple_run_info_merges_disjoint_seeds(tmp_path):
    systems = ("P_fusion_minilm", "P_neu_minilm", "P_sym")
    seeds = (13, 21, 42, 7, 99)
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path, systems=systems, seeds=seeds)
    (preds / "RUN_INFO.json").unlink()
    _write_run_info(
        preds, "RUN_INFO_20260101-000000.json", systems, (13,), corpus_path
    )
    _write_run_info(
        preds, "RUN_INFO_20260102-000000.json", systems, (21, 42, 7, 99), corpus_path
    )

    metrics, _, info = ec.run(preds, corpus_path, tmp_path / "out", n_boot=5)

    assert (metrics["n_seeds"] == 5).all()
    assert info["run_info_files"] == [
        "RUN_INFO_20260101-000000.json",
        "RUN_INFO_20260102-000000.json",
    ]
    assert info["n_score_files"] == len(systems) * 2 * len(seeds)
    sums = preds / "SHA256SUMS"
    assert info["preds_sha256sums_sha256"] == hashlib.sha256(
        sums.read_bytes()
    ).hexdigest()


def test_duplicate_seed_across_run_info_errors(tmp_path):
    systems = ("P_fusion_minilm", "P_neu_minilm", "P_sym")
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path, systems=systems, seeds=(13, 21))
    (preds / "RUN_INFO.json").unlink()
    _write_run_info(
        preds, "RUN_INFO_20260101-000000.json", systems, (13,), corpus_path
    )
    _write_run_info(
        preds, "RUN_INFO_20260102-000000.json", systems, (13, 21), corpus_path
    )

    with pytest.raises(ValueError, match="seed"):
        ec.run(preds, corpus_path, tmp_path / "out", n_boot=5)


def test_unlisted_csv_errors(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )
    (preds / "stray.csv").write_text("id,prompt_id,score,threshold\n")

    with pytest.raises(ValueError, match="non attendu"):
        ec.run(preds, corpus_path, tmp_path / "out", n_boot=5)


def test_expect_seeds_mismatch_errors(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )

    with pytest.raises(ValueError, match="expect-seeds"):
        ec.run(
            preds, corpus_path, tmp_path / "out", n_boot=5,
            expect_seeds=(13, 21),
        )


def test_bootstrap_p_value_never_zero():
    assert ec._bootstrap_p_value(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.5)
    assert ec._bootstrap_p_value(np.array([-0.1, -0.2, -0.3])) == pytest.approx(0.5)
    assert ec._bootstrap_p_value(np.array([0.0, 0.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["corpus_metrics.csv", "corpus_paired.csv"])
def test_outputs_refuse_overwrite_before_evaluate(monkeypatch, tmp_path, name):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text("existing\n")
    called = []
    real = ec.evaluate

    def wrapped(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(ec, "evaluate", wrapped)
    with pytest.raises(FileExistsError, match=name):
        ec.run(preds, corpus_path, out, n_boot=5)
    assert not called


def test_overwrite_allows_existing_outputs(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )
    out = tmp_path / "out"

    ec.run(preds, corpus_path, out, n_boot=5)
    ec.run(preds, corpus_path, out, n_boot=5, overwrite=True)

    assert (out / "corpus_metrics.csv").exists()
    assert (out / "corpus_paired.csv").exists()


def test_ece_na_decided_from_full_subset(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=6, per_prompt=4)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )
    for path in preds.glob("*.csv"):
        frame = pd.read_csv(path)
        frame["score"] = 0.5
        frame.loc[frame["prompt_id"] == "p0", "score"] = 2.0
        frame.to_csv(path, index=False)
    _rewrite_sums(preds)

    metrics, _, _ = ec.run(preds, corpus_path, tmp_path / "out", n_boot=20)

    rows = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "ece")
    ]
    assert len(rows) == 3 * 2
    assert (rows["estimate"] == "n/a").all()
    assert (rows["ci_low"] == "n/a").all()
    assert (rows["ci_high"] == "n/a").all()


def test_permute_gold_writes_elsewhere(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=24, per_prompt=8)
    preds = _write_preds(
        tmp_path, corpus, corpus_path, seeds=(13, 21, 42, 7, 99)
    )
    out = tmp_path / "permuted"

    metrics, _, info = ec.run(
        preds, corpus_path, out, n_boot=10, permute_gold=7
    )

    rows = metrics[
        (metrics["subset"] == "core_norefusal")
        & (metrics["gold"] == "gold_raw")
        & (metrics["metric"] == "auroc")
    ]
    assert 0.4 <= float(rows["estimate"].mean()) <= 0.6
    assert info["permuted"] is True
    assert info["permute_gold"] == 7


def test_permute_gold_refuses_default_results(tmp_path):
    corpus, corpus_path = _write_corpus(tmp_path, n_prompts=4, per_prompt=4)
    preds = _write_preds(tmp_path, corpus, corpus_path)

    with pytest.raises(ValueError, match="results"):
        ec.run(
            preds, corpus_path, ec.ROOT / "results", n_boot=5, permute_gold=7
        )
